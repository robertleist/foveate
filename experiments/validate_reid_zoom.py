"""Validate the re-identification score ``g`` across zoom levels.

For each held-out target object we build a series of square crops centred on it, sweeping from
zoomed-OUT (object small, lots of surrounding context) through well-framed to OVER-zoomed (the crop
sits *inside* the object, only a fragment visible). Each crop is scored with every ``reid_mode``
against an exemplar bank of *other* instances of the same class (built from disjoint images), and we
check the hypothesis: **g rises as the object comes into frame and falls once we over-zoom past it**,
peaking near the framing the exemplars themselves carry (tight bbox + ``pad_frac``).

The masked modes (mean / kmeans / full) score the object's foreground patches, which here come from
the ground-truth target mask projected into each crop — so this isolates the *scorer*, not the Where
stage. ``cls`` scores the whole-crop CLS token.

Run:  .venv/Scripts/python.exe experiments/validate_reid_zoom.py --dataset corals_coco
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")                # Windows console is cp1252 by default
except Exception:
    pass

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from foveate import Config, DINOv3Backbone, features as featlib
from foveate.foreground import build_extractor
from foveate.reid import build_reid_scorer

MODES = ["cls", "mean", "kmeans", "full"]
# Square crop side as a multiple of the object's larger bbox dimension. <1 = over-zoom (crop inside
# the object); 1 = tight bbox; >1 = zoomed out with context.
ZOOMS = np.array([0.35, 0.45, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0])
IN_FRAME = (0.85, 2.0)                      # zoom band we call "object well in frame"


def poly_to_mask(seg, h: int, w: int) -> np.ndarray:
    """Rasterize a COCO polygon list into a bool mask."""
    m = np.zeros((h, w), np.uint8)
    for poly in seg:
        pts = np.array(poly, np.float64).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def load_instances(ds_dir: Path):
    """All instances as dicts {file, h, w, bbox, area, seg}, grouped is left to the caller."""
    d = json.load(open(ds_dir / "annotations.json"))
    by_id = {im["id"]: im for im in d["images"]}
    out = []
    for a in d["annotations"]:
        if a.get("iscrowd") or not isinstance(a.get("segmentation"), list):
            continue
        im = by_id[a["image_id"]]
        out.append(dict(image_id=a["image_id"], file=im["file_name"], h=im["height"],
                        w=im["width"], bbox=a["bbox"], area=float(a.get("area", 0.0)),
                        seg=a["segmentation"]))
    return out


def square_crop(img: np.ndarray, mask: np.ndarray, cx: float, cy: float, side: float):
    """Square-ish crop of ``img`` (and matching ``mask`` slice) centred at (cx, cy), clamped."""
    H, W = img.shape[:2]
    half = side / 2.0
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = max(0, y0), min(H, y1)
    x0, x1 = max(0, x0), min(W, x1)
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="corals_coco")
    ap.add_argument("--n-exemplars", type=int, default=6)
    ap.add_argument("--n-targets", type=int, default=14)
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--min-area", type=float, default=40000.0)
    ap.add_argument("--foreground", choices=["gt", "predict"], default="gt",
                    help="masked-mode foreground: gt = ground-truth target mask (isolates the "
                         "scorer); predict = the Where extractor's mask (the live-cascade setting).")
    ap.add_argument("--debias", action="store_true",
                    help="positional debias for the masked family + the Where extractor.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = f"{args.foreground}_{'debias' if args.debias else 'nodebias'}"
    root = Path(__file__).resolve().parent.parent
    ds_dir = root / "datasets" / args.dataset
    img_dir = ds_dir / "images"
    out_dir = root / "experiments" / "reid_zoom_out"; out_dir.mkdir(exist_ok=True)
    out_png = Path(args.out) if args.out else (out_dir / f"reid_zoom_{args.dataset}_{tag}.png")

    rng = random.Random(args.seed)
    insts = [i for i in load_instances(ds_dir) if i["area"] >= args.min_area]
    rng.shuffle(insts)

    # Exemplars from ONE image, targets from the OTHERS → a genuine cross-image re-id test that also
    # works on datasets with only a handful of images (polyps has 4).
    by_img: dict[int, list] = {}
    for it in insts:
        by_img.setdefault(it["image_id"], []).append(it)
    ex_img_id = max(by_img, key=lambda k: len(by_img[k]))
    ex_insts = by_img[ex_img_id][:args.n_exemplars]
    tgt_insts = [it for it in insts if it["image_id"] != ex_img_id][:args.n_targets]
    print(f"{args.dataset}: {len(insts)} instances ≥{args.min_area:.0f}px | "
          f"{len(ex_insts)} exemplars, {len(tgt_insts)} targets")

    cfg = Config(reid_mode="cls", debias=args.debias)      # image_size is a backbone arg, not a Config field
    backbone = DINOv3Backbone(image_size=args.image_size)
    print(f"backbone on {backbone.device} | foreground={args.foreground} debias={args.debias}")

    _img_cache: dict[str, np.ndarray] = {}

    def load_img(file: str) -> np.ndarray:
        if file not in _img_cache:
            from PIL import Image
            _img_cache[file] = np.array(Image.open(img_dir / file).convert("RGB"))
        return _img_cache[file]

    # --- exemplar bank per mode (full-image + full-mask; build_reid_scorer crops each) ---
    ex_images = [load_img(it["file"]) for it in ex_insts]
    ex_masks = [poly_to_mask(it["seg"], it["h"], it["w"]) for it in ex_insts]
    scorers = {}
    for mode in MODES:
        c = Config(reid_mode=mode, reid_kmeans_k=cfg.reid_kmeans_k, debias=args.debias)
        scorers[mode] = build_reid_scorer(c, backbone, ex_images, ex_masks)
    print("built scorers:", ", ".join(MODES))

    # The live-cascade foreground: the Where extractor's mask, not the ground truth.
    extractor = None
    if args.foreground == "predict":
        extractor = build_extractor(cfg)
        extractor.set_reference(backbone, ex_images, ex_masks, None, cfg)
        print("built Where extractor (insid3) for predict-foreground")

    # --- score every target crop at every zoom ---
    G = {m: np.full((len(tgt_insts), len(ZOOMS)), np.nan) for m in MODES}
    for ti, it in enumerate(tgt_insts):
        img = load_img(it["file"])
        mask = poly_to_mask(it["seg"], it["h"], it["w"])
        x, y, w, h = it["bbox"]
        cx, cy, L = x + w / 2.0, y + h / 2.0, max(w, h)
        crops, mcrops, keep = [], [], []
        for zi, s in enumerate(ZOOMS):
            crop, mcrop = square_crop(img, mask, cx, cy, s * L)
            if crop.shape[0] < 16 or crop.shape[1] < 16:
                continue
            crops.append(crop); mcrops.append(mcrop); keep.append(zi)
        embedded = featlib.embed_batch(backbone, crops, chunk=8, standardize=cfg.standardize)
        for zi, (feat, cls), mcrop in zip(keep, embedded, mcrops):
            if extractor is not None:
                fg = extractor.predict(feat, cls=cls).foreground   # live Where mask
            else:
                fg = featlib.resize_mask_to_grid(mcrop, feat.shape[:2])   # ground-truth mask
            for mode in MODES:
                G[mode][ti, zi] = scorers[mode].score(feat, cls, fg)
        print(f"  target {ti + 1}/{len(tgt_insts)} ({it['file'][:24]}…) done")

    # --- aggregate: per-target min-max normalize each mode's curve, then average across targets ---
    def norm_rows(A):
        lo = np.nanmin(A, axis=1, keepdims=True)
        hi = np.nanmax(A, axis=1, keepdims=True)
        return (A - lo) / np.clip(hi - lo, 1e-9, None)

    summary = {}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mode in MODES:
        Araw = G[mode]
        An = norm_rows(Araw)
        mean, std = np.nanmean(An, axis=0), np.nanstd(An, axis=0)
        ax.plot(ZOOMS, mean, marker="o", label=mode)
        ax.fill_between(ZOOMS, mean - std, mean + std, alpha=0.12)
        peak_s = ZOOMS[np.nanargmax(Araw, axis=1)]                       # per-target peak zoom
        in_frame = np.mean((peak_s >= IN_FRAME[0]) & (peak_s <= IN_FRAME[1]))
        # over-zoom penalty: mean normalized drop from each target's peak to its most-over-zoomed crop
        overzoom_drop = np.nanmean(np.nanmax(An, axis=1) - An[:, 0])
        zoomout_drop = np.nanmean(np.nanmax(An, axis=1) - An[:, -1])
        summary[mode] = dict(median_peak=float(np.median(peak_s)), in_frame=float(in_frame),
                             overzoom_drop=float(overzoom_drop), zoomout_drop=float(zoomout_drop))

    ax.axvspan(*IN_FRAME, color="green", alpha=0.06, label="'in frame' band")
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(ZOOMS); ax.set_xticklabels([f"{s:g}" for s in ZOOMS], fontsize=7)
    ax.set_xlabel("zoom  (crop side / object size)   ←over-zoom · in-frame · zoomed-out→")
    ax.set_ylabel("g  (per-target min-max normalized, mean±std)")
    ax.set_title(f"re-id score g vs zoom — {args.dataset}  [{tag}]  "
                 f"({len(tgt_insts)} targets, {len(ex_insts)} exemplars)")
    ax.legend(loc="lower center", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130)

    print("\n=== summary (want: peak in-frame, positive over-zoom drop) ===")
    print(f"{'mode':8} {'median_peak_s':>13} {'%peak_in_frame':>15} "
          f"{'overzoom_drop':>14} {'zoomout_drop':>13}")
    for mode in MODES:
        s = summary[mode]
        print(f"{mode:8} {s['median_peak']:>13.2f} {100 * s['in_frame']:>14.0f}% "
              f"{s['overzoom_drop']:>14.2f} {s['zoomout_drop']:>13.2f}")
    print(f"\nsaved plot → {out_png}")


if __name__ == "__main__":
    main()
