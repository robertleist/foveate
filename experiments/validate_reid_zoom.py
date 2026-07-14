"""Re-identification score ``g`` vs. relative crop scale — the publication line-curve figure.

For each held-out target object a series of square crops centred on it sweeps from over-zoom (crop
inside the object, ``s<1``) through the tight bounding box (``s=1``) to zoomed-out context
(``s=3``). Every crop is scored with the five ``g`` variants (see :mod:`reid_eval_common`) against a
cross-image exemplar bank, and the per-object curves are min-max normalized and averaged. It
visualizes the core claim: ``g`` rises as the object comes into frame and falls once we over-zoom.

The masked modes read the extracted foreground: ``--foreground gt`` uses the ground-truth target
mask (isolates the scorer); ``--foreground predict`` uses the Where extractor's mask (the operating
regime). The unmasked / CLS modes never use a mask.

Run:  .venv/Scripts/python.exe experiments/validate_reid_zoom.py --dataset corals_coco --foreground predict
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np
from PIL import Image

from foveate import Config, DINOv3Backbone, features as featlib
from foveate.foreground import build_extractor
from experiments.reid_eval_common import (
    ZOOMS, IN_FRAME, PLOT_MODES, MODE_KEYS, DATASET_TITLE,
    poly_to_mask, square_crop, load_instances, build_bank, score_modes,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TICKS = [0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="corals_coco")
    ap.add_argument("--n-exemplars", type=int, default=6)
    ap.add_argument("--n-targets", type=int, default=40)
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--min-area", type=float, default=40000.0)
    ap.add_argument("--foreground", choices=["gt", "predict"], default="predict")
    ap.add_argument("--debias", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = f"{args.foreground}_{'debias' if args.debias else 'nodebias'}"
    root = Path(__file__).resolve().parent.parent
    ds_dir = root / "datasets" / args.dataset
    img_dir = ds_dir / "images"
    out_dir = root / "experiments" / "reid_zoom_out"; out_dir.mkdir(exist_ok=True)
    stem = Path(args.out) if args.out else (out_dir / f"reid_zoom_{args.dataset}_{tag}")

    rng = random.Random(args.seed)
    insts = [i for i in load_instances(ds_dir) if i["area"] >= args.min_area]
    rng.shuffle(insts)
    by_img: dict[int, list] = {}
    for it in insts:
        by_img.setdefault(it["image_id"], []).append(it)
    ex_img_id = max(by_img, key=lambda k: len(by_img[k]))
    ex_insts = by_img[ex_img_id][:args.n_exemplars]
    tgt_insts = [it for it in insts if it["image_id"] != ex_img_id][:args.n_targets]
    print(f"{args.dataset}: {len(ex_insts)} exemplars, {len(tgt_insts)} targets | "
          f"foreground={args.foreground} debias={args.debias}")

    cfg = Config(reid_mode="cls", debias=args.debias)
    backbone = DINOv3Backbone(image_size=args.image_size)
    _cache: dict[str, np.ndarray] = {}

    def load_img(file):
        if file not in _cache:
            _cache[file] = np.array(Image.open(img_dir / file).convert("RGB"))
        return _cache[file]

    ex_images = [load_img(it["file"]) for it in ex_insts]
    ex_masks = [poly_to_mask(it["seg"], it["h"], it["w"]) for it in ex_insts]
    bank = build_bank(backbone, ex_images, ex_masks, debias=args.debias)
    extractor = None
    if args.foreground == "predict":
        extractor = build_extractor(cfg)
        extractor.set_reference(backbone, ex_images, ex_masks, None, cfg)
    print("built bank" + (" + Where extractor" if extractor else ""))

    G = {k: np.full((len(tgt_insts), len(ZOOMS)), np.nan) for k in MODE_KEYS}
    for ti, it in enumerate(tgt_insts):
        img = load_img(it["file"])
        mask = poly_to_mask(it["seg"], it["h"], it["w"])
        x, y, w, h = it["bbox"]
        cx, cy, L = x + w / 2.0, y + h / 2.0, max(w, h)
        crops, mcrops, keep = [], [], []
        for zi, s in enumerate(ZOOMS):
            crop, mcrop = square_crop(img, mask, cx, cy, s * L)
            if crop.shape[0] >= 16 and crop.shape[1] >= 16:
                crops.append(crop); mcrops.append(mcrop); keep.append(zi)
        for zi, (feat, cls), mcrop in zip(keep, featlib.embed_batch(
                backbone, crops, chunk=8, standardize=cfg.standardize), mcrops):
            if extractor is not None:
                fg = extractor.predict(feat, cls=cls).foreground
            else:
                fg = featlib.resize_mask_to_grid(mcrop, feat.shape[:2])
            vals = score_modes(bank, feat, cls, fg)
            for k in MODE_KEYS:
                G[k][ti, zi] = vals[k]
        if (ti + 1) % 5 == 0 or ti + 1 == len(tgt_insts):
            print(f"  scored {ti + 1}/{len(tgt_insts)} targets")

    _plot(G, args, tag, tgt_insts, ex_insts, stem)


def _plot(G, args, tag, tgt_insts, ex_insts, stem):
    def norm_rows(A):
        lo, hi = np.nanmin(A, axis=1, keepdims=True), np.nanmax(A, axis=1, keepdims=True)
        return (A - lo) / np.clip(hi - lo, 1e-9, None)

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.axvspan(*IN_FRAME, color="#000000", alpha=0.035, lw=0, zorder=0)
    ax.axvline(1.0, color="#9a9a95", ls=(0, (4, 3)), lw=1.0, zorder=1)
    print(f"\n{'mode':14}{'peak_s':>8}{'%in_frame':>11}{'overzoom_drop':>15}")
    for spec in PLOT_MODES:
        An = norm_rows(G[spec["key"]])
        mean = np.nanmean(An, axis=0)
        n = np.maximum(np.sum(np.isfinite(An), axis=0), 1)
        sem = np.nanstd(An, axis=0) / np.sqrt(n)                  # SEM: confidence in the mean curve
        ax.fill_between(ZOOMS, mean - sem, mean + sem, color=spec["color"], alpha=0.16, lw=0, zorder=2)
        ax.plot(ZOOMS, mean, color=spec["color"], marker=spec["marker"], ms=5, lw=2.0,
                markeredgecolor="white", markeredgewidth=0.6, label=spec["label"], zorder=3)
        peak_s = ZOOMS[np.nanargmax(G[spec["key"]], axis=1)]
        inf = np.mean((peak_s >= IN_FRAME[0]) & (peak_s <= IN_FRAME[1]))
        drop = np.nanmean(np.nanmax(An, axis=1) - An[:, 0])
        print(f"{spec['label']:14}{np.median(peak_s):>8.2f}{100 * inf:>10.0f}%{drop:>15.2f}")

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in TICKS]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(ZOOMS[0] * 0.93, ZOOMS[-1] * 1.07)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("Relative crop scale  (crop size ÷ object size)")
    ax.set_ylabel("Re-identification score $g$  (normalized)")
    for xf, txt in [(0.44, "over-zoom"), (1.28, "object in frame"), (2.65, "context")]:
        ax.text(xf, 1.045, txt, ha="center", va="top", fontsize=8.5, style="italic",
                color="#6a6a66", zorder=4)
    ax.grid(True, which="major", axis="both", color="#000000", alpha=0.06, lw=0.8)
    ax.set_title(f"{DATASET_TITLE.get(args.dataset, args.dataset)}", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, frameon=False,
              handletextpad=0.4, columnspacing=1.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    print(f"\nsaved → {stem}.pdf / .png")


if __name__ == "__main__":
    main()
