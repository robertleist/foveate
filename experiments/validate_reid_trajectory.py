"""Operating-regime evaluation of the re-id score ``g`` — fair to foveate's coarse-to-fine design.

The zoom sweep scores *absolute* crops; foveate instead consumes ``g`` as a *relative* signal on the
Where extractor's **coarse predicted** foreground. This script measures that:

1. **Refinement monotonicity (the survivor rule).** Along a coarse→fine chain centred on a target,
   does ``g`` rise as the crop resolves a single instance (``g(child) > g(parent)``) and fall once we
   over-zoom past it?
2. **Accept / reject.** Does ``g`` separate a concept crop from (a) background and (b) a *different*
   concept (the other dataset)? Reported as ROC-AUC per mode — the numbers to quote in the paper.

Five ``g`` variants (see :mod:`reid_eval_common`), always on the predicted foreground.

Run:  .venv/Scripts/python.exe experiments/validate_reid_trajectory.py --dataset corals_coco
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from foveate import Config, DINOv3Backbone, features as featlib
from foveate.foreground import build_extractor
from experiments.reid_eval_common import (
    ZOOMS, PLOT_MODES, MODE_KEYS, DATASET_TITLE, OTHER,
    poly_to_mask, square_crop, load_instances, build_bank, score_modes,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ANCHOR = int(np.argmin(np.abs(ZOOMS - 1.0)))     # tight-bbox index: refine above it, over-zoom below
POS_ZOOM = int(np.argmin(np.abs(ZOOMS - 1.15)))  # a "well-framed instance" crop for accept/reject
METRICS = ["refine_rise", "overzoom_fall", "auroc_bg", "auroc_xconcept"]


def _score_crops(bank, extractor, backbone, cfg, crops):
    out = {k: [] for k in MODE_KEYS}
    for feat, cls in featlib.embed_batch(backbone, crops, chunk=8, standardize=cfg.standardize):
        fg = extractor.predict(feat, cls=cls).foreground
        vals = score_modes(bank, feat, cls, fg)
        for k in MODE_KEYS:
            out[k].append(vals[k])
    return out


def _sample_background(img, union, n, L, rng):
    H, W = img.shape[:2]
    out, tries = [], 0
    while len(out) < n and tries < n * 60:
        tries += 1
        s = rng.uniform(1.0, 2.0) * L
        if s >= min(H, W):
            continue
        cx, cy = rng.uniform(s / 2, W - s / 2), rng.uniform(s / 2, H - s / 2)
        crop, mcrop = square_crop(img, union, cx, cy, s)
        if crop.shape[0] >= 16 and crop.shape[1] >= 16 and mcrop.mean() < 0.05:
            out.append(crop)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="corals_coco")
    ap.add_argument("--n-exemplars", type=int, default=6)
    ap.add_argument("--n-targets", type=int, default=40)
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--min-area", type=float, default=40000.0)
    ap.add_argument("--other-min-area", type=float, default=2500.0)
    ap.add_argument("--debias", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    ds_dir = root / "datasets" / args.dataset
    img_dir = ds_dir / "images"
    tag = "debias" if args.debias else "nodebias"
    out_dir = root / "experiments" / "reid_zoom_out"; out_dir.mkdir(exist_ok=True)
    stem = out_dir / f"reid_traj_{args.dataset}_{tag}"

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
          f"foreground=predict debias={args.debias}")

    cfg = Config(reid_mode="cls", debias=args.debias)
    backbone = DINOv3Backbone(image_size=args.image_size)
    _cache: dict[str, np.ndarray] = {}

    def load_img(dir_, file):
        key = f"{dir_}/{file}"
        if key not in _cache:
            _cache[key] = np.array(Image.open(dir_ / file).convert("RGB"))
        return _cache[key]

    ex_images = [load_img(img_dir, it["file"]) for it in ex_insts]
    ex_masks = [poly_to_mask(it["seg"], it["h"], it["w"]) for it in ex_insts]
    bank = build_bank(backbone, ex_images, ex_masks, debias=args.debias)
    extractor = build_extractor(cfg)
    extractor.set_reference(backbone, ex_images, ex_masks, None, cfg)
    print("built bank + Where extractor")

    # ---- 1. refinement trajectory + collect well-framed positives ----
    G = {k: np.full((len(tgt_insts), len(ZOOMS)), np.nan) for k in MODE_KEYS}
    pos = {k: [] for k in MODE_KEYS}
    for ti, it in enumerate(tgt_insts):
        img = load_img(img_dir, it["file"])
        x, y, w, h = it["bbox"]
        cx, cy, L = x + w / 2.0, y + h / 2.0, max(w, h)
        crops, keep = [], []
        for zi, s in enumerate(ZOOMS):
            crop, _ = square_crop(img, np.zeros(img.shape[:2], bool), cx, cy, s * L)
            if crop.shape[0] >= 16 and crop.shape[1] >= 16:
                crops.append(crop); keep.append(zi)
        scored = _score_crops(bank, extractor, backbone, cfg, crops)
        for k in MODE_KEYS:
            for zi, val in zip(keep, scored[k]):
                G[k][ti, zi] = val
            pos[k].append(G[k][ti, POS_ZOOM])
        if (ti + 1) % 5 == 0 or ti + 1 == len(tgt_insts):
            print(f"  scored {ti + 1}/{len(tgt_insts)} targets")

    # ---- 2. accept/reject negatives ----
    all_by_img: dict[int, list] = {}
    for it in load_instances(ds_dir):
        all_by_img.setdefault(it["image_id"], []).append(it)
    bg_crops = []
    for it in tgt_insts[:12]:
        img = load_img(img_dir, it["file"])
        union = np.zeros(img.shape[:2], bool)
        for jt in all_by_img[it["image_id"]]:
            union |= poly_to_mask(jt["seg"], jt["h"], jt["w"])
        bg_crops += _sample_background(img, union, 4, max(it["bbox"][2], it["bbox"][3]), rng)
    bg = _score_crops(bank, extractor, backbone, cfg, bg_crops) if bg_crops else {k: [] for k in MODE_KEYS}

    other = OTHER[args.dataset]
    other_dir = root / "datasets" / other / "images"
    o_insts = [i for i in load_instances(root / "datasets" / other) if i["area"] >= args.other_min_area]
    rng.shuffle(o_insts)
    xc_crops = []
    for it in o_insts[:args.n_targets]:
        img = load_img(other_dir, it["file"])
        x, y, w, h = it["bbox"]
        crop, _ = square_crop(img, np.zeros(img.shape[:2], bool),
                              x + w / 2.0, y + h / 2.0, 1.15 * max(w, h))
        if crop.shape[0] >= 16 and crop.shape[1] >= 16:
            xc_crops.append(crop)
    xc = _score_crops(bank, extractor, backbone, cfg, xc_crops)
    print(f"  background crops: {len(bg_crops)} | cross-concept ({other}) crops: {len(xc_crops)}")

    # ---- metrics ----
    def auroc(p, n):
        p = [v for v in p if np.isfinite(v)]; n = [v for v in n if np.isfinite(v)]
        return roc_auc_score([1] * len(p) + [0] * len(n), p + n) if p and n else np.nan

    rows = {}
    for k in MODE_KEYS:
        A = G[k]
        rise = [np.mean([A[ti, zi] > A[ti, zi + 1] for zi in range(ANCHOR, len(ZOOMS) - 1)
                         if np.isfinite(A[ti, zi]) and np.isfinite(A[ti, zi + 1])] or [np.nan])
                for ti in range(A.shape[0])]
        fall = [np.mean([A[ti, zi] < A[ti, zi + 1] for zi in range(0, ANCHOR)
                         if np.isfinite(A[ti, zi]) and np.isfinite(A[ti, zi + 1])] or [np.nan])
                for ti in range(A.shape[0])]
        rows[k] = dict(refine_rise=float(np.nanmean(rise)), overzoom_fall=float(np.nanmean(fall)),
                       auroc_bg=auroc(pos[k], bg[k]), auroc_xconcept=auroc(pos[k], xc[k]))

    label = {m["key"]: m["label"] for m in PLOT_MODES}
    print("\n=== g as a cascade control signal (predict foreground; all want HIGH, ~[0.5,1]) ===")
    print(f"{'mode':14}" + "".join(f"{m:>16}" for m in METRICS))
    for k in MODE_KEYS:
        print(f"{label[k]:14}" + "".join(f"{rows[k][m]:>16.2f}" for m in METRICS))

    # supplementary bar chart (the numbers are the deliverable; this just visualizes them)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xpos = np.arange(len(MODE_KEYS)); wdt = 0.2
    mcolors = ["#2a78d6", "#eda100", "#1baf7a", "#4a3aa7"]
    for j, m in enumerate(METRICS):
        ax.bar(xpos + (j - 1.5) * wdt, [rows[k][m] for k in MODE_KEYS], wdt,
               color=mcolors[j], label=m.replace("_", " "))
    ax.axhline(0.5, color="#9a9a95", ls=(0, (4, 3)), lw=1.0)
    ax.set_xticks(xpos); ax.set_xticklabels([label[k] for k in MODE_KEYS])
    ax.set_ylim(0, 1.06); ax.set_ylabel("score (higher = better)")
    ax.set_title(f"{DATASET_TITLE.get(args.dataset, args.dataset)} — g as a cascade control signal",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=9, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(True, axis="y", color="#000000", alpha=0.06)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    print(f"\nsaved → {stem}.pdf / .png")


if __name__ == "__main__":
    main()
