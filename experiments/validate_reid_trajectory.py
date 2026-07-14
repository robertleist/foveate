"""Operating-regime evaluation of the re-id score ``g`` — fair to foveate's coarse-to-fine design.

The zoom sweep in ``validate_reid_zoom.py`` scored *absolute* crops against a ground-truth mask. But
foveate never sees a clean instance mask: the Where stage returns a **coarse over-segmentation** and
the cascade *refines* it by zooming. So this script judges ``g`` the way the cascade actually
consumes it, always on the Where extractor's **predicted** foreground (never GT):

1. **Refinement monotonicity (the survivor rule).** Along a coarse→fine chain centred on a target,
   does ``g`` *rise* as the crop resolves a single instance out of the coarse blob (``g(child) >
   g(parent)``), and *fall* once we over-zoom past it? These, not an absolute peak, are what the
   ``_survivors`` rule relies on.
2. **Accept / reject.** Does ``g`` separate a concept crop from (a) background and (b) a *different*
   concept (the other dataset)? This is ``g``'s role as the ``crop_sim_floor`` acceptance test.
   Reported as ROC-AUC per mode.

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
from foveate.reid import build_reid_scorer
from experiments.validate_reid_zoom import (
    MODES, ZOOMS, IN_FRAME, poly_to_mask, square_crop, load_instances,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ANCHOR = int(np.argmin(np.abs(ZOOMS - 1.0)))     # tight-bbox index: refine above it, over-zoom below
POS_ZOOM = int(np.argmin(np.abs(ZOOMS - 1.15)))  # a "well-framed instance" crop for accept/reject
OTHER = {"corals_coco": "polyps_coco", "polyps_coco": "corals_coco"}


def _score_all(scorers, extractor, backbone, cfg, crops):
    """Predict the Where foreground for each crop and score it with every mode → dict[mode] -> list."""
    out = {m: [] for m in MODES}
    embedded = featlib.embed_batch(backbone, crops, chunk=8, standardize=cfg.standardize)
    for feat, cls in embedded:
        fg = extractor.predict(feat, cls=cls).foreground
        for m in MODES:
            out[m].append(scorers[m].score(feat, cls, fg))
    return out


def _sample_background(img, union, n, L, rng):
    """Square crops with < 5% target-class coverage — 'is this the concept or just substrate?'."""
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
    ap.add_argument("--n-targets", type=int, default=14)
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
    out_png = out_dir / f"reid_traj_{args.dataset}_{tag}.png"

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
    scorers = {m: build_reid_scorer(Config(reid_mode=m, debias=args.debias),
                                    backbone, ex_images, ex_masks) for m in MODES}
    extractor = build_extractor(cfg)
    extractor.set_reference(backbone, ex_images, ex_masks, None, cfg)
    print("built scorers + Where extractor")

    # ---- 1. refinement trajectory: g over the coarse->fine->overzoom chain (predict foreground) ----
    G = {m: np.full((len(tgt_insts), len(ZOOMS)), np.nan) for m in MODES}
    pos = {m: [] for m in MODES}                     # well-framed concept crops (accept/reject +)
    for ti, it in enumerate(tgt_insts):
        img = load_img(img_dir, it["file"])
        x, y, w, h = it["bbox"]
        cx, cy, L = x + w / 2.0, y + h / 2.0, max(w, h)
        crops, keep = [], []
        for zi, s in enumerate(ZOOMS):
            crop, _ = square_crop(img, np.zeros(img.shape[:2], bool), cx, cy, s * L)
            if crop.shape[0] >= 16 and crop.shape[1] >= 16:
                crops.append(crop); keep.append(zi)
        scored = _score_all(scorers, extractor, backbone, cfg, crops)
        for m in MODES:
            for zi, val in zip(keep, scored[m]):
                G[m][ti, zi] = val
            pos[m].append(G[m][ti, POS_ZOOM])            # the well-framed crop = accept/reject positive
        print(f"  target {ti + 1}/{len(tgt_insts)} done")

    # background: sample from target images against the union of ALL target-class masks in that image
    all_insts_by_img: dict[int, list] = {}
    for it in load_instances(ds_dir):
        all_insts_by_img.setdefault(it["image_id"], []).append(it)
    bg_crops = []
    for it in tgt_insts[:8]:
        img = load_img(img_dir, it["file"])
        union = np.zeros(img.shape[:2], bool)
        for jt in all_insts_by_img[it["image_id"]]:
            union |= poly_to_mask(jt["seg"], jt["h"], jt["w"])
        L = max(it["bbox"][2], it["bbox"][3])
        bg_crops += _sample_background(img, union, 4, L, rng)
    bg = _score_all(scorers, extractor, backbone, cfg, bg_crops) if bg_crops else {m: [] for m in MODES}
    print(f"  background crops: {len(bg_crops)}")

    # cross-concept distractors: well-framed crops of the OTHER dataset's objects
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
    xc = _score_all(scorers, extractor, backbone, cfg, xc_crops)
    print(f"  cross-concept ({other}) crops: {len(xc_crops)}")

    # ---- metrics ----
    def auroc(p, n):
        p = [v for v in p if np.isfinite(v)]; n = [v for v in n if np.isfinite(v)]
        if not p or not n:
            return np.nan
        return roc_auc_score([1] * len(p) + [0] * len(n), p + n)

    rows = {}
    for m in MODES:
        A = G[m]
        # refine: fraction of zoom-IN steps on s>=1.0 where g rises (child tighter than parent)
        rise = []
        for ti in range(A.shape[0]):
            steps = [A[ti, zi] > A[ti, zi + 1] for zi in range(ANCHOR, len(ZOOMS) - 1)
                     if np.isfinite(A[ti, zi]) and np.isfinite(A[ti, zi + 1])]
            if steps:
                rise.append(np.mean(steps))
        fall = []
        for ti in range(A.shape[0]):
            steps = [A[ti, zi] < A[ti, zi + 1] for zi in range(0, ANCHOR)
                     if np.isfinite(A[ti, zi]) and np.isfinite(A[ti, zi + 1])]
            if steps:
                fall.append(np.mean(steps))
        rows[m] = dict(refine_rise=float(np.mean(rise)), overzoom_fall=float(np.mean(fall)),
                       auroc_bg=auroc(pos[m], bg[m]), auroc_xconcept=auroc(pos[m], xc[m]))

    # ---- report + plot ----
    metrics = ["refine_rise", "overzoom_fall", "auroc_bg", "auroc_xconcept"]
    print("\n=== g as a cascade control signal (predict foreground; all want HIGH, ~[0.5,1]) ===")
    print(f"{'mode':8}" + "".join(f"{k:>16}" for k in metrics))
    for m in MODES:
        print(f"{m:8}" + "".join(f"{rows[m][k]:>16.2f}" for k in metrics))

    fig, ax = plt.subplots(figsize=(9, 5))
    xpos = np.arange(len(MODES)); wdt = 0.2
    for j, k in enumerate(metrics):
        ax.bar(xpos + (j - 1.5) * wdt, [rows[m][k] for m in MODES], wdt, label=k)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(xpos); ax.set_xticklabels(MODES)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score (higher = better)")
    ax.set_title(f"g as a cascade control signal — {args.dataset} [{tag}, predict foreground]")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print(f"\nsaved plot → {out_png}")


if __name__ == "__main__":
    main()
