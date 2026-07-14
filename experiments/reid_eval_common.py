"""Shared plumbing for the re-id score ``g`` evaluations (zoom curves + trajectory metrics).

Both experiment scripts compare the same five ways of turning an exemplar bank + a target crop into
a scalar ``g``. Three are library ``reid_mode``s; two (``unmasked_*``) are cheap foreground-free
baselines defined here — they pool over *every* patch of the crop instead of the extracted
foreground, so they need no Where mask:

* ``cls``            — cosine of the crop CLS token to the exemplar CLS bank (whole-crop, framing).
* ``unmasked_mean``  — mean over ALL crop patches of cosine to the exemplar foreground prototype.
* ``unmasked_max``   — max  over ALL crop patches of that cosine (the single most concept-like patch).
* ``mean`` (masked)  — mean over the crop's FOREGROUND patches (needs the Where / GT mask).
* ``full`` (masked)  — every foreground patch matched to the exemplar patch set.

Hue groups the ablation: blue = CLS, warm = unmasked (no foreground), cool = masked (foreground).
Palette + adjacent-CVD ordering validated with the data-viz skill's checker (worst ΔE 16.2 ≥ 12).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from foveate import Config, features as featlib
from foveate.debias import project_out
from foveate.reid import build_reid_scorer, ReidScorer

# Square crop side as a multiple of the object's larger bbox dimension. <1 = over-zoom (crop inside
# the object); 1 = tight bbox; >1 = zoomed out with context.
ZOOMS = np.array([0.35, 0.45, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0])
IN_FRAME = (0.85, 2.0)                          # zoom band we call "object well in frame"

# (key, publication label, hex, marker). Order = legend order = fixed categorical assignment.
PLOT_MODES = [
    dict(key="cls",           label="CLS",           color="#2a78d6", marker="o"),
    dict(key="unmasked_mean", label="unmasked mean", color="#eb6834", marker="^"),
    dict(key="unmasked_max",  label="unmasked max",  color="#e34948", marker="v"),
    dict(key="mean",          label="masked mean",   color="#1baf7a", marker="s"),
    dict(key="full",          label="masked full",   color="#008300", marker="D"),
]
MODE_KEYS = [m["key"] for m in PLOT_MODES]

DATASET_TITLE = {"corals_coco": "Coral fragments", "polyps_coco": "Coral polyps"}
OTHER = {"corals_coco": "polyps_coco", "polyps_coco": "corals_coco"}


def poly_to_mask(seg, h: int, w: int) -> np.ndarray:
    """Rasterize a COCO polygon list into a bool mask."""
    m = np.zeros((h, w), np.uint8)
    for poly in seg:
        pts = np.array(poly, np.float64).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def load_instances(ds_dir: Path):
    """All instances as dicts {image_id, file, h, w, bbox, area, seg}."""
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
    y0, y1 = max(0, int(round(cy - half))), min(H, int(round(cy + half)))
    x0, x1 = max(0, int(round(cx - half))), min(W, int(round(cx + half)))
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


@dataclass
class Bank:
    """The exemplar bank as the three library scorers + the material for the unmasked baselines."""
    cls: ReidScorer
    mean: ReidScorer
    full: ReidScorer
    protos: list                                # exemplar foreground-mean prototypes, each (1, D)
    B: torch.Tensor | None                      # debias basis (None if debias off)


def build_bank(backbone, ex_images, ex_masks, *, debias: bool) -> Bank:
    def mk(mode):
        return build_reid_scorer(Config(reid_mode=mode, debias=debias), backbone, ex_images, ex_masks)
    mean = mk("mean")
    return Bank(cls=mk("cls"), mean=mean, full=mk("full"), protos=mean.exemplars, B=mean.B)


def score_modes(bank: Bank, feat: torch.Tensor, cls: torch.Tensor,
                fg: np.ndarray | None) -> dict[str, float]:
    """All five ``g`` values for one crop. ``fg`` (the Where/GT mask) drives only the masked modes."""
    out = {"cls": bank.cls.score(feat, cls, None),
           "mean": bank.mean.score(feat, cls, fg),
           "full": bank.full.score(feat, cls, fg)}
    hp, wp, d = feat.shape
    t = project_out(feat.reshape(hp * wp, d), bank.B)              # (P, D) all patches, debiased
    per_mean, per_max = [], []
    for e in bank.protos:                                         # exemplar foreground prototype
        s = (t @ e.to(t.device, t.dtype).T).squeeze(1)           # (P,) per-patch cosine
        per_mean.append(float(s.mean()))
        per_max.append(float(s.max()))
    out["unmasked_mean"] = float(np.mean(per_mean))
    out["unmasked_max"] = float(np.mean(per_max))
    return out
