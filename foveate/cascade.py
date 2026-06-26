"""Recursive instance discovery — zoom until the exemplar is re-identified.

Breadth-first, batched, connected-components fixed point. For each crop:

1. **Re-embed** (batched per level) and **extract the class foreground** with the configured
   foreground extractor (INSID3 by default, the prototype bank as an alternative) → the region
   of the crop that belongs to the exemplar concept. This answers *where* on the crop the class
   is.
2. **Connected components** propose tighter crops:
   - ≥ 2 components  → enqueue each (tighter child crops);
   - 1 component whose bbox does **not** fill the crop → enqueue the tighter crop (keep zooming);
   - 1 component whose bbox **fills** the crop → *converged* (no tighter crop extractable).
3. A converged crop is **accepted as a leaf** by a single test: the mean cosine similarity of
   the crop's CLS token to *all* exemplar CLS clears ``config.cls_threshold`` — a majority-vote
   that the crop contains the exemplar concept (this answers *what* is in the bbox). A
   converged-but-rejected large crop is a seamless clump → optional watershed split; otherwise
   accepted as a single instance (or discarded).

The exemplar defines the target *scale and granularity*. Only leaves are returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label

from foveate import clustering, features as featlib, individuation, merge
from foveate.config import Config
from foveate.foreground import build_extractor
from foveate.types import Instance, Stats

_CONN8 = generate_binary_structure(2, 2)   # 8-connectivity: don't over-split single instances


@dataclass
class _Region:
    box: tuple[int, int, int, int]
    depth: int


def _grid_bbox(comp):
    rows, cols = np.where(comp)
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


def _child_box(comp, box, pad_frac):
    """Pixel bbox (in original coords) of a grid component within ``box``, padded."""
    y0, y1, x0, x1 = box
    ch, cw = y1 - y0, x1 - x0
    hp, wp = comp.shape
    rmin, rmax, cmin, cmax = _grid_bbox(comp)
    py0 = y0 + int(np.floor(rmin / hp * ch)); py1 = y0 + int(np.ceil(rmax / hp * ch))
    px0 = x0 + int(np.floor(cmin / wp * cw)); px1 = x0 + int(np.ceil(cmax / wp * cw))
    pady, padx = int(round((py1 - py0) * pad_frac)), int(round((px1 - px0) * pad_frac))
    return (max(y0, py0 - pady), min(y1, py1 + pady),
            max(x0, px0 - padx), min(x1, px1 + padx))


def _box_area(box):
    return (box[1] - box[0]) * (box[3] - box[2])


def _split_clump(feat, comp_grid, cfg):
    """Watershed fallback for a seamless clump → list of sub-component grids (≥1)."""
    labels = clustering.agglomerative_oversegment(feat, comp_grid, cfg.cluster_tau)
    indiv = individuation.individuate(
        feat, comp_grid, labels, mode=cfg.marker_mode, alpha=cfg.elevation_alpha,
        beta=cfg.elevation_beta, marker_min_distance=cfg.marker_min_distance,
    )
    merged = merge.merge_instances(
        feat, indiv.instances, indiv.feature_boundary,
        similarity_threshold=cfg.merge_similarity, boundary_threshold=cfg.merge_boundary,
    )
    return [merged == i for i in np.unique(merged) if i != 0]


def discover_instances(
    backbone,
    image: np.ndarray,
    exemplar_masks: list[np.ndarray],
    negative_masks: list[np.ndarray] | None = None,
    config: Config | dict | None = None,
    *,
    exemplar_image: np.ndarray | None = None,
    observer=None,
) -> tuple[list[Instance], Stats]:
    """Discover instances of the exemplar class by recursive, batched zoom-in.

    Parameters
    ----------
    config:
        A :class:`~foveate.config.Config` (or dict) holding every tunable knob.
    exemplar_image:
        If given, the exemplar lives in *this* image while ``image`` is a **different target**
        (the in-context / cross-image setting). The foreground extractor's reference is built
        from ``exemplar_image`` and every target crop is matched against it. ``None`` (default)
        = intra-image (reference and targets share the image). Cross-image matching carries a
        DINOv3 positional bias — enable ``config.debias`` to correct it.
    observer:
        Optional ``callable(info: dict)`` invoked once per processed region for tracing.
    """
    cfg = config if isinstance(config, Config) else Config.from_dict(config)

    H, W = image.shape[:2]
    same_image = exemplar_image is None
    exemplar_area = float(np.logical_or.reduce([m.astype(bool) for m in exemplar_masks]).sum())
    leaves: list[Instance] = []
    stats = Stats()

    # Foreground extractor: set the reference (image + exemplar masks) once, then predict the
    # class region on every target crop. This is the INSID3 reference-at-gating-time flow.
    extractor = build_extractor(cfg)
    ref_image = image if same_image else exemplar_image
    extractor.set_reference(backbone, ref_image, exemplar_masks, negative_masks, cfg)
    cls_bank = extractor.cls_bank                       # (S, D) L2-normalized exemplar CLS
    stats.n_embeds += 1

    def classify(cls: torch.Tensor) -> float:
        """Mean cosine of the crop's CLS to all exemplar CLS — the "what is in the bbox" test."""
        gallery = cls_bank.to(cls.device, cls.dtype)
        return float((cls @ gallery.T).mean())

    def accept(cls_score: float) -> bool:
        return cls_score >= cfg.cls_threshold

    def emit(region, comp_grid, score):
        y0, y1, x0, x1 = region.box
        mask_local = cv2.resize(comp_grid.astype(np.uint8), (x1 - x0, y1 - y0),
                                interpolation=cv2.INTER_NEAREST)
        if int(mask_local.sum()) < cfg.cascade_min_instance_area:
            stats.discarded += 1
            return
        full = np.zeros((H, W), np.uint8)
        full[y0:y1, x0:x1] = mask_local
        leaves.append(Instance(full, region.box, region.depth, score))
        stats.leaves += 1

    frontier = [_Region((0, H, 0, W), 0)]
    level_idx = 0
    while frontier and stats.n_embeds < cfg.max_total_embeds:
        stats.level_sizes.append(len(frontier))
        crops = [image[r.box[0]:r.box[1], r.box[2]:r.box[3]] for r in frontier]
        embedded = featlib.embed_batch(backbone, crops, chunk=cfg.embed_batch_size,
                                       standardize=cfg.standardize)
        stats.n_embeds += len(crops)

        nxt: list[_Region] = []
        for r, (feat, cls) in zip(frontier, embedded):
            stats.max_depth = max(stats.max_depth, r.depth)

            fg = extractor.predict(feat).foreground
            labels, n = (label(fg, structure=_CONN8) if fg.any()
                         else (np.zeros_like(fg, dtype=int), 0))

            floor = (r.box[1] - r.box[0]) <= cfg.min_crop or (r.box[3] - r.box[2]) <= cfg.min_crop
            decision, children = "empty", []
            cls_score = classify(cls)

            if n == 0:
                pass
            elif r.depth >= cfg.max_depth or floor:       # forced convergence: emit components
                decision = "leaf-cap"
                for cid in range(1, n + 1):
                    emit(r, labels == cid, cls_score)
            elif n >= 2:                                  # multiple instances → tighter crops
                decision = "split"
                children = [_child_box(labels == cid, r.box, cfg.pad_frac) for cid in range(1, n + 1)]
            else:                                         # single component
                comp = labels == 1
                child = _child_box(comp, r.box, cfg.pad_frac)
                if _box_area(child) / max(_box_area(r.box), 1) < cfg.shrink_stop:
                    decision, children = "zoom", [child]  # strictly tighter → keep zooming
                else:                                     # converged — no tighter crop
                    comp_area = float(comp.sum()) / comp.size * _box_area(r.box)
                    if accept(cls_score) or comp_area <= cfg.clump_area_factor * exemplar_area:
                        decision = "leaf"; emit(r, comp, cls_score)
                    elif cfg.split_mode != "none":
                        subs = _split_clump(feat, comp, cfg)
                        if len(subs) >= 2:
                            decision = "clump-split"
                            children = [_child_box(s, r.box, cfg.pad_frac) for s in subs]
                        elif cfg.discard_rejected:
                            decision = "discard"; stats.discarded += 1
                        else:
                            decision = "leaf"; emit(r, comp, cls_score)  # unsplittable → accept
                    elif cfg.discard_rejected:
                        decision = "discard"; stats.discarded += 1
                    else:
                        decision = "leaf"; emit(r, comp, cls_score)

            if observer is not None:
                observer(dict(level=level_idx, depth=r.depth, box=r.box, decision=decision,
                              n_components=int(n), cls_score=cls_score, feat=feat, fg=fg,
                              comp_labels=labels, children=list(children)))
            nxt += [_Region(cb, r.depth + 1) for cb in children]

        frontier = nxt
        level_idx += 1

    return leaves, stats
