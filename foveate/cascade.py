"""Recursive instance discovery — zoom until the exemplar is re-identified.

Breadth-first, batched, connected-components fixed point. For each crop:

1. **Re-embed** (batched per level) and **gate** by cosine to the exemplar bank → foreground.
2. **Connected components** propose tighter crops:
   - ≥ 2 components  → enqueue each (tighter child crops);
   - 1 component whose bbox does **not** fill the crop → enqueue the tighter crop (keep zooming);
   - 1 component whose bbox **fills** the crop → *converged* (no tighter crop extractable).
3. A converged crop is **accepted as a leaf** per ``config.accept_mode`` (CLS re-identification,
   mean-prototype similarity, or both). A converged-but-rejected large crop is a seamless clump
   → optional watershed split; otherwise accepted as a single instance (or discarded).

The exemplar defines the target *scale and granularity*. Only leaves are returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label

from foveate import border, clustering, features as featlib, gate as gatelib, individuation, merge, thresholding
from foveate.config import Config
from foveate.debias import estimate_positional_basis, project_out
from foveate.prototypes import build_bank
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
        (the in-context / cross-image setting). The bank is built from ``exemplar_image`` and
        every target crop is gated against it. ``None`` (default) = intra-image. Cross-image
        matching carries a DINOv3 positional bias — enable ``config.debias`` to correct it.
    observer:
        Optional ``callable(info: dict)`` invoked once per processed region for tracing.
    """
    cfg = config if isinstance(config, Config) else Config.from_dict(config)

    H, W = image.shape[:2]
    same_image = exemplar_image is None
    exemplar_area = float(np.logical_or.reduce([m.astype(bool) for m in exemplar_masks]).sum())
    leaves: list[Instance] = []
    stats = Stats()

    debias_B = None
    if cfg.debias:
        debias_B = estimate_positional_basis(
            backbone, subspace_dim=cfg.debias_subspace_dim, n_noise=cfg.debias_n_noise,
            seed=cfg.debias_seed, standardize=cfg.standardize,
        )

    bank = build_bank(
        backbone, image if same_image else exemplar_image, exemplar_masks,
        reduction=cfg.prototype_reduction, budget=cfg.prototype_budget,
        per_exemplar_min=cfg.prototype_per_exemplar_min, n_prototypes=cfg.n_prototypes,
        standardize=cfg.standardize, debias_B=debias_B,
    )
    bank_tensor, bank_cls, proto = bank.prototypes, bank.cls, bank.proto
    stats.n_embeds += 1

    def gate_to_bank(feat: torch.Tensor) -> np.ndarray:
        """Foreground grid: per-patch max cosine to the bank, thresholded (static/adaptive)."""
        hp, wp, d = feat.shape
        flat = project_out(feat.reshape(hp * wp, d), debias_B)
        sims = (flat @ bank_tensor.T).max(dim=1).values.reshape(hp, wp).cpu().numpy()
        if cfg.gate_threshold_mode == "static":
            return sims >= cfg.gate_threshold
        return thresholding.foreground(
            sims, cfg.gate_threshold_mode, static=cfg.gate_threshold,
            percentile=cfg.gate_percentile,
        )

    def proto_score(feat: torch.Tensor, comp: np.ndarray) -> float:
        hp, wp, d = feat.shape
        sims = (feat.reshape(hp * wp, d) @ proto).reshape(hp, wp).cpu().numpy()
        vals = sims[comp]
        return float(vals.mean()) if vals.size else 0.0

    def accept(cls_score: float, p_score: float) -> bool:
        cls_ok = cls_score >= cfg.cls_threshold
        proto_ok = p_score >= cfg.accept_proto_threshold
        if cfg.accept_mode == "cls":
            return cls_ok
        if cfg.accept_mode == "proto":
            return proto_ok
        if cfg.accept_mode == "both":
            return cls_ok and proto_ok
        raise ValueError(f"unknown accept_mode {cfg.accept_mode!r}")

    def combined(cls_score: float, p_score: float) -> float:
        if cfg.accept_mode == "proto":
            return p_score
        if cfg.accept_mode == "both":
            return 0.5 * (cls_score + p_score)
        return cls_score

    def emit(region, comp_grid, score, feat):
        if cfg.border_mode != "static":
            comp_grid, _, _ = border.refine(feat, comp_grid, proto, mode=cfg.border_mode)
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

            # Root, intra-image: use the accurate in-image gate -- but only if the exemplar is
            # large enough to land on the full-image grid (else fall back to the bank gate).
            use_inimage_gate = (
                r.depth == 0 and same_image
                and featlib.stack_exemplar_patches(feat, exemplar_masks).shape[0] > 0
            )
            if use_inimage_gate:
                fg = gatelib.semantic_gate(feat, exemplar_masks, negative_masks,
                                           cfg.gate_threshold).foreground
            else:
                fg = gate_to_bank(feat)
            labels, n = (label(fg, structure=_CONN8) if fg.any()
                         else (np.zeros_like(fg, dtype=int), 0))

            floor = (r.box[1] - r.box[0]) <= cfg.min_crop or (r.box[3] - r.box[2]) <= cfg.min_crop
            decision, children, score = "empty", [], None
            cls_score = float(cls @ bank_cls)

            if n == 0:
                pass
            elif r.depth >= cfg.max_depth or floor:       # forced convergence: emit components
                decision = "leaf-cap"
                for cid in range(1, n + 1):
                    comp = labels == cid
                    score = combined(cls_score, proto_score(feat, comp))
                    emit(r, comp, score, feat)
            elif n >= 2:                                  # multiple instances → tighter crops
                decision = "split"
                children = [_child_box(labels == cid, r.box, cfg.pad_frac) for cid in range(1, n + 1)]
            else:                                         # single component
                comp = labels == 1
                child = _child_box(comp, r.box, cfg.pad_frac)
                if _box_area(child) / max(_box_area(r.box), 1) < cfg.shrink_stop:
                    decision, children = "zoom", [child]  # strictly tighter → keep zooming
                else:                                     # converged — no tighter crop
                    p_score = proto_score(feat, comp)
                    score = combined(cls_score, p_score)
                    comp_area = float(comp.sum()) / comp.size * _box_area(r.box)
                    if accept(cls_score, p_score) or comp_area <= cfg.clump_area_factor * exemplar_area:
                        decision = "leaf"; emit(r, comp, score, feat)
                    elif cfg.split_mode != "none":
                        subs = _split_clump(feat, comp, cfg)
                        if len(subs) >= 2:
                            decision = "clump-split"
                            children = [_child_box(s, r.box, cfg.pad_frac) for s in subs]
                        elif cfg.discard_rejected:
                            decision = "discard"; stats.discarded += 1
                        else:
                            decision = "leaf"; emit(r, comp, score, feat)  # unsplittable → accept
                    elif cfg.discard_rejected:
                        decision = "discard"; stats.discarded += 1
                    else:
                        decision = "leaf"; emit(r, comp, score, feat)

            if observer is not None:
                observer(dict(level=level_idx, depth=r.depth, box=r.box, decision=decision,
                              n_components=int(n), cls_score=cls_score, feat=feat, fg=fg,
                              comp_labels=labels, children=list(children)))
            nxt += [_Region(cb, r.depth + 1) for cb in children]

        frontier = nxt
        level_idx += 1

    return leaves, stats
