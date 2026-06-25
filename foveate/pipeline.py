"""Pipeline orchestration for INSID3-Instance.

``run`` threads one image + exemplar masks through all stages and returns an
:class:`InSID3Result` carrying *every* intermediate, so the notebook can render
each step and the production model wrapper can grab just the final masks.

The full-resolution per-instance masks are produced here (nearest-neighbour
upsampling of the patch-level label grid), keeping the model wrapper thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch

from foveate import clustering, features as featlib, gate as gatelib, individuation, merge
from foveate.config import Config


@dataclass
class InSID3Result:
    params: Config
    image: np.ndarray
    grid_hw: tuple[int, int]
    features: torch.Tensor
    gate: gatelib.GateResult
    atom_labels: np.ndarray
    individuation: individuation.IndividuationResult
    merged_labels: np.ndarray             # (Hp, Wp) patch-level final instances
    instance_masks: np.ndarray            # (N, H, W) uint8 full-res masks
    instance_scores: np.ndarray           # (N,) float
    stats: dict[str, Any] = field(default_factory=dict)


def _filter_and_score(
    labels: np.ndarray,
    score_map: np.ndarray,
    min_area: int,
    score_threshold: float,
) -> tuple[np.ndarray, list[int], list[float]]:
    """Drop tiny / low-confidence instances; return relabelled grid + ids + scores."""
    out = np.zeros_like(labels)
    kept_ids, kept_scores = [], []
    next_id = 1
    for rid in np.unique(labels):
        if rid == 0:
            continue
        region = labels == rid
        area = int(region.sum())
        score = float(score_map[region].mean())
        if area < min_area or score < score_threshold:
            continue
        out[region] = next_id
        kept_ids.append(next_id)
        kept_scores.append(score)
        next_id += 1
    return out, kept_ids, kept_scores


def run(
    backbone,
    image: np.ndarray,
    positive_masks: list[np.ndarray],
    negative_masks: list[np.ndarray] | None = None,
    params: Config | dict[str, Any] | None = None,
) -> InSID3Result:
    """Execute the full single-pass pipeline and return all intermediates."""
    if not isinstance(params, Config):
        params = Config.from_dict(params)

    orig_h, orig_w = image.shape[:2]

    # Stage 1 -- features.
    feats = featlib.embed_image(backbone, image, standardize=params.standardize)
    hp, wp = feats.shape[:2]

    # Stage 2 -- semantic gate.
    gate = gatelib.semantic_gate(feats, positive_masks, negative_masks, params.gate_threshold)

    # Stage 3 -- over-segmentation into atoms.
    atoms = clustering.agglomerative_oversegment(feats, gate.foreground, params.cluster_tau)

    # Stage 4 -- individuation via marker-controlled watershed.
    indiv = individuation.individuate(
        feats,
        gate.foreground,
        atoms,
        mode=params.marker_mode,
        alpha=params.elevation_alpha,
        beta=params.elevation_beta,
        marker_min_distance=params.marker_min_distance,
    )

    # Stage 5 -- merge over-split instances.
    merged = merge.merge_instances(
        feats,
        indiv.instances,
        indiv.feature_boundary,
        similarity_threshold=params.merge_similarity,
        boundary_threshold=params.merge_boundary,
    )

    # Filter + score, then upsample masks to original resolution.
    final_labels, kept_ids, kept_scores = _filter_and_score(
        merged, gate.score_map, params.min_instance_area, params.score_threshold,
    )
    full = cv2.resize(
        final_labels.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)

    if kept_ids:
        masks = np.stack([(full == i).astype(np.uint8) for i in kept_ids])
        scores = np.asarray(kept_scores, dtype=np.float32)
    else:
        masks = np.empty((0, orig_h, orig_w), dtype=np.uint8)
        scores = np.empty((0,), dtype=np.float32)

    return InSID3Result(
        params=params,
        image=image,
        grid_hw=(hp, wp),
        features=feats,
        gate=gate,
        atom_labels=atoms,
        individuation=indiv,
        merged_labels=final_labels,
        instance_masks=masks,
        instance_scores=scores,
        stats={
            "n_atoms": int(atoms.max()),
            "n_watershed_instances": int(indiv.instances.max()),
            "n_after_merge": int(merged.max()),
            "n_final": int(masks.shape[0]),
            "foreground_patches": int(gate.foreground.sum()),
        },
    )
