"""Stage 2 -- semantic gate: *which patches belong to the exemplar class*.

This is where the old WatershedDINO went wrong. It fed the class-similarity map
straight into watershed, treating a *semantic* signal as if it carried *instance*
structure. Here the similarity is used only as a **gate**: it answers "is this
patch the prompted class?" and nothing about "which instance?". Instance
individuation happens later (Stage 4), on a separate signal.

We combine two cues, both intra-image:

* **Prototype similarity** (INSID3 Eq. 2): cosine of each patch to the mean
  exemplar prototype. Smooth, class-level, good for scoring.
* **Backward correspondence** (INSID3 Eq. 6-8): for each patch, its nearest
  neighbour among the *exemplar* patches (positives) vs. the *negative* exemplar
  patches. A patch is kept when its best positive match beats its best negative
  match and clears an absolute threshold. This is sharper than a single global
  threshold and naturally exploits the unannotated negatives the request carries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from foveate.features import (
    prototype,
    resize_mask_to_grid,
    stack_exemplar_patches,
)


@dataclass
class GateResult:
    foreground: np.ndarray      # (Hp, Wp) bool -- gated class region
    score_map: np.ndarray       # (Hp, Wp) float in [0, 1] -- gate confidence
    prototype_map: np.ndarray   # (Hp, Wp) float -- cosine to positive prototype (viz)
    exemplar_grid: np.ndarray   # (Hp, Wp) bool -- patches covered by exemplars


def _max_similarity(features_flat: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    """Per-patch max cosine similarity to any vector in ``gallery`` ((N, D))."""
    if gallery.shape[0] == 0:
        return features_flat.new_zeros(features_flat.shape[0])
    sims = features_flat @ gallery.T          # (P, N), cosine (inputs are L2-normed)
    return sims.max(dim=1).values


def semantic_gate(
    features: torch.Tensor,
    positive_masks: list[np.ndarray],
    negative_masks: list[np.ndarray] | None,
    threshold: float,
) -> GateResult:
    """Produce the foreground class region for the exemplar concept.

    Parameters
    ----------
    features:
        ``(Hp, Wp, D)`` L2-normalized DINOv3 features.
    positive_masks / negative_masks:
        Full-resolution binary exemplar masks (will be resized to the grid).
    threshold:
        Absolute minimum positive cosine similarity required to keep a patch.
    """
    hp, wp, d = features.shape
    flat = features.reshape(hp * wp, d)

    pos_patches = stack_exemplar_patches(features, positive_masks)
    if pos_patches.shape[0] == 0:
        raise ValueError("No exemplar patch overlaps the feature grid; image too small "
                         "or exemplar mask empty. Increase backbone image_size.")
    neg_patches = stack_exemplar_patches(features, negative_masks or [])

    # Cue 1: prototype similarity (smooth, class-level).
    proto = prototype(pos_patches)
    proto_sim = (flat @ proto).reshape(hp, wp)

    # Cue 2: backward correspondence -- best positive vs. best negative match.
    sim_pos = _max_similarity(flat, pos_patches)
    sim_neg = _max_similarity(flat, neg_patches)
    keep = (sim_pos >= threshold) & (sim_pos > sim_neg)

    foreground = keep.reshape(hp, wp).cpu().numpy()

    # Confidence: positive margin over negatives, squashed to [0, 1].
    margin = (sim_pos - sim_neg).clamp(min=0.0)
    score = (sim_pos * margin.clamp(max=1.0)).reshape(hp, wp).cpu().numpy()
    score = _minmax(score)

    exemplar_grid = np.zeros((hp, wp), dtype=bool)
    for m in positive_masks:
        exemplar_grid |= resize_mask_to_grid(m, (hp, wp))

    return GateResult(
        foreground=foreground,
        score_map=score,
        prototype_map=proto_sim.cpu().numpy(),
        exemplar_grid=exemplar_grid,
    )


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
