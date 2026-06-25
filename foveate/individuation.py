"""Stage 4 -- instance individuation via marker-controlled watershed.

This is the corrected watershed step. The old model watershedded the *inverted
class-similarity map* with no markers and no mask, so its basins were noise peaks
inside the class region -- unrelated to object boundaries. Here watershed is used
the way it is meant to be:

* **elevation** -- a *boundary* map that is high at instance borders, built from
  two complementary cues so it copes with mixed morphology:
    - ``feature_boundary``: ``1 - mean cosine sim to spatial neighbours`` -- high
      where DINOv3 features change quickly (texture seams between touching
      instances, part borders of irregular objects);
    - ``1 - distance_transform``: high near the edge of the foreground blob --
      drives a geometric split between convex touching instances.
* **markers** -- one seed per instance: regional maxima of the distance transform
  (geometric centres) and/or atom centroids from Stage 3 (feature centres).
* **mask** -- the gated foreground, so flooding never leaves the class region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


@dataclass
class IndividuationResult:
    instances: np.ndarray         # (Hp, Wp) int -- per-instance labels (0 = bg)
    elevation: np.ndarray         # (Hp, Wp) float -- watershed elevation map
    feature_boundary: np.ndarray  # (Hp, Wp) float -- feature discontinuity cue
    distance: np.ndarray          # (Hp, Wp) float -- foreground distance transform
    markers: np.ndarray           # (Hp, Wp) int -- seed labels fed to watershed


def feature_boundary_map(features: torch.Tensor, foreground: np.ndarray | None = None) -> np.ndarray:
    """``1 - mean cosine similarity to 4-neighbours``; high at feature borders.

    Patches are L2-normalized, so neighbour similarity is a dot product. Edge
    patches average over the neighbours they have.
    """
    hp, wp, d = features.shape
    sim_sum = torch.zeros((hp, wp), device=features.device)
    count = torch.zeros((hp, wp), device=features.device)

    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rs, re = max(0, dr), hp + min(0, dr)
        cs, ce = max(0, dc), wp + min(0, dc)
        a = features[rs:re, cs:ce, :]
        b = features[rs - dr:re - dr, cs - dc:ce - dc, :]
        sim = (a * b).sum(dim=-1)
        sim_sum[rs:re, cs:ce] += sim
        count[rs:re, cs:ce] += 1

    mean_sim = (sim_sum / count.clamp(min=1)).cpu().numpy()
    boundary = 1.0 - mean_sim
    boundary = np.clip(boundary, 0.0, None)
    if boundary.max() > 1e-12:
        boundary = boundary / boundary.max()
    if foreground is not None:
        boundary = boundary * foreground
    return boundary.astype(np.float32)


def _build_markers(
    foreground: np.ndarray,
    distance: np.ndarray,
    atom_labels: np.ndarray,
    mode: str,
    min_distance: int,
) -> np.ndarray:
    """Seed image (int labels) for watershed. ``mode`` in {geometric, feature, hybrid}."""
    hp, wp = foreground.shape
    seed_mask = np.zeros((hp, wp), dtype=bool)

    if mode in ("geometric", "hybrid"):
        # Distance-transform peaks -> one centre per convex blob.
        coords = peak_local_max(
            distance, min_distance=max(1, min_distance), labels=foreground, exclude_border=False,
        )
        seed_mask[tuple(coords.T)] = True

    if mode in ("feature", "hybrid"):
        # Atom centroids -> one seed per feature-coherent part.
        for atom_id in np.unique(atom_labels):
            if atom_id == 0:
                continue
            ys, xs = np.where(atom_labels == atom_id)
            cy, cx = int(round(ys.mean())), int(round(xs.mean()))
            if not foreground[cy, cx]:                    # centroid may fall in a hole
                cy, cx = ys[0], xs[0]
            seed_mask[cy, cx] = True

    if not seed_mask.any():
        # Degenerate: seed the single global distance maximum.
        cy, cx = np.unravel_index(np.argmax(distance), distance.shape)
        seed_mask[cy, cx] = True

    markers, _ = ndi.label(seed_mask)
    return markers


def individuate(
    features: torch.Tensor,
    foreground: np.ndarray,
    atom_labels: np.ndarray,
    *,
    mode: str = "hybrid",
    alpha: float = 1.0,
    beta: float = 0.5,
    marker_min_distance: int = 2,
) -> IndividuationResult:
    """Split the foreground class region into individual instances.

    Parameters
    ----------
    mode:
        Marker source -- ``"geometric"`` (distance peaks, best for convex/blobby),
        ``"feature"`` (atom centroids, best for irregular parts), or ``"hybrid"``.
    alpha, beta:
        Weights of the feature-boundary and geometric (1 - distance) terms in the
        watershed elevation.
    """
    fb = feature_boundary_map(features, foreground)
    distance = ndi.distance_transform_edt(foreground).astype(np.float32)
    dist_norm = distance / distance.max() if distance.max() > 1e-12 else distance

    elevation = alpha * fb + beta * (1.0 - dist_norm)
    elevation = elevation * foreground                     # irrelevant outside FG

    markers = _build_markers(foreground, distance, atom_labels, mode, marker_min_distance)
    instances = watershed(elevation, markers=markers, mask=foreground)

    return IndividuationResult(
        instances=instances.astype(np.int32),
        elevation=elevation.astype(np.float32),
        feature_boundary=fb,
        distance=distance,
        markers=markers.astype(np.int32),
    )
