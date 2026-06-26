"""Stage 3 -- agglomerative over-segmentation into part-level atoms (INSID3 Sec. 3.2).

We partition the gated foreground into many small, feature-coherent, *spatially
contiguous* clusters. These "atoms" are the units the later stages group into
instances. A single distance threshold ``tau`` controls granularity (no fixed K),
matching INSID3's open-world motivation.

Spatial contiguity is enforced via a connectivity graph over the 4-neighbourhood
of foreground patches, so a cluster can never jump across a gap in the image --
crucial for keeping morphologically distinct instances from being merged purely
because their features are similar (they always are: same class).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering


def _grid_connectivity(foreground: np.ndarray) -> tuple[np.ndarray, csr_matrix]:
    """4-neighbour adjacency among foreground patches.

    Returns the ``(N, 2)`` array of foreground patch coordinates (row, col) in a
    stable order and the sparse ``(N, N)`` adjacency used by scikit-learn's
    connectivity-constrained agglomerative clustering.
    """
    hp, wp = foreground.shape
    coords = np.argwhere(foreground)                       # (N, 2)
    index = -np.ones((hp, wp), dtype=np.int64)
    index[foreground] = np.arange(coords.shape[0])

    rows, cols = [], []
    for n, (r, c) in enumerate(coords):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < hp and 0 <= cc < wp and foreground[rr, cc]:
                rows.append(n)
                cols.append(index[rr, cc])
    n = coords.shape[0]
    data = np.ones(len(rows), dtype=np.uint8)
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    return coords, adj


def agglomerative_oversegment(
    features: torch.Tensor,
    foreground: np.ndarray,
    distance_threshold: float,
    linkage: str = "average",
) -> np.ndarray:
    """Cluster foreground patches into atoms; return a ``(Hp, Wp)`` int label grid.

    Label ``0`` is background (non-foreground); atoms are labelled ``1..K``.

    ``linkage`` selects scikit-learn's agglomerative linkage. ``"average"`` (the
    default) keeps the part-level over-segmentation behaviour; INSID3 prefers
    ``"single"`` so feature-coherent chains stay together. ``"single"`` /
    ``"average"`` / ``"complete"`` all combine fine with the cosine metric and the
    connectivity graph (only ``"ward"`` would forbid a non-euclidean metric).
    """
    hp, wp, d = features.shape
    labels = np.zeros((hp, wp), dtype=np.int32)

    coords, adj = _grid_connectivity(foreground)
    if coords.shape[0] == 0:
        return labels
    if coords.shape[0] == 1:
        labels[tuple(coords[0])] = 1
        return labels

    feats = features[foreground].cpu().numpy()             # (N, D), L2-normed

    # Cosine affinity is undefined for zero vectors. They arise legitimately when clustering the
    # *whole* grid (cluster_all) — e.g. a pure-black patch under standardize=false L2-normalizes
    # to all-zeros. Nudge any zero-norm row onto a single constant axis so it stays valid and
    # such rows simply cluster together (a degenerate "flat" region), instead of crashing sklearn.
    zero = np.linalg.norm(feats, axis=1) < 1e-8
    if zero.any():
        feats = feats.copy()
        feats[zero] = 0.0
        feats[zero, 0] = 1.0

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage=linkage,
        connectivity=adj,
    )
    atom_ids = clusterer.fit_predict(feats)                # 0..K-1

    labels[coords[:, 0], coords[:, 1]] = atom_ids + 1      # reserve 0 for background
    return labels


def cluster_all(
    features: torch.Tensor,
    distance_threshold: float,
    linkage: str = "average",
) -> np.ndarray:
    """Cluster **every** patch in pure feature space (the official INSID3 step).

    Unlike :func:`agglomerative_oversegment` (which constrains merges to spatial neighbours to
    keep atoms contiguous), INSID3's fine-grained clustering uses **no connectivity graph** and
    **average** linkage over a *precomputed cosine-distance* matrix ``D = 1 - X Xᵀ``. This is
    crucial: with a spatial graph + single linkage, smooth DINOv3 features chain every adjacent
    patch into one giant cluster, so the seed cluster (and thus the foreground) swallows the
    whole image. Feature-space average linkage instead recovers many part-level clusters, and a
    cluster may legitimately span spatially-disjoint instances of the same appearance (the
    cascade separates those later via connected components).

    ``distance_threshold`` is ``1 - tau``. Returns a ``(Hp, Wp)`` int grid with labels
    ``0..K-1`` — every patch belongs to a cluster (there is no background label here).
    """
    hp, wp, d = features.shape
    n = hp * wp
    if n <= 1:
        return np.zeros((hp, wp), dtype=np.int32)

    flat = features.reshape(n, d)
    dist = (1.0 - (flat @ flat.T).clamp(-1.0, 1.0)).cpu().numpy().astype(np.float64)
    np.fill_diagonal(dist, 0.0)                            # guard FP noise on the diagonal

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(distance_threshold),
        metric="precomputed",
        linkage=linkage,
    )
    labels = clusterer.fit_predict(dist)                  # 0..K-1
    return labels.reshape(hp, wp).astype(np.int32)
