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
) -> np.ndarray:
    """Cluster foreground patches into atoms; return a ``(Hp, Wp)`` int label grid.

    Label ``0`` is background (non-foreground); atoms are labelled ``1..K``.
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

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
        connectivity=adj,
    )
    atom_ids = clusterer.fit_predict(feats)                # 0..K-1

    labels[coords[:, 0], coords[:, 1]] = atom_ids + 1      # reserve 0 for background
    return labels
