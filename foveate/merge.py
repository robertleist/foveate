"""Stage 5 -- merge over-split instances by intra-image self-similarity (INSID3 Sec. 3.4).

The watershed seeds one instance per marker, so an irregular object (e.g. a coral
fragment with several lobes) is typically *over*-split: distinct lobes become
distinct labels. INSID3 recovers full extent by aggregating regions that are both
semantically aligned and structurally coherent. We do the intra-image analogue:
merge two **spatially adjacent** watershed regions when

* their prototypes are highly self-similar (DINOv3's strong intra-image
  self-similarity, paper Eq. 12), **and**
* the feature boundary along their shared border is weak (no real seam).

A weak-boundary AND high-affinity test is what stops same-class but genuinely
separate instances (a real seam between them) from being glued together.
"""

from __future__ import annotations

import numpy as np
import torch


class _UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _region_prototypes(features: torch.Tensor, labels: np.ndarray) -> dict[int, torch.Tensor]:
    protos = {}
    for rid in np.unique(labels):
        if rid == 0:
            continue
        sel = torch.from_numpy(labels == rid).to(features.device)
        vec = features[sel].mean(dim=0)
        protos[int(rid)] = vec / (vec.norm() + 1e-8)
    return protos


def _adjacent_pairs(labels: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Map each adjacent region pair to the patch coordinates along their border."""
    pairs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    hp, wp = labels.shape
    for dr, dc in ((1, 0), (0, 1)):
        a = labels[: hp - dr, : wp - dc]
        b = labels[dr:, dc:]
        ys, xs = np.where((a != b) & (a > 0) & (b > 0))
        for y, x in zip(ys.tolist(), xs.tolist()):
            la, lb = int(a[y, x]), int(b[y, x])
            key = (min(la, lb), max(la, lb))
            pairs.setdefault(key, []).append((y, x))
    return pairs


def merge_instances(
    features: torch.Tensor,
    instances: np.ndarray,
    feature_boundary: np.ndarray,
    *,
    similarity_threshold: float = 0.6,
    boundary_threshold: float = 0.5,
) -> np.ndarray:
    """Merge adjacent over-split instances; return a relabelled ``(Hp, Wp)`` grid.

    Parameters
    ----------
    similarity_threshold:
        Minimum cosine similarity between region prototypes to consider merging.
    boundary_threshold:
        Maximum mean feature-boundary strength along the shared border to allow a
        merge (a strong seam blocks merging, keeping true instances apart).
    """
    region_ids = [int(r) for r in np.unique(instances) if r != 0]
    if len(region_ids) <= 1:
        return instances

    protos = _region_prototypes(features, instances)
    pairs = _adjacent_pairs(instances)
    uf = _UnionFind(region_ids)

    for (a, b), border in pairs.items():
        sim = float(protos[a] @ protos[b])
        border_strength = float(np.mean([feature_boundary[y, x] for y, x in border]))
        if sim >= similarity_threshold and border_strength <= boundary_threshold:
            uf.union(a, b)

    # Relabel to a dense 1..K range.
    remap: dict[int, int] = {}
    out = np.zeros_like(instances)
    next_id = 1
    for rid in region_ids:
        root = uf.find(rid)
        if root not in remap:
            remap[root] = next_id
            next_id += 1
        out[instances == rid] = remap[root]
    return out
