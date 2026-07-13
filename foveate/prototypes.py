"""Exemplar prototype bank with a *global* budget (plan Sec. 4).

The old ``_exemplar_bank`` extracted prototypes per exemplar and concatenated them with no
global cap, so many exemplars produced a huge bank. :func:`build_bank` pools the masked patches
of *all* exemplars and reduces them to at most ``budget`` prototypes, decoupling representation
size from the number of exemplars.

Reductions:
* ``"all"``    — keep every masked patch (farthest-point subsample if over budget);
* ``"mean"``   — one mean vector per exemplar;
* ``"cluster"``— legacy per-exemplar k-means (``n_prototypes`` each), then concatenate;
* ``"kmeans"`` — MiniBatchKMeans over the *pooled* patches to ``budget`` centroids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from foveate import features as featlib
from foveate.debias import project_out
from foveate.foreground import normalize_reference


@dataclass
class Bank:
    prototypes: torch.Tensor               # (K, D) L2-normalized
    exemplar_cls: torch.Tensor                 # (S, D) per-exemplar CLS stack, L2-normalized
    proto: torch.Tensor                    # (D,) single mean prototype (border refine / accept)
    provenance: list[int] = field(default_factory=list)  # exemplar index per prototype

    def __len__(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def cls(self) -> torch.Tensor:
        """Back-compat: the L2-normalized mean of the per-exemplar CLS stack ``(D,)``."""
        return featlib.l2_normalize(self.exemplar_cls.mean(dim=0), dim=0)


def _mask_bbox(mask: np.ndarray, pad_frac: float) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    h, w = mask.shape
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    py, px = int((y1 - y0) * pad_frac), int((x1 - x0) * pad_frac)
    return max(0, y0 - py), min(h, y1 + py), max(0, x0 - px), min(w, x1 + px)


def _farthest_point_subsample(x: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy farthest-point sampling of ``k`` rows (cosine), for ``reduction="all"`` cap."""
    n = x.shape[0]
    if n <= k:
        return x
    chosen = [0]
    dist = 1.0 - (x @ x[0])
    for _ in range(1, k):
        nxt = int(torch.argmax(dist))
        chosen.append(nxt)
        dist = torch.minimum(dist, 1.0 - (x @ x[nxt]))
    return x[chosen]


def build_bank(
    backbone,
    image: "np.ndarray | list[np.ndarray]",
    masks: list[np.ndarray],
    *,
    reduction: str = "all",
    budget: int = 64,
    per_exemplar_min: int = 1,
    n_prototypes: int = 4,
    pad_frac: float = 0.15,
    standardize: bool = True,
    debias_B: torch.Tensor | None = None,
) -> Bank:
    """Build the prototype bank, cropping **each** exemplar separately.

    ``image`` is a single array or a list parallel to ``masks`` (multi-image exemplars — each
    cropped from its own image). ``debias_B`` (if given) is projected out of the bank patches so
    they match debiased query patches in the cross-image gate.
    """
    images, valid_masks = normalize_reference(image, masks)

    boxes = [_mask_bbox(m, pad_frac) for m in valid_masks]
    crops = [img[y0:y1, x0:x1] for img, (y0, y1, x0, x1) in zip(images, boxes)]
    embedded = featlib.embed_batch(backbone, crops, chunk=8, standardize=standardize)

    per_exemplar: list[torch.Tensor] = []   # (M_i, D) masked patches, optionally debiased
    cls_list: list[torch.Tensor] = []
    for m, (feat, cls), (y0, y1, x0, x1) in zip(valid_masks, embedded, boxes):
        mask_grid = featlib.resize_mask_to_grid(m[y0:y1, x0:x1], feat.shape[:2])
        if not mask_grid.any():
            mask_grid = np.ones(feat.shape[:2], dtype=bool)
        patches = feat[torch.from_numpy(mask_grid).to(feat.device)]   # (M_i, D)
        patches = project_out(patches, debias_B)
        per_exemplar.append(patches)
        cls_list.append(cls)

    protos, provenance = _reduce(per_exemplar, reduction, budget, per_exemplar_min, n_prototypes)
    exemplar_cls = featlib.l2_normalize(torch.stack(cls_list), dim=1)   # (S, D)
    proto = featlib.l2_normalize(torch.cat(per_exemplar, dim=0).mean(dim=0), dim=0)
    return Bank(prototypes=protos, exemplar_cls=exemplar_cls, proto=proto, provenance=provenance)


def _reduce(per_exemplar, reduction, budget, per_exemplar_min, n_prototypes):
    pooled = torch.cat(per_exemplar, dim=0)
    prov_all = [i for i, p in enumerate(per_exemplar) for _ in range(p.shape[0])]

    if reduction == "mean":
        protos = torch.stack([featlib.l2_normalize(p.mean(dim=0), dim=0) for p in per_exemplar])
        return protos, list(range(len(per_exemplar)))

    if reduction == "cluster":
        out, prov = [], []
        for i, p in enumerate(per_exemplar):
            centers = _kmeans(p, min(n_prototypes, p.shape[0]))
            out.append(centers)
            prov.extend([i] * centers.shape[0])
        return torch.cat(out, dim=0), prov

    if reduction == "all":
        if pooled.shape[0] <= budget:
            return pooled, prov_all
        return _farthest_point_subsample(pooled, budget), [-1] * budget

    if reduction == "kmeans":
        # Guarantee each exemplar contributes >= per_exemplar_min, then fill to budget globally.
        out, prov = [], []
        if per_exemplar_min > 0:
            for i, p in enumerate(per_exemplar):
                centers = _kmeans(p, min(per_exemplar_min, p.shape[0]))
                out.append(centers)
                prov.extend([i] * centers.shape[0])
        remaining = max(0, budget - sum(c.shape[0] for c in out))
        if remaining > 0 and pooled.shape[0] > remaining:
            out.append(_kmeans(pooled, remaining))
            prov.extend([-1] * remaining)
        elif remaining > 0:
            out.append(pooled)
            prov.extend(prov_all)
        return torch.cat(out, dim=0), prov

    raise ValueError(f"unknown reduction {reduction!r}")


def _kmeans(patches: torch.Tensor, k: int) -> torch.Tensor:
    """MiniBatchKMeans centroids (L2-normalized) of ``patches``; falls back to the mean."""
    k = max(1, min(k, patches.shape[0]))
    if k == 1 or patches.shape[0] <= k:
        if patches.shape[0] <= k:
            return featlib.l2_normalize(patches, dim=1)
        return featlib.l2_normalize(patches.mean(dim=0, keepdim=True), dim=1)
    from sklearn.cluster import MiniBatchKMeans

    x = patches.detach().cpu().numpy()
    km = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=0).fit(x)
    centers = torch.from_numpy(km.cluster_centers_).to(patches.device, patches.dtype)
    return featlib.l2_normalize(centers, dim=1)
