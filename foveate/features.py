"""Stage 1 -- DINOv3 dense features and exemplar prototypes.

Everything downstream operates on a single L2-normalized patch-feature grid
``F`` of shape ``(Hp, Wp, D)``. We L2-normalize per patch so that every dot
product in the pipeline *is* a cosine similarity, which keeps the gate, the
clustering metric, the boundary map and the merge step mutually consistent.

Unlike the cross-image INSID3 setting, the exemplar(s) and the instances to
discover live in the **same** image, so there is no cross-image positional bias
to remove (Sec. 3.1 of the paper). We therefore keep the original features and
skip the noise-image debiasing -- it would only suppress the spatial structure
the individuation stage relies on.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


def embed_image(backbone, image: np.ndarray, standardize: bool = True) -> torch.Tensor:
    """Run DINOv3 on ``image`` and return per-patch features as ``(Hp, Wp, D)``.

    Parameters
    ----------
    backbone:
        A :class:`DINOv3Backbone` (or anything exposing ``preprocess`` + ``__call__``
        returning ``(1, C, Hp, Wp)``).
    standardize:
        If ``True``, z-score each feature dimension across patches before
        L2-normalization. This matches the existing WatershedDINO behaviour and
        empirically sharpens cosine contrast.

    Returns
    -------
    torch.Tensor
        ``(Hp, Wp, D)`` float tensor, L2-normalized along ``D``.
    """
    pixel_values = backbone.preprocess(image)              # (1, 3, H, W)
    features = backbone(pixel_values)                      # (1, C, Hp, Wp)
    return _standardize_and_norm(features.squeeze(0), standardize)


def l2_normalize(features: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize ``features`` so dot products become cosine similarities."""
    return features / (features.norm(dim=dim, keepdim=True) + eps)


def _standardize_and_norm(grid_chw: torch.Tensor, standardize: bool) -> torch.Tensor:
    """``(C, Hp, Wp)`` -> ``(Hp, Wp, C)`` standardized (optional) and L2-normalized."""
    feats = grid_chw.permute(1, 2, 0).float()      # (Hp, Wp, C)
    if standardize:
        mean = feats.mean(dim=(0, 1), keepdim=True)
        std = feats.std(dim=(0, 1), keepdim=True)
        feats = (feats - mean) / (std + 1e-8)
    return l2_normalize(feats)


def embed_batch(
    backbone,
    images: list[np.ndarray],
    chunk: int = 8,
    standardize: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Embed many crops, returning ``[(feat_grid (Hp,Wp,D), cls (D,))]`` per image.

    Crops are resized to the backbone's square input, so a heterogeneous list batches into
    one forward per ``chunk``. ``cls`` is L2-normalized for cosine re-identification.
    """
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(images), chunk):
        batch = images[start:start + chunk]
        pixel_values = backbone.preprocess(batch)              # (B, 3, H, W)
        grids, cls = backbone(pixel_values, return_cls=True)   # (B,C,Hp,Wp), (B,C)
        for i in range(grids.shape[0]):
            feat = _standardize_and_norm(grids[i], standardize)
            out.append((feat, l2_normalize(cls[i].float(), dim=0)))
    return out


def resize_mask_to_grid(mask: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize a full-res binary ``mask`` onto the patch grid.

    ``grid_hw`` is ``(Hp, Wp)``; ``cv2.resize`` takes ``(width, height)``.
    """
    hp, wp = grid_hw
    resized = cv2.resize(mask.astype(np.uint8), (wp, hp), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def stack_exemplar_patches(
    features: torch.Tensor,
    masks: list[np.ndarray],
) -> torch.Tensor:
    """Collect the feature vectors of every patch covered by any exemplar mask.

    Returns a ``(M, D)`` tensor (``M`` = number of foreground patches across all
    exemplars). Empty ``(0, D)`` if no mask covers any patch.
    """
    hp, wp, d = features.shape
    if not masks:
        return features.new_zeros((0, d))
    grids = [resize_mask_to_grid(m, (hp, wp)) for m in masks]
    union = np.logical_or.reduce(grids) if len(grids) > 1 else grids[0]
    sel = torch.from_numpy(union).to(features.device)
    return features[sel]


def prototype(patch_vectors: torch.Tensor) -> torch.Tensor:
    """Mean-then-renormalize prototype of a set of patch vectors (INSID3 Eq. 2)."""
    if patch_vectors.shape[0] == 0:
        raise ValueError("Cannot build a prototype from zero patches.")
    return l2_normalize(patch_vectors.mean(dim=0), dim=0)
