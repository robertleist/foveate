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


def resize_labels_to_grid(labels: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    """Sample an integer label map onto the patch grid at each **patch centre**.

    Patch ``j`` of a ViT covers pixels ``[j·P, (j+1)·P)``, and every consumer that maps a patch back
    to pixels (``cascade._child_box``, ``features.upsample_mask``) uses exactly that interval. The
    sample point must therefore be the interval's **centre**.

    ``cv2.INTER_NEAREST`` does *not* do this: it samples at ``j·scale``, the interval's **left
    edge**. Marking a patch by its left edge and then reconstructing it as the interval starting
    there shifts every derived box by **half a patch**, in both axes. At a large downscale that is
    most of a small object — it was silently moving the cascade's child crops off their instances,
    and it is why padding the crop appeared to help so much.
    """
    hp, wp = grid_hw
    h, w = labels.shape[:2]
    rows = np.clip(((np.arange(hp) + 0.5) * h / hp).astype(np.intp), 0, h - 1)
    cols = np.clip(((np.arange(wp) + 0.5) * w / wp).astype(np.intp), 0, w - 1)
    return labels[rows[:, None], cols[None, :]]


def resize_mask_to_grid(mask: np.ndarray, grid_hw: tuple[int, int], *,
                        mode: str = "center") -> np.ndarray:
    """Resize a full-res binary ``mask`` onto the patch grid. ``grid_hw`` is ``(Hp, Wp)``.

    ``mode="center"``
        A patch is foreground when its **centre** is inside the mask. The faithful downsample, and
        the right choice when the grid must *represent* the mask (e.g. an exemplar's own region).
    ``mode="any"``
        A patch is foreground when it contains **any** mask pixel. The right choice when the grid is
        a **proposal** — "the concept may be here, go and look" — because centre sampling silently
        deletes anything smaller than a patch: an object below one patch wide misses every centre,
        so it gets no patch, no component, and no crop, and the recursion never learns it exists.
        Under ``"any"`` every object marks at least one patch, and the patch-interval box is
        guaranteed to *contain* the object rather than under-cover it.

    See :func:`resize_labels_to_grid` for why the sample point is the patch centre and not
    ``cv2.INTER_NEAREST``'s left edge.
    """
    m = np.asarray(mask)
    if mode == "any":
        hp, wp = grid_hw
        # INTER_AREA averages the source pixels falling in each cell, so > 0 ⇔ the cell contains at
        # least one mask pixel. (It degrades to nearest when upsampling, which is what we want.)
        return cv2.resize(m.astype(np.float32), (wp, hp), interpolation=cv2.INTER_AREA) > 0.0
    if mode != "center":
        raise ValueError(f"unknown resize mode {mode!r}; expected 'center' or 'any'")
    return resize_labels_to_grid(m.astype(np.uint8), grid_hw).astype(bool)


def upsample_mask(mask: np.ndarray, size_wh: tuple[int, int], *, bilinear: bool = False) -> np.ndarray:
    """Upsample a coarse (patch-grid) binary ``mask`` to pixel resolution ``size_wh`` (W, H).

    ``bilinear`` smooths the blocky patch boundaries: resize the {0,1} mask as float with
    ``INTER_LINEAR`` and re-binarize at 0.5 (the level set halfway between inside and outside).
    Nearest-neighbour (default) keeps the exact patch-grid staircase. Returns ``uint8`` {0,1}.
    """
    w, h = size_wh
    if bilinear:
        soft = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return (soft >= 0.5).astype(np.uint8)
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


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
