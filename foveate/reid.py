"""The re-identification score ``g`` — a standalone, pluggable scorer over the exemplar bank.

``g(c)`` answers the paper's "is the concept in this crop?" test. Conceptually it is a similarity
between the **exemplar bank** (built once from the exemplar image crops + masks) and the region a
crop actually yields — it has nothing to do with *which* foreground extractor found that region, so
it lives here, not on any extractor.

Crucially ``g`` scores the crop's **extracted foreground**, not the whole crop: what we ultimately
emit is a mask, so ``g`` must measure how exemplar-like *that mask* is. Every masked mode therefore
takes the foreground from the *Where* stage (:meth:`ForegroundExtractor.predict`) and is computed
*after* it — only ``"cls"``, a whole-crop descriptor, needs no mask (``needs_foreground`` is False).

All modes are the **same** set similarity, differing only in how a crop is represented as a set of
L2-normalized vectors. Per exemplar, ``g`` is the mean over the *target* set of its best cosine to
the *exemplar* set (an asymmetric Chamfer: every part of the extracted mask should resemble some
exemplar part). The per-exemplar scores are then aggregated by a **top-k mean** (``cfg.reid_top_k``;
``0`` ⇒ all) — "run ``g`` k times and average".

The representations (``cfg.reid_mode``):

* ``"cls"``    — the crop's CLS token (a singleton set). Framing-sensitive; the paper's Eq. 1/2.
* ``"mean"``   — the mean of the foreground patch features (a singleton set). Content, not framing.
* ``"kmeans"`` — ``cfg.reid_kmeans_k`` k-means centroids of the foreground patches (``k=1`` ≡ mean).
* ``"full"``   — every foreground patch (no reduction; the most faithful and the most expensive).

The masked family (``mean`` / ``kmeans`` / ``full``) matches on **positionally debiased** features
(exemplar and target are different crops sharing a spurious DINOv3 position signal — see
:mod:`foveate.debias`), estimating its own basis when ``cfg.debias`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from foveate import features as featlib
from foveate.debias import estimate_positional_basis, project_out
from foveate.foreground import normalize_reference

_MASKED_MODES = ("mean", "kmeans", "full")
_MODES = ("cls",) + _MASKED_MODES


def _mask_bbox(mask: np.ndarray, pad_frac: float) -> tuple[int, int, int, int]:
    """Padded bbox of a binary mask (shared with :mod:`foveate.prototypes` / :mod:`foveate.insid3`)."""
    ys, xs = np.where(mask)
    h, w = mask.shape
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    py, px = int((y1 - y0) * pad_frac), int((x1 - x0) * pad_frac)
    return max(0, y0 - py), min(h, y1 + py), max(0, x0 - px), min(w, x1 + px)


def _reduce_patches(patches: torch.Tensor, mode: str, k: int) -> torch.Tensor:
    """Represent a set of L2-normalized patch features as the ``mode``'s vector set ``(m, D)``."""
    if mode == "mean":
        return featlib.l2_normalize(patches.mean(dim=0, keepdim=True), dim=1)
    if mode == "full":
        return patches
    # kmeans: k centroids (k=1 or too few patches ⇒ the mean, i.e. the "mean" mode)
    k = max(1, min(k, patches.shape[0]))
    if k == 1:
        return featlib.l2_normalize(patches.mean(dim=0, keepdim=True), dim=1)
    from sklearn.cluster import KMeans

    x = patches.detach().cpu().numpy()
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(x)
    centers = torch.from_numpy(km.cluster_centers_).to(patches.device, patches.dtype)
    return featlib.l2_normalize(centers, dim=1)


@dataclass
class ReidScorer:
    """Scores a target crop against the exemplar bank → ``g`` (a scalar).

    ``exemplars`` is the per-exemplar vector set ``(m_i, D)`` in this mode's representation;
    ``reid_top_k`` aggregates the per-exemplar scores (``0`` / ``>= S`` ⇒ mean, ``1`` ⇒ max, ``k`` ⇒
    top-k mean). ``B`` is the debias basis (masked family) and ``kmeans_k`` the k for ``"kmeans"``.
    """

    mode: str
    reid_top_k: int
    exemplars: list[torch.Tensor] = field(default_factory=list)
    B: torch.Tensor | None = None
    kmeans_k: int = 4

    @property
    def needs_foreground(self) -> bool:
        """Masked modes score the extracted mask, so they need the *Where* foreground; ``cls`` does not."""
        return self.mode in _MASKED_MODES

    def _target_set(self, feat: torch.Tensor, cls: torch.Tensor,
                    foreground: np.ndarray | None) -> torch.Tensor:
        """The target crop as this mode's ``(m, D)`` vector set."""
        if self.mode == "cls":
            return cls.view(1, -1)
        hp, wp, d = feat.shape
        flat = feat.reshape(hp * wp, d)
        if foreground is not None and foreground.any():
            patches = flat[torch.from_numpy(foreground.reshape(-1)).to(flat.device)]
        else:                                                # no mask (or empty) → whole grid, low g
            patches = flat
        patches = project_out(patches, self.B)
        return _reduce_patches(patches, self.mode, self.kmeans_k)

    def score(self, feat: torch.Tensor, cls: torch.Tensor,
              foreground: np.ndarray | None = None) -> float:
        """``g`` for a target crop. ``foreground`` is the *Where* mask (ignored in ``cls`` mode)."""
        t = self._target_set(feat, cls, foreground)                      # (mt, D)
        per_exemplar = torch.empty(len(self.exemplars))
        for s, e in enumerate(self.exemplars):
            sims = t @ e.to(t.device, t.dtype).T                         # (mt, me)
            per_exemplar[s] = sims.max(dim=1).values.mean()              # each target part → best match
        k = self.reid_top_k
        if 0 < k < per_exemplar.shape[0]:
            per_exemplar = per_exemplar.topk(k).values
        return float(per_exemplar.mean())


def build_reid_scorer(
    cfg,
    backbone,
    ref_image: "np.ndarray | list[np.ndarray]",
    ref_masks: list[np.ndarray],
    *,
    exemplar_cls: torch.Tensor | None = None,
    mode: str | None = None,
) -> ReidScorer:
    """Build the re-id scorer from the exemplar crops + masks (built once, reused per target).

    ``ref_image`` is a single array or a list parallel to ``ref_masks`` (multi-image exemplars).
    In ``"cls"`` mode a caller that already embedded the exemplars (e.g. the *Where* extractor) may
    pass their ``exemplar_cls`` stack to skip re-embedding; the masked modes always embed, since
    they need the exemplar foreground patches the CLS stack does not carry.

    ``mode`` overrides ``cfg.reid_mode`` — used to build a **second** scorer in a masked mode for the
    final leaf *confidence* (see :data:`Config.confidence_reid_mode`) while the recursion ``g`` keeps
    its own (``cls``) mode. All other knobs (top-k, debias, kmeans-k) come from ``cfg``.
    """
    mode = mode or cfg.reid_mode
    if mode not in _MODES:
        raise ValueError(f"unknown reid_mode {mode!r} (expected one of {_MODES})")

    if mode == "cls" and exemplar_cls is not None:
        return ReidScorer(mode="cls", reid_top_k=cfg.reid_top_k,
                          exemplars=[c.view(1, -1) for c in exemplar_cls])

    images, masks = normalize_reference(ref_image, ref_masks)
    B = None
    if mode in _MASKED_MODES and cfg.debias:
        B = estimate_positional_basis(
            backbone, subspace_dim=cfg.debias_subspace_dim, n_noise=cfg.debias_n_noise,
            seed=cfg.debias_seed, standardize=cfg.standardize,
        )

    boxes = [_mask_bbox(m, cfg.pad_frac) for m in masks]
    crops = [img[y0:y1, x0:x1] for img, (y0, y1, x0, x1) in zip(images, boxes)]
    embedded = featlib.embed_batch(backbone, crops, chunk=8, standardize=cfg.standardize)

    exemplars: list[torch.Tensor] = []
    for m, (y0, y1, x0, x1), (feat, cls) in zip(masks, boxes, embedded):
        if mode == "cls":
            exemplars.append(featlib.l2_normalize(cls, dim=0).view(1, -1))
            continue
        hp, wp, d = feat.shape
        mask_grid = featlib.resize_mask_to_grid(m[y0:y1, x0:x1], (hp, wp))
        if not mask_grid.any():                                          # mask thinner than a patch
            mask_grid = np.ones((hp, wp), dtype=bool)
        patches = project_out(feat[torch.from_numpy(mask_grid).to(feat.device)], B)  # (M, D)
        exemplars.append(_reduce_patches(patches, mode, cfg.reid_kmeans_k))

    return ReidScorer(mode=mode, reid_top_k=cfg.reid_top_k, exemplars=exemplars,
                      B=B, kmeans_k=cfg.reid_kmeans_k)
