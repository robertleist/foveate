"""A simple Otsu foreground extractor — the training-free **Where** baseline.

Where INSID3 (:mod:`foveate.insid3`) answers *"which patches are the concept?"* with fine-grained
clustering + forward/backward matching, this extractor is the cheapest possible alternative for the
**Where** ablation axis: build one similarity map of the target patches to the exemplar bank and
threshold it with Otsu.

Concretely, per crop:

1. Select the ``otsu_top_k`` exemplars whose CLS token is most similar to the crop (the same
   top-k selection INSID3 uses), so the reference scale matches the crop.
2. Score every target patch by its cosine similarity to the selected exemplars' foreground
   prototypes, reduced over the selection by ``otsu_reduce`` (``mean`` or ``max``) → a
   ``(Hp, Wp)`` similarity map.
3. Binarize the map with **Otsu's** between-class-variance threshold (:func:`foveate.thresholding.otsu`).
   No granularity / aggregation knobs, no clustering — just one adaptive cut on the similarity map.

Like INSID3 it runs the matching on **positionally debiased** features when ``cfg.debias`` is set
(reference and target are different crops), and exposes the same :attr:`exemplar_cls` bank so the
cascade's re-identification score ``g`` is computed identically regardless of which Where extractor
produced the foreground.
"""

from __future__ import annotations

import numpy as np
import torch

from foveate import features as featlib, thresholding
from foveate.debias import estimate_positional_basis, project_out
from foveate.foreground import GateResult, normalize_reference


def _mask_bbox(mask: np.ndarray, pad_frac: float) -> tuple[int, int, int, int]:
    """Padded bbox of a binary mask (patch-scale framing shared with the cascade's crops)."""
    ys, xs = np.where(mask)
    h, w = mask.shape
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    py, px = int((y1 - y0) * pad_frac), int((x1 - x0) * pad_frac)
    return max(0, y0 - py), min(h, y1 + py), max(0, x0 - px), min(w, x1 + px)


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class OtsuExtractor:
    """Otsu-on-similarity-map foreground extractor behind the :class:`ForegroundExtractor` interface."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.exemplar_cls: torch.Tensor | None = None      # (S, D) L2-normalized per-exemplar CLS
        self._B: torch.Tensor | None = None                # debias basis (or None)
        self._protos: torch.Tensor | None = None           # (S, D) per-exemplar foreground prototype

    # ------------------------------------------------------------------ reference
    def set_reference(
        self,
        backbone,
        ref_image: "np.ndarray | list[np.ndarray]",
        ref_masks: list[np.ndarray],
        negative_masks: list[np.ndarray] | None,
        cfg,
    ) -> None:
        """Build the per-exemplar CLS bank and foreground prototypes from a padded crop of each
        exemplar mask (each cropped from its own image, so multi-image exemplars simply stack).

        ``negative_masks`` is accepted for interface parity and ignored (Otsu needs no background
        reference — the threshold is derived from the target similarity map itself)."""
        images, valid = normalize_reference(ref_image, ref_masks)

        self._B = None
        if cfg.debias:
            self._B = estimate_positional_basis(
                backbone, subspace_dim=cfg.debias_subspace_dim, n_noise=cfg.debias_n_noise,
                seed=cfg.debias_seed, standardize=cfg.standardize,
            )

        boxes = [_mask_bbox(m, cfg.pad_frac) for m in valid]
        crops = [img[y0:y1, x0:x1] for img, (y0, y1, x0, x1) in zip(images, boxes)]
        embedded = featlib.embed_batch(backbone, crops, chunk=8, standardize=cfg.standardize)
        device = embedded[0][0].device

        protos, cls_list = [], []
        for m, (y0, y1, x0, x1), (feat, cls) in zip(valid, boxes, embedded):
            hp, wp, d = feat.shape
            mask_grid = featlib.resize_mask_to_grid(m[y0:y1, x0:x1], (hp, wp))
            if not mask_grid.any():                              # mask thinner than a patch
                mask_grid = np.ones((hp, wp), dtype=bool)
            fd = project_out(feat.reshape(hp * wp, d), self._B)  # (hp*wp, D) debiased
            fg = torch.from_numpy(mask_grid.reshape(-1)).to(device)
            protos.append(featlib.l2_normalize(fd[fg].mean(dim=0), dim=0))
            cls_list.append(cls)

        self._protos = torch.stack(protos).to(device)                        # (S, D)
        self.exemplar_cls = featlib.l2_normalize(torch.stack(cls_list), dim=1).to(device)  # (S, D)

    # ------------------------------------------------------------------- select
    def _select(self, cls: torch.Tensor | None, device):
        """Top-``otsu_top_k`` exemplars by CLS cosine to the crop (all of them if ``cls`` is None)."""
        n = self._protos.shape[0]
        if cls is None:
            sel, sim = list(range(n)), None
        else:
            sims = (self.exemplar_cls.to(cls.device, cls.dtype) @ cls).detach().cpu().numpy()  # (S,)
            k = max(1, min(int(self.cfg.otsu_top_k), n))
            sel = sorted(int(i) for i in np.argsort(-sims)[:k])
            sim = float(np.mean(sims[sel]))
        return self._protos[sel].to(device), sel, sim

    # ------------------------------------------------------------------- predict
    def predict(
        self, target_feat: torch.Tensor, *, cls: torch.Tensor | None = None,
        return_internals: bool = False,
    ) -> GateResult:
        """Otsu on the per-patch similarity to the top-k exemplar prototypes → foreground grid."""
        hp, wp, d = target_feat.shape
        device = target_feat.device
        t_deb = project_out(target_feat.reshape(hp * wp, d), self._B)         # (P, D) debiased

        protos_sel, sel, sim = self._select(cls, device)                     # (K, D)
        patch_sims = t_deb @ protos_sel.T                                    # (P, K) cosine
        if self.cfg.otsu_reduce == "max":
            sims = patch_sims.max(dim=1).values
        else:
            sims = patch_sims.mean(dim=1)
        sim_grid = sims.reshape(hp, wp).detach().cpu().numpy()               # (Hp, Wp)

        tau = thresholding.otsu(sim_grid)
        foreground = sim_grid >= tau
        score_map = np.clip(_minmax(sim_grid), 0.0, 1.0)

        internals: dict = {}
        if return_internals:
            internals = {
                "forward_sim": score_map,                # per-patch similarity map (heatmap)
                "candidate_mask": foreground,            # Otsu-thresholded foreground
                "foreground": foreground,
                "otsu_threshold": float(tau),
                "selected_exemplars": list(sel),
                "select_sim": sim,
                "tau_used": float(tau),                  # reuse the Where "foreground on crop" caption
                "aggregate_used": self.cfg.otsu_reduce,
            }
        return GateResult(
            foreground=foreground,
            score_map=score_map,
            exemplar_cls=self.exemplar_cls.cpu().numpy(),
            internals=internals,
        )
