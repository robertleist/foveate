"""The legacy bank gate, repackaged as a :class:`ForegroundExtractor`.

Before INSID3 this was the whole "where is the class" story: build a prototype
bank from the exemplars, then score every target patch by its **max cosine to
any bank prototype** and threshold (statically or adaptively). It is one-sided
-- it never asks whether a patch matches the reference *background* even better
-- but it is cheap, robust and a useful ablation baseline, so we keep it behind
the same strategy interface as INSID3 (:mod:`foveate.foreground`).

This is the gating logic that used to live as the ``gate_to_bank`` closure in the
cascade, lifted verbatim into :meth:`BankExtractor.predict`.
"""

from __future__ import annotations

import numpy as np
import torch

# Re-exported for backward-compat: GateResult now lives in foveate.foreground.
from foveate.foreground import GateResult  # noqa: F401
from foveate import thresholding
from foveate.debias import estimate_positional_basis, project_out
from foveate.prototypes import build_bank


class BankExtractor:
    """Per-patch max-cosine-to-bank foreground extractor."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.exemplar_cls: torch.Tensor | None = None
        self.prototypes: torch.Tensor | None = None
        self._B: torch.Tensor | None = None

    def set_reference(
        self,
        backbone,
        ref_image: "np.ndarray | list[np.ndarray]",
        ref_masks: list[np.ndarray],
        negative_masks: list[np.ndarray] | None,
        cfg,
    ) -> None:
        """Build the prototype bank (optionally positionally debiased) from the reference.

        ``ref_image`` may be a single array or a list parallel to ``ref_masks`` (multi-image
        exemplars); ``build_bank`` normalizes it."""
        self._B = None
        if cfg.debias:
            self._B = estimate_positional_basis(
                backbone, subspace_dim=cfg.debias_subspace_dim, n_noise=cfg.debias_n_noise,
                seed=cfg.debias_seed, standardize=cfg.standardize,
            )
        bank = build_bank(
            backbone, ref_image, ref_masks,
            reduction=cfg.prototype_reduction, budget=cfg.prototype_budget,
            per_exemplar_min=cfg.prototype_per_exemplar_min, n_prototypes=cfg.n_prototypes,
            standardize=cfg.standardize, debias_B=self._B,
        )
        self.prototypes = bank.prototypes
        self.exemplar_cls = bank.exemplar_cls

    def predict(
        self, target_feat: torch.Tensor, *, cls: torch.Tensor | None = None,
        return_internals: bool = False,
    ) -> GateResult:
        """Foreground = per-patch max cosine to the bank, thresholded (static/adaptive).

        ``cls`` is accepted for interface parity with INSID3 but unused (the bank gate scores
        every target patch against the whole prototype bank, no per-crop exemplar selection)."""
        hp, wp, d = target_feat.shape
        flat = project_out(target_feat.reshape(hp * wp, d), self._B)
        sims = (flat @ self.prototypes.T).max(dim=1).values.reshape(hp, wp).cpu().numpy()

        if self.cfg.gate_threshold_mode == "static":
            foreground = sims >= self.cfg.gate_threshold
        else:
            foreground = thresholding.foreground(
                sims, self.cfg.gate_threshold_mode, static=self.cfg.gate_threshold,
                percentile=self.cfg.gate_percentile,
            )

        score_map = _minmax(sims)
        internals = {"sims": sims} if return_internals else {}
        return GateResult(
            foreground=foreground, score_map=score_map,
            exemplar_cls=self.exemplar_cls.cpu().numpy(), internals=internals,
        )


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
