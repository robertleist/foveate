"""Pluggable "where is the class" foreground extractors (plan Sec. 4).

The pipeline used to hardwire one notion of *which patches belong to the
exemplar class*: a per-patch cosine to a prototype bank. We are now replacing
that with the official INSID3 algorithm, but we want to keep the old bank gate
around for ablations and as a cheap fallback. The clean way to do both is a
small strategy interface: every extractor turns a frozen patch-feature grid
into a foreground mask plus a confidence map, and exposes a per-exemplar CLS
bank for the downstream "what is in this bbox" classification.

Everything an extractor sees is already L2-normalized, so the consumers never
need to know whether the features were debiased, standardized or raw -- that is
each extractor's private business. The shared output is :class:`GateResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import torch


@dataclass
class GateResult:
    """The contract every foreground extractor returns.

    Attributes
    ----------
    foreground:
        ``(Hp, Wp)`` bool grid of the class region. It may contain several
        instances -- individuation happens later, on a separate signal.
    score_map:
        ``(Hp, Wp)`` float in ``[0, 1]`` -- a per-patch class confidence used
        downstream for scoring and filtering.
    cls_bank:
        ``(S, D)`` float -- the per-exemplar CLS stack, L2-normalized. One CLS
        per exemplar crop, the granularity the leaf classifier compares against.
    internals:
        Optional viz payload (cluster maps, intermediate similarities). Empty by
        default; populated only when ``predict(..., return_internals=True)``.
    """

    foreground: np.ndarray
    score_map: np.ndarray
    cls_bank: np.ndarray
    internals: dict = field(default_factory=dict)


@runtime_checkable
class ForegroundExtractor(Protocol):
    """Strategy interface for the foreground stage.

    An extractor is configured once against a reference (the exemplar image and
    its masks) via :meth:`set_reference`, then queried per target grid via
    :meth:`predict`. After ``set_reference`` it exposes :attr:`cls_bank`, the
    ``(S, D)`` L2-normalized per-exemplar CLS stack, so the integration layer can
    classify leaf crops without re-deriving it.
    """

    cls_bank: torch.Tensor                  # (S, D) L2-normalized per-exemplar CLS

    def set_reference(
        self,
        backbone,
        ref_image: np.ndarray,
        ref_masks: list[np.ndarray],
        negative_masks: list[np.ndarray] | None,
        cfg,
    ) -> None:
        ...

    def predict(
        self, target_feat: torch.Tensor, *, return_internals: bool = False
    ) -> GateResult:
        ...


def build_extractor(cfg) -> ForegroundExtractor:
    """Instantiate the extractor selected by ``cfg.foreground_extractor``.

    ``"insid3"`` -> the official INSID3 algorithm; ``"bank"`` -> the legacy
    per-patch bank gate. Imports are deferred so picking one strategy never pulls
    the other's dependencies.
    """
    name = cfg.foreground_extractor
    if name == "insid3":
        from foveate.insid3 import InSID3Extractor

        return InSID3Extractor(cfg)
    if name == "bank":
        from foveate.gate import BankExtractor

        return BankExtractor(cfg)
    raise ValueError(f"unknown foreground_extractor {name!r}")
