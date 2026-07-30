"""Pluggable "where is the class" foreground extractors (plan Sec. 4).

The pipeline used to hardwire one notion of *which patches belong to the
exemplar class*: a per-patch cosine to a prototype bank. We are now replacing
that with the official INSID3 algorithm, but we want to keep the old bank gate
around for ablations and as a cheap fallback. The clean way to do both is a
small strategy interface: every extractor turns a frozen patch-feature grid
into a foreground mask plus a confidence map, and exposes the exemplar bank's
per-exemplar CLS stack for the downstream re-identification score ``g``.

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
    exemplar_cls:
        ``(S, D)`` float -- the exemplar bank's CLS stack, L2-normalized. One CLS
        token per exemplar crop, what the re-identification score ``g`` compares against.
    internals:
        Optional viz payload (cluster maps, intermediate similarities). Empty by
        default; populated only when ``predict(..., return_internals=True)``.
    """

    foreground: np.ndarray
    score_map: np.ndarray
    exemplar_cls: np.ndarray
    internals: dict = field(default_factory=dict)


@runtime_checkable
class ForegroundExtractor(Protocol):
    """Strategy interface for the foreground stage.

    An extractor is configured once against a reference (the exemplar image(s) and
    their masks) via :meth:`set_reference`, then queried per target grid via
    :meth:`predict`. After ``set_reference`` it exposes :attr:`exemplar_cls`, the
    ``(S, D)`` L2-normalized per-exemplar CLS stack, so the integration layer can
    reidentify leaf crops without re-deriving it.
    """

    exemplar_cls: torch.Tensor                  # (S, D) L2-normalized per-exemplar CLS

    def set_reference(
        self,
        backbone,
        ref_image: "np.ndarray | list[np.ndarray]",
        ref_masks: list[np.ndarray],
        negative_masks: list[np.ndarray] | None,
        cfg,
    ) -> None:
        """``ref_image`` is a single array (all exemplars share it) or a list parallel to
        ``ref_masks`` (each exemplar on its own image — multi-image exemplars). Normalize
        with :func:`normalize_reference`."""
        ...

    def predict(
        self, target_feat: torch.Tensor, *, cls: torch.Tensor | None = None,
        box: tuple[int, int, int, int] | None = None, return_internals: bool = False,
    ) -> GateResult:
        """``cls`` (the crop's CLS token, optional) lets an extractor pick which exemplars to run
        against per crop; extractors that don't need it ignore it. ``box`` (the crop's
        ``(y0, y1, x0, x1)`` in original-image coords) is likewise optional — only the oracle
        extractor, which slices a ground-truth mask to the crop, consumes it."""
        ...


def normalize_reference(
    ref_image: "np.ndarray | list[np.ndarray]",
    ref_masks: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Normalize a reference into parallel ``(images, masks)`` lists over non-empty masks.

    ``ref_image`` is either a single array (all exemplars live on it — the intra / single
    support-image case) or a list parallel to ``ref_masks`` (each exemplar lives on its own
    image — the multi-image exemplar case). A length-1 list broadcasts to every mask. Empty
    masks are dropped in lockstep with their image, so downstream stages never special-case
    how many images the reference spans.
    """
    masks = list(ref_masks)
    if isinstance(ref_image, (list, tuple)):
        images = list(ref_image)
        if len(images) == 1:
            images = images * len(masks)
        if len(images) != len(masks):
            raise ValueError(
                f"exemplar images ({len(images)}) must be 1 or match exemplar masks ({len(masks)})"
            )
    else:
        images = [ref_image] * len(masks)

    pairs = [(im, m.astype(bool)) for im, m in zip(images, masks) if m.any()]
    if not pairs:
        raise ValueError("All reference masks are empty.")
    imgs, msks = zip(*pairs)
    return list(imgs), list(msks)


def build_extractor(cfg) -> ForegroundExtractor:
    """Instantiate the **Where** stage selected by ``cfg.foreground_extractor``.

    This is the first half of :class:`foveate.extract.CompositeExtractor` (and the whole foreground
    stage of the single-pass :mod:`foveate.pipeline` / :mod:`foveate.refine`), not a cascade slot of
    its own — the cascade asks one question per crop and gets instances back. See
    :mod:`foveate.extract`.

    ``"insid3"`` -> the official INSID3 algorithm (clusters + forward/backward matching);
    ``"otsu"`` -> Otsu on the similarity map to the top-k exemplars (the cheap Where baseline);
    ``"bank"`` -> the legacy per-patch bank gate;
    ``"oracle"`` -> the ground-truth foreground (upper-bound ablation; needs ``gt_foreground``);
    ``"oracle_cc"`` -> the ground-truth foreground with inter-instance seams carved so connected
    components pre-separate touching instances (needs the GT instance-label map). Imports are
    deferred so picking one strategy never pulls the others' dependencies.
    """
    name = cfg.foreground_extractor
    if name == "insid3":
        from foveate.insid3 import InSID3Extractor

        return InSID3Extractor(cfg)
    if name == "otsu":
        from foveate.otsu import OtsuExtractor

        return OtsuExtractor(cfg)
    if name == "bank":
        from foveate.gate import BankExtractor

        return BankExtractor(cfg)
    if name == "oracle":
        from foveate.oracle import OracleExtractor

        return OracleExtractor(cfg)
    if name == "oracle_cc":
        from foveate.oracle import OracleCCExtractor

        return OracleCCExtractor(cfg)
    raise ValueError(f"unknown foreground_extractor {name!r}")


#: Reads better from the composite extractor, where this is explicitly the ``where`` half.
build_where = build_extractor
