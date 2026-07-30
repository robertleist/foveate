"""EXTRACT — propose the instances of the concept on this crop (slot 1 of 3).

The cascade is three swappable slots (see :mod:`foveate.cascade`):

* **Extract** (*this module*) — which instances of the concept are on this crop?
* **Stop** (:mod:`foveate.stop`) — descend, emit or reject?
* **Merge** (:mod:`foveate.merge_rule`) — combine everything the recursion emitted.

One question, one call::

    Extractor(crop, exemplars) -> [instance mask, ...]

This slot used to be two — a *Where* stage that answered "which patches are the concept" and an
*Extract* stage that cut that foreground into instances. That contract is lossy for exactly the
extractors worth plugging in: SAM-in-the-crop and NTT produce **instance masks directly**, and
forcing them through a per-patch foreground meant flattening those masks into one region and then
re-splitting it — discarding the instance information they had just produced. It also caused a
class of bug: two stages meant two oracles mapping ground truth onto the patch grid, and they
disagreed (roadmap §A0.1). One slot makes that impossible by construction.

Implementations:

:class:`CompositeExtractor` (``composite``, default)
    The *factorizable* extractor: a Where stage (:mod:`foveate.foreground` — ``insid3``, ``otsu``,
    ``bank``, and the GT ones) followed by a grouping (:mod:`foveate.grouping` — ``cc``, ``kmeans``,
    ``agglomerative``, ``watershed``). Keeps every pre-two-slot config resolving, and keeps the
    Where-vs-grouping ablation available for the one extractor where that decomposition is real.
:class:`OracleExtractor` (``oracle``)
    Monolithic upper bound: the ground-truth instances of the crop. Replaces all three former oracle
    spellings (``foreground_extractor: oracle``, ``foreground_extractor: oracle_cc``,
    ``instance_extractor: oracle``) with one object and one way of putting the GT on the grid.

Selected by ``cfg.extractor``; ``None`` (the default) derives it from the legacy key pair, so no
existing YAML has to change. A monolithic ``SAMExtractor`` / ``NTTExtractor`` is a registry entry
away — that is the point of the reshaping (roadmap §1.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from foveate import features as featlib, oracle as oraclelib
from foveate.foreground import build_where
from foveate.grouping import build_grouping


@dataclass
class ExtractResult:
    """What every extractor returns for one crop.

    Attributes
    ----------
    instances:
        One ``(Hp, Wp)`` bool patch grid per instance candidate, in no particular order. May be
        empty (the concept is not on this crop). Candidates **may overlap**: they are crop
        proposals, and two objects sharing a patch genuinely need two crops.
    foreground:
        ``(Hp, Wp)`` bool — the union region the extractor calls the concept. Usually the union of
        :attr:`instances`, but kept separate because it is a different claim ("the concept is here")
        and because the masked re-identification scorer and the GT-aware zoom diagnostics both read
        it. A monolithic extractor with no notion of a class region may return the union.
    score_map:
        ``(Hp, Wp)`` float in ``[0, 1]`` — per-patch confidence, for scoring and visualization.
    internals:
        Optional visualization payload; empty unless ``extract(..., return_internals=True)``.
    """

    instances: list[np.ndarray]
    foreground: np.ndarray
    score_map: np.ndarray
    internals: dict = field(default_factory=dict)


@runtime_checkable
class Extractor(Protocol):
    """Strategy interface for the Extract slot.

    An extractor is configured once against a reference (the exemplar image(s) and their masks) via
    :meth:`set_reference`, then queried per crop via :meth:`extract`. After ``set_reference`` it
    exposes :attr:`exemplar_cls`, the ``(S, D)`` L2-normalized per-exemplar CLS stack, so the
    re-identification score ``g`` can be built without re-embedding the exemplars.

    An extractor that needs the target's ground truth (the oracle) additionally implements
    ``set_target_instances(labels)``; the cascade calls it when it is present.
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
        ...

    def extract(
        self, feat: torch.Tensor, *, cls: torch.Tensor | None = None,
        box: tuple[int, int, int, int] | None = None, return_internals: bool = False,
    ) -> ExtractResult:
        """``feat`` is the crop's ``(Hp, Wp, D)`` patch grid, ``cls`` its CLS token, ``box`` its
        ``(y0, y1, x0, x1)`` in original-image coordinates (needed by extractors that consult
        something defined on the whole image, i.e. the oracle)."""
        ...


def component_label_map(comps: list[np.ndarray], shape) -> np.ndarray:
    """Fold instance grids back into one ``(Hp, Wp)`` int label image (``0`` = background).

    Only the tracing/visualization layer wants the label *image* — the cascade itself works on the
    instance list — so the cascade builds this lazily, when an observer is attached.
    """
    out = np.zeros(shape, dtype=int)
    for i, c in enumerate(comps, start=1):
        out[c] = i
    return out


class CompositeExtractor:
    """``composite`` — a Where stage followed by a grouping (the factorizable extractor).

    ``where`` (``cfg.foreground_extractor``) answers *which patches are the concept*; ``group``
    (``cfg.instance_extractor``, alias ``grouping``) turns that region into instances. Together they
    reproduce every pre-two-slot configuration exactly, and they remain separately ablatable — which
    is the only thing the old two-slot split ever bought, now available without forcing a monolithic
    extractor through a foreground it does not have.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.where = build_where(cfg)
        self.grouping = build_grouping(cfg)

    @property
    def exemplar_cls(self):
        return self.where.exemplar_cls

    def set_reference(self, backbone, ref_image, ref_masks, negative_masks, cfg) -> None:
        self.where.set_reference(backbone, ref_image, ref_masks, negative_masks, cfg)

    def set_target_instances(self, gt: np.ndarray) -> None:
        """Forward this image's GT to a ground-truth Where stage; a no-op for the real ones."""
        if hasattr(self.where, "set_target_foreground"):
            self.where.set_target_foreground(gt)

    def extract(self, feat, *, cls=None, box=None, return_internals=False) -> ExtractResult:
        gr = self.where.predict(feat, cls=cls, box=box, return_internals=return_internals)
        return ExtractResult(
            instances=self.grouping.group(feat, gr.foreground, box=box),
            foreground=gr.foreground,
            score_map=gr.score_map,
            internals=gr.internals,
        )


class OracleExtractor:
    """``oracle`` — the ground-truth instances of the crop; the Extract slot's upper bound.

    One object where there used to be three (``foreground_extractor: oracle`` for the region,
    ``oracle_cc`` for the region with inter-instance seams carved, ``instance_extractor: oracle`` for
    the decomposition). They existed because the region and the decomposition were separate slots;
    with one slot the GT decomposition is simply what the extractor returns, and the ground truth
    reaches the patch grid at exactly one place — the defect that cost §A0 cannot recur.

    ``oracle_coverage`` (default ``"any"``) decides how the GT is put on the grid: a patch is
    positive when it holds **any** pixel of an instance. That is the right upper bound because
    extraction is a *proposal* — the cascade foveates onto what it proposes and finds out. Centre
    sampling would delete every object smaller than a patch before the recursion could see it, which
    is a downsampling artefact, not an extraction error an oracle should inherit (§A0.4).

    Instances may overlap under ``"any"`` coverage: two objects sharing a patch both claim it, and
    both must get a crop.

    The exemplar bank is built exactly like the real extractors', because the re-identification score
    ``g`` is independent of this slot and must be comparable across arms — only the instances are
    oracular, never ``g``.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # Composed, not inherited: the GT Where extractor already owns the exemplar bank and the one
        # sanctioned mask → grid path, and reusing it keeps a single implementation of both.
        self._where = oraclelib.OracleExtractor(cfg)

    @property
    def exemplar_cls(self):
        return self._where.exemplar_cls

    def set_reference(self, backbone, ref_image, ref_masks, negative_masks, cfg) -> None:
        self._where.set_reference(backbone, ref_image, ref_masks, negative_masks, cfg)

    def set_target_instances(self, gt: np.ndarray) -> None:
        """Install this image's GT instance-label map (``0`` = bg, ``i`` = the i-th instance).

        A bool union mask degrades to a single instance, which is the documented degenerate case:
        the oracle then bounds the *region*, not the decomposition. Transient per-image state, so a
        cached extractor reused across inter-protocol targets is simply refreshed each image.
        """
        self._where.set_target_foreground(gt)

    def extract(self, feat, *, cls=None, box=None, return_internals=False) -> ExtractResult:
        gr = self._where.predict(feat, cls=cls, box=box, return_internals=return_internals)
        instances = self._by_instance(gr.foreground, box)
        if return_internals:
            gr.internals["instance_labels"] = component_label_map(instances, gr.foreground.shape)
        return ExtractResult(instances=instances, foreground=gr.foreground,
                             score_map=gr.score_map, internals=gr.internals)

    def _by_instance(self, region: np.ndarray, box) -> list[np.ndarray]:
        """One grid per GT instance overlapping ``region``, under the configured coverage."""
        region = np.asarray(region, dtype=bool)
        labels = self._where.gt_labels
        if labels is None:
            raise RuntimeError(
                "OracleExtractor used before set_target_instances — the cascade must receive "
                "gt_foreground for the oracle Extract slot."
            )
        if not region.any():
            return []
        y0, y1, x0, x1 = box if box is not None else (0, labels.shape[0], 0, labels.shape[1])
        sub = labels[y0:y1, x0:x1]
        mode = oraclelib.cfg_mode(self.cfg)
        out: list[np.ndarray] = []
        if mode == "any":
            for lab in np.unique(sub):
                if lab == 0:
                    continue
                part = region & featlib.resize_mask_to_grid(sub == lab, region.shape, mode="any")
                if part.any():
                    out.append(part)
        else:
            grid = featlib.resize_labels_to_grid(sub.astype(np.int32), region.shape)
            for lab in np.unique(grid[region]):
                if lab == 0:                             # background patches: not an instance
                    continue
                part = region & (grid == lab)
                if part.any():
                    out.append(part)
        # A region covering only background GT cannot be decomposed; hand it back whole so the Stop
        # slot still gets to reject it.
        return out or [region]


#: ``cfg.extractor`` → implementation.
_EXTRACTORS = {
    "composite": CompositeExtractor,
    "oracle": OracleExtractor,
}


def resolve_extractor_name(cfg) -> str:
    """Which extractor ``cfg`` asks for, honouring the pre-two-slot key pair.

    ``cfg.extractor`` wins when set. Otherwise the legacy spelling decides, and only a configuration
    whose *decomposition* was oracular maps to the monolithic oracle: ``instance_extractor: oracle``
    (the GT decomposition) or ``foreground_extractor: oracle_cc`` (a GT region already carved into
    instances — itself a Where+group method, which is precisely why it collapses). A plain
    ``foreground_extractor: oracle`` paired with a real grouping stays a **composite** with a perfect
    Where, so the Where-headroom ablation keeps meaning what it meant.
    """
    name = getattr(cfg, "extractor", None)
    if name:
        return str(name)
    if str(cfg.instance_extractor) == "oracle" or str(cfg.foreground_extractor) == "oracle_cc":
        return "oracle"
    return "composite"


def build_extractor(cfg) -> Extractor:
    """Build the Extract strategy named by ``cfg.extractor`` (see :func:`resolve_extractor_name`)."""
    name = resolve_extractor_name(cfg)
    try:
        return _EXTRACTORS[name](cfg)
    except KeyError:
        raise ValueError(
            f"Unknown extractor {name!r}; expected one of {sorted(_EXTRACTORS)}."
        ) from None
