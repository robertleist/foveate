"""STOP — descend, emit or reject (slot 3 of 3).

The cascade is three swappable slots (see :mod:`foveate.cascade`): **Where**
(:mod:`foveate.foreground`) finds the concept on a crop, **Extract** (:mod:`foveate.extract`) turns
that foreground into instance candidates, and **Stop** (*this module*) decides what happens to them.

Three decisions, and every one of them is a *policy* — which is why they belong behind one interface
rather than inline in the loop:

``accept(reid, box)``
    Is this crop the concept at all? The reject floor τ_C (``cfg.crop_sim_floor``). Not an *accept*
    threshold — clearing it only means "not disqualified".
``survivors(...)``
    Which children are pursued, and hence (when none are) where the cascade stops and emits.
``allow_retry(...)``
    May a hair-thin zoom peak try one split before being emitted?

What is deliberately **not** here: the size floor ρ (``cfg.min_crop``). That is the cascade's
finiteness *invariant*, not a policy — zoom strictly shrinks the box and any crop at or below ρ is
emitted, so the recursion terminates whatever a Stop rule does, including an adversarial one. Moving
it behind this interface would let a plugin break termination.

Selected by ``cfg.stop_rule``: ``"reid"`` (default, the paper's rule) or ``"oracle"`` (upper bound).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

_Box = tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# The rule itself, as pure functions over scores (shared by every StopRule)
# ---------------------------------------------------------------------------
def reid_survivors(parent_reid: float, child_scores) -> list[int]:
    """Indices of children that re-identify the exemplar *more strongly* than the parent.

    Strict ``>``: a child only continues zooming if its re-id score ``g`` genuinely improves on the
    crop it came from. An empty result means no child beat the parent — the parent was the ``g``
    peak, so the cascade stops and emits it (the ``reid-stop`` rule). Any child that ties or falls
    below the parent is dropped. This is the **single-object zoom** guard; splits use
    :func:`survivors`.
    """
    return [i for i, s in enumerate(child_scores) if s > parent_reid]


def survivors(parent_reid: float, child_scores, *, crop_sim_floor: float) -> list[int]:
    """Which of a parent's children to pursue — zoom and split handled differently.

    **One child (zoom)** → the over-zoom peak guard: keep it only if its re-id score ``g`` strictly
    beats the crop it came from (:func:`reid_survivors`). Descending a single object, ``g`` should
    keep rising; when it stops rising we have passed the peak and emit the parent.

    **Several children (a split)** → the parent's ``g`` is a *biased baseline*. If the crop already
    contains a strong exemplar match, its ``g`` is pulled up by that sub-region, so gating each child
    on "beat the parent" wrongly discards a genuinely novel sibling instance that is concept-like but
    (being a *different* instance) scores lower than that inflated parent. Instead:

    1. **Confirm the split is real.** Its BEST child must *strictly* beat the parent's ``g``.
       Isolating a real object from a mixed crop *raises* ``g`` — the other instance and the
       background that were diluting the parent's CLS token drop away — so a genuine clump always has
       a sub-crop above the parent. A single object, by contrast, only yields *partial* sub-crops
       that score *below* the whole (and flat/tied ``g`` never beats the parent), so its split is not
       confirmed → keep nothing and the caller emits the parent. This stops a uniform blob
       over-segmenting.
    2. **Keep every child that improves on the parent OR clears the crop similarity floor τ_C.** A
       child that *improved* on the parent has found a better crop and must never be discarded — even
       when it is still below the floor (it will keep zooming and can rise above it). The floor
       additionally rescues a genuinely novel sibling that is concept-like but scores *below* the
       exemplar-biased parent. Only a child that falls below **both** the parent and the floor is
       pruned.
    """
    if len(child_scores) <= 1:
        return reid_survivors(parent_reid, child_scores)
    if max(child_scores) > parent_reid:                       # isolating a real object raised g
        # Never discard a child that improved on the parent (even below the floor); additionally
        # keep any child that clears the floor. Prune only children below BOTH parent and floor.
        return [i for i, s in enumerate(child_scores)
                if s > parent_reid or s >= crop_sim_floor]
    return []                                                # no sub-crop beat the parent → emit it


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------
@runtime_checkable
class StopRule(Protocol):
    """Strategy interface for the Stop stage.

    Every method takes the crop **boxes** alongside the scores. The default rule ignores them (it
    decides purely on ``g``); an oracle or a geometry-aware rule needs them. Keeping both in the
    signature is what lets the two be swapped without touching the cascade.
    """

    def accept(self, reid: float, box: _Box) -> bool:
        """Is this crop the concept (clears the reject floor)?"""
        ...

    def survivors(
        self, parent_reid: float, parent_box: _Box, child_scores, child_boxes: list[_Box]
    ) -> list[int]:
        """Indices of the children to pursue; empty ⇒ the parent was the peak → emit it."""
        ...

    def allow_retry(self, parent_reid: float, child_reid: float) -> bool:
        """May this marginal zoom peak attempt one k=2 split before being emitted?"""
        ...


class ReidStopRule:
    """``reid`` (default) — the paper's rule: re-identification ``g`` is the only stopping signal.

    A thin adapter over :func:`survivors` / :func:`reid_survivors` and the floor τ_C, so the
    behaviour of record lives in one testable place and the cascade holds no policy.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def accept(self, reid: float, box: _Box) -> bool:
        return reid >= self.cfg.crop_sim_floor

    def survivors(self, parent_reid, parent_box, child_scores, child_boxes) -> list[int]:
        # Module-level lookup (not a bound import) so tests can monkeypatch the rule in place.
        return survivors(parent_reid, child_scores, crop_sim_floor=self.cfg.crop_sim_floor)

    def allow_retry(self, parent_reid: float, child_reid: float) -> bool:
        """A zoom that peaked by only a HAIR may be sitting on a clump that tightening onto one
        component cannot improve — so it is worth one k=2 split before emitting. ``0`` disables."""
        eps = self.cfg.zoom_split_retry_eps
        return eps > 0 and 0.0 <= parent_reid - child_reid < eps


class OracleStopRule:
    """``oracle`` — the upper bound on the *stopping rule*, holding the rule's shape fixed.

    The point of this ablation is to separate two questions the default rule conflates:

    * is the **signal** good enough (does ``g`` peak where the instance is isolated)?
    * is the **rule** right (peak guard + confirm-then-floor)?

    So this rule keeps the very same survivor logic and swaps only the score: instead of ``g`` it
    uses a ground-truth *isolation* score — the best IoU between a crop box and any single GT
    instance. That score is maximal exactly when the box frames one instance tightly, so
    "descend while it rises, emit at the peak" means *emit at the GT-matching crop*. A gap between
    ``reid`` and ``oracle`` under an otherwise identical configuration is attributable to the
    signal; no gap means the rule is what limits us.

    The GT is injected per image by the cascade (``cascade(..., gt_foreground=...)``, the same
    payload the oracle *Where* extractor consumes) via :meth:`set_target_instances`. It must be an
    **int instance-label map** (``0`` = bg, ``i`` = the i-th GT instance) for the isolation score to
    mean anything; a bool union mask degenerates to a single "instance" (the union), which frames the
    whole class region rather than one object. Emitted confidences are untouched — this rule replaces
    the stop *decision*, never the score a leaf is ranked by.
    """

    #: Floor on the isolation score: > 0 ⇒ the box overlaps some GT instance. The oracle never
    #: rejects a crop that contains the concept and always rejects one that does not.
    floor = 1e-9

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._labels: np.ndarray | None = None
        self._areas: np.ndarray | None = None      # per-instance pixel area, indexed by label id

    def set_target_instances(self, gt: np.ndarray) -> None:
        """Inject the target image's GT instances (transient per-image state)."""
        # A bool union mask casts to a single instance (id 1) — the documented degenerate case.
        labels = np.asarray(gt).astype(np.int32)
        self._labels = labels
        self._areas = np.bincount(labels.ravel())

    def _isolation(self, box: _Box) -> float:
        """Best IoU between ``box`` (as a mask) and any single GT instance; 0 if no GT is injected."""
        if self._labels is None or self._areas is None or self._areas.size < 2:
            return 0.0
        y0, y1, x0, x1 = box
        sub = self._labels[y0:y1, x0:x1]
        if sub.size == 0:
            return 0.0
        inter = np.bincount(sub.ravel(), minlength=self._areas.size)[1:].astype(np.float64)
        union = float(sub.size) + self._areas[1:] - inter
        return float(np.max(np.where(union > 0, inter / np.maximum(union, 1.0), 0.0)))

    def accept(self, reid: float, box: _Box) -> bool:
        return self._isolation(box) > self.floor

    def survivors(self, parent_reid, parent_box, child_scores, child_boxes) -> list[int]:
        return survivors(self._isolation(parent_box), [self._isolation(b) for b in child_boxes],
                         crop_sim_floor=self.floor)

    def allow_retry(self, parent_reid: float, child_reid: float) -> bool:
        """Never. The retry exists to hedge a *noisy* peak; an oracle score has no noise to hedge."""
        return False


#: ``cfg.stop_rule`` → implementation.
_STOP_RULES = {
    "reid": ReidStopRule,
    "oracle": OracleStopRule,
}


def build_stop_rule(cfg) -> StopRule:
    """Build the Stop strategy named by ``cfg.stop_rule``."""
    name = str(cfg.stop_rule)
    try:
        return _STOP_RULES[name](cfg)
    except KeyError:
        raise ValueError(
            f"Unknown stop_rule {name!r}; expected one of {sorted(_STOP_RULES)}."
        ) from None
