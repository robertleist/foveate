"""STOP — descend, emit or reject (slot 2 of 3).

The cascade is three swappable slots (see :mod:`foveate.cascade`): **Extract**
(:mod:`foveate.extract`) proposes the instances on a crop, **Stop** (*this module*) decides what
happens to them, and **Merge** (:mod:`foveate.merge_rule`) combines what the recursion emitted.

Three decisions, and every one of them is a *policy* — which is why they belong behind one interface
rather than inline in the loop:

``accept(reid, box)``
    Is this crop the concept at all? The reject floor τ_C (``cfg.crop_sim_floor``). Not an *accept*
    threshold — clearing it only means "not disqualified".
``converged(instances, box, seed, seed_box)``
    Did zooming change the answer? The **fixed point**: a crop whose extraction is the single
    instance it was cropped for has nothing left to gain, so it is emitted.
``survivors(...)``
    Which children are pursued, and hence (when none are) where the cascade falls back and emits.
    The **similarity peak guard**.

Why two conditions, and what they replaced
------------------------------------------
The pre-two-slot rule had a third mechanism: a converged crop was *always* split k=2 and the split
was then confirmed by a re-identification lookahead. That existed only because k-means cannot decide
an instance count — the splitter proposed and ``g`` disposed. An extractor that returns instances
decides the count itself, so the confirm dance is gone and what remains is the pair the paper
already claims:

* the **fixed point** — the instance set is unchanged between parent and child. Compared in
  original-image coordinates with an IoU tolerance, because parent and child crops have different
  pixel extents and therefore different patch grids; the same object is a coarser mask in the parent
  and a finer one in the child, so equality has to be approximate (``cfg.stop_fixed_point_iou``).
* the **peak guard** — no child re-identifies the exemplar more strongly than the crop it came from.
  This is what catches an extractor that is scale-invariant but *wrong*: it keeps returning a
  confident answer at every scale, so the fixed point never fires, but ``g`` stops rising.

What is deliberately **not** here: the size floor ρ (``cfg.min_crop``) and the strict box shrink.
Those are the cascade's finiteness *invariant*, not a policy — zoom strictly shrinks the box and any
crop at or below ρ is emitted, so the recursion terminates whatever a Stop rule does, including an
adversarial one. Moving them behind this interface would let a plugin break termination.

Selected by ``cfg.stop_rule``: ``"reid"`` (default, the paper's rule) or ``"oracle"`` (upper bound).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from foveate import features as featlib

_Box = tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# The rule itself, as pure functions (shared by every StopRule)
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
       over-segmenting, and it is what makes an appearance-blind cut (``group=kmeans``) safe.
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


def instance_set_unchanged(
    instances: list[np.ndarray], box: _Box, seed: np.ndarray, seed_box: _Box, *, iou: float
) -> bool:
    """Is this crop a fixed point — did re-extracting at the finer scale change the answer?

    ``seed`` is the single instance grid (on the parent's crop ``seed_box``) that this crop was
    derived from; ``instances`` is what the extractor returns on ``box``. The crop is a fixed point
    when the extractor gives back **exactly that one instance**: more than one means zooming
    genuinely separated something and the branch must continue, and a different single mask means the
    answer is still moving.

    The comparison is in original-image coordinates (:func:`foveate.features.grid_iou`) because the
    two crops have different pixel extents, and it is approximate because the same object is a
    coarser mask on the parent's grid than on the child's — ``iou`` is the tolerance for that
    re-quantization, not for a difference of opinion.
    """
    if len(instances) != 1:
        return False
    return featlib.grid_iou(instances[0], box, seed, seed_box) >= iou


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------
@runtime_checkable
class StopRule(Protocol):
    """Strategy interface for the Stop stage.

    Every method takes the crop **boxes** alongside the scores. The default rule ignores them where
    it can (it decides on ``g``); an oracle or a geometry-aware rule needs them. Keeping both in the
    signature is what lets the two be swapped without touching the cascade.
    """

    def accept(self, reid: float, box: _Box) -> bool:
        """Is this crop the concept (clears the reject floor)?"""
        ...

    def converged(self, instances: list[np.ndarray], box: _Box,
                  seed: np.ndarray, seed_box: _Box) -> bool:
        """Is the extraction on ``box`` a fixed point of the crop it was derived from?"""
        ...

    def survivors(
        self, parent_reid: float, parent_box: _Box, child_scores, child_boxes: list[_Box]
    ) -> list[int]:
        """Indices of the children to pursue; empty ⇒ the parent was the peak → emit it."""
        ...


class _FixedPointMixin:
    """The fixed point, shared by every rule: it is geometry, not a signal.

    An oracle Stop rule swaps the *signal* the peak guard runs on; whether re-extraction changed the
    instance set is not a signal question, so both rules answer it the same way. A future rule is
    free to override it.
    """

    def converged(self, instances, box, seed, seed_box) -> bool:
        return instance_set_unchanged(instances, box, seed, seed_box,
                                      iou=self.cfg.stop_fixed_point_iou)


class ReidStopRule(_FixedPointMixin):
    """``reid`` (default) — the paper's rule: re-identification ``g`` is the stopping signal.

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


class OracleStopRule(_FixedPointMixin):
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

    The fixed point is inherited unchanged — it asks whether the extractor's answer moved, which no
    ground truth can improve on.

    The GT is injected per image by the cascade (``cascade(..., gt_foreground=...)``, the same
    payload the oracle *Extract* slot consumes) via :meth:`set_target_instances`. It must be an
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
        self._boxes: np.ndarray | None = None      # (N, 4) per-instance GT bbox y0,y1,x0,x1
        self._centers: np.ndarray | None = None    # (N, 2) per-instance bbox centre (y, x)

    def set_target_instances(self, gt: np.ndarray) -> None:
        """Inject the target image's GT instances (transient per-image state)."""
        # A bool union mask casts to a single instance (id 1) — the documented degenerate case.
        labels = np.asarray(gt).astype(np.int32)
        self._labels = labels
        self._areas = np.bincount(labels.ravel())

        n = int(labels.max())
        boxes = np.zeros((n, 4), dtype=np.float64)
        for i in range(1, n + 1):
            ys, xs = np.where(labels == i)
            if ys.size:
                boxes[i - 1] = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
        self._boxes = boxes
        self._centers = np.stack([(boxes[:, 0] + boxes[:, 1]) / 2,
                                  (boxes[:, 2] + boxes[:, 3]) / 2], axis=1)

    def _isolation(self, box: _Box) -> float:
        """How close is ``box`` to framing exactly one GT instance? In ``[0, 1]``.

        ``cfg.oracle_isolation`` picks the definition:

        ``"bbox"`` (default)
            IoU between the crop box and a GT instance's **bounding box**. This is the only
            definition that reaches **1 exactly when the crop equals the instance's bbox**, which is
            what the Stop slot is supposed to detect, and it is scale-free across instance shapes.
        ``"mask"`` (the original, kept for the ablation)
            IoU between the crop box treated as a filled rectangle and the instance's **mask**. Its
            maximum is the instance's *fill ratio* (mask area / bbox area), not 1 — 0.2 or less for a
            thin or diagonal object — and the score can be raised by shrinking the box into the
            densest part of the mask. So the peak sits *tighter* than the true bbox, which biases the
            rule toward over-zooming, and the value means different things for different shapes.

        Other instances being visible never lowers the score: the question is "am I framing one
        instance?", not "am I seeing only one instance".

        Target selection (``cfg.oracle_isolation_select``): ``"max"`` takes the best-scoring instance
        — the upper envelope of the per-instance curves, so the peak is the best crop available;
        ``"center"`` takes the instance whose bbox centre is nearest the crop centre, which keeps the
        target identity fixed along a zoom chain but can track an instance the crop is not framing.
        """
        if self._labels is None or self._areas is None or self._areas.size < 2:
            return 0.0
        if str(getattr(self.cfg, "oracle_isolation", "bbox")) == "mask":
            return self._isolation_mask(box)
        return self._isolation_bbox(box)

    def _isolation_bbox(self, box: _Box) -> float:
        boxes = self._boxes
        if boxes is None or boxes.size == 0:
            return 0.0
        y0, y1, x0, x1 = box
        iy0 = np.maximum(boxes[:, 0], y0); iy1 = np.minimum(boxes[:, 1], y1)
        ix0 = np.maximum(boxes[:, 2], x0); ix1 = np.minimum(boxes[:, 3], x1)
        inter = np.clip(iy1 - iy0, 0, None) * np.clip(ix1 - ix0, 0, None)
        crop_area = max((y1 - y0) * (x1 - x0), 1)
        gt_area = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
        iou = inter / np.maximum(crop_area + gt_area - inter, 1.0)

        if str(getattr(self.cfg, "oracle_isolation_select", "max")) == "center":
            cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
            d = (self._centers[:, 0] - cy) ** 2 + (self._centers[:, 1] - cx) ** 2
            # Only instances actually touching the crop are candidates; otherwise a distant
            # instance with no overlap could be selected and the score would read 0 forever.
            visible = np.where(inter > 0)[0]
            if visible.size == 0:
                return 0.0
            return float(iou[visible[np.argmin(d[visible])]])
        return float(iou.max())

    def _isolation_mask(self, box: _Box) -> float:
        """The original definition: crop-as-rectangle vs instance **mask** (see :meth:`_isolation`)."""
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
