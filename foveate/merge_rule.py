"""MERGE — combine everything the recursion emitted (slot 3 of 3).

The cascade is three swappable slots (see :mod:`foveate.cascade`): **Extract**
(:mod:`foveate.extract`) proposes the instances on a crop, **Stop** (:mod:`foveate.stop`) decides
descend / emit / reject, and **Merge** (*this module*) turns the emitted leaves into the final
instance set.

The slot exists because deduplication is a *policy*, and a recursive segmenter needs it more than a
single-pass one: independent branches converge on the same object (the two sub-crops of a split have
disjoint patches but nesting pixel boxes), so one object is emitted several times, at several
depths and scales. Which of those to keep, and whether to keep more than one, is a question the
dense regime answers differently from the general one — §A0 measured that every *relaxation* of
suppression made the dense slice worse, i.e. NMS is not deleting good detections but holding back a
flood of overlapping ones.

Rules (``cfg.merge_rule``):

``nms`` (default)
    Greedy score-ranked suppression by mask IoU **and** containment, optionally preceded by the
    fragment union. The behaviour of record.
``soft``
    Soft suppression: an overlapping detection has its score *decayed* rather than deleted, and is
    dropped only when the score falls below a floor. Keeps the second-best detection of a crowded
    region alive, which is where hard NMS costs recall.
``none``
    No deduplication. The diagnostic arm — also what :mod:`experiments.miss_diagnostics` needs to
    obtain the pre-merge leaves from the same forward pass as the post-merge ones.

.. note::
   ``soft`` is soft-NMS (Bodla et al., 2017) with a Gaussian decay — the standard, extractor-agnostic
   soft rule. It is **not** the semantic-aware soft merge of *No Time to Train!*, which uses the
   extractor's own semantic scores; that belongs next to an ``NTTExtractor`` and is a separate
   registry entry, not a rename of this one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.ndimage import binary_dilation

from foveate.types import Instance


def boxes_overlap(b1, b2) -> bool:
    """Do two ``(y0, y1, x0, x1)`` boxes intersect? A cheap pre-filter before any mask math."""
    return (min(b1[1], b2[1]) > max(b1[0], b2[0]) and
            min(b1[3], b2[3]) > max(b1[2], b2[2]))


def mask_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """``(IoU, containment)`` of two boolean masks; containment = intersection / smaller area.

    Containment catches the *nested* duplicate a plain IoU misses: a tightly-zoomed mask sitting
    inside a looser one has low IoU (small∩ over big∪) yet containment ≈ 1.
    """
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0, 0.0
    union = int(np.logical_or(a, b).sum())
    smaller = min(int(a.sum()), int(b.sum())) or 1
    return inter / union, inter / smaller


def touches_border(mask: np.ndarray, box) -> bool:
    """Does this instance's mask reach the edge of the crop it was found in?

    A component touching its crop border is *provably* only partially in frame — the object
    continues outside the crop — so the emitted mask is a fragment, not an instance.
    """
    y0, y1, x0, x1 = box
    sub = mask[y0:y1, x0:x1]
    if sub.size == 0:
        return False
    return bool(sub[0].any() or sub[-1].any() or sub[:, 0].any() or sub[:, -1].any())


def merge_fragments(instances: list[Instance], gap: int) -> tuple[list[Instance], int]:
    """Union detections that are pieces of one object cut apart by crop boundaries.

    **Why suppression cannot do this.** It compares a pair by IoU or containment and *deletes* the
    weaker one. Two halves of an instance found in two different crops are **disjoint**: their IoU is
    ~0 and neither contains the other, so no suppression rule relates them at all — and deleting one
    would be wrong anyway, because each holds pixels the other lacks. The repair is a **union**, and
    the pair is recognised by geometry (adjacent) plus provenance (at least one piece is clipped by
    its own crop border), not by overlap.

    Merged score is the maximum over the pieces: a piece that does *not* touch a border was seen
    whole, so its confidence is the trustworthy one.

    Off by default and kept as a recorded **negative** result (§A0.2): in a crowded scene nearly
    every neighbouring pair is adjacent *and* border-touching, so the union chains separate objects
    into blobs — dense AP50 0.551 → 0.181. A stricter version would have to require the contact to
    lie along the shared crop edge.
    """
    if len(instances) < 2:
        return instances, 0

    masks = [inst.mask.astype(bool) for inst in instances]
    partial = [touches_border(m, inst.box) for m, inst in zip(masks, instances)]
    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        boxes.append((ys.min(), ys.max() + 1, xs.min(), xs.max() + 1) if ys.size else (0, 0, 0, 0))

    parent = list(range(len(instances)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    struct = np.ones((2 * gap + 1, 2 * gap + 1), dtype=bool)
    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            if not (partial[i] or partial[j]):
                continue                      # both were seen whole → different objects
            bi, bj = boxes[i], boxes[j]       # cheap reject: tight boxes further apart than the gap
            if (bi[0] > bj[1] + gap or bj[0] > bi[1] + gap
                    or bi[2] > bj[3] + gap or bj[2] > bi[3] + gap):
                continue
            y0 = max(0, min(bi[0], bj[0]) - gap); y1 = max(bi[1], bj[1]) + gap
            x0 = max(0, min(bi[2], bj[2]) - gap); x1 = max(bi[3], bj[3]) + gap
            a = masks[i][y0:y1, x0:x1]
            b = masks[j][y0:y1, x0:x1]
            if (a & b).any() or (binary_dilation(a, structure=struct) & b).any():
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(instances)):
        groups.setdefault(find(i), []).append(i)
    if len(groups) == len(instances):
        return instances, 0

    out = []
    for members in groups.values():
        if len(members) == 1:
            out.append(instances[members[0]])
            continue
        merged = np.logical_or.reduce([masks[i] for i in members]).astype(np.uint8)
        best = max(members, key=lambda i: instances[i].score)
        ys, xs = np.where(merged)
        box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        out.append(Instance(merged, box, instances[best].depth, instances[best].score))
    return out, len(instances) - len(out)


def nms(instances: list[Instance], iou_thresh: float, contain_thresh: float
        ) -> tuple[list[Instance], int]:
    """Greedy, score-ranked NMS dropping duplicate detections of the *same* object.

    Independent branches converge on one object — a split's two sub-crops carry disjoint foreground
    *patches* but their padded pixel boxes can nest or overlap, so each re-discovers the whole object
    down its own branch and it is emitted twice, at possibly different depths/scales. Keep the
    highest-scoring instance and suppress any later one that overlaps it above ``iou_thresh`` OR is
    contained in it beyond ``contain_thresh`` (the nested case). Returns ``(kept in original emission
    order, n_suppressed)``.
    """
    order = sorted(range(len(instances)), key=lambda i: instances[i].score, reverse=True)
    kept: list[int] = []
    kept_masks: list[np.ndarray] = []
    for i in order:
        m = instances[i].mask.astype(bool)
        dup = False
        for j, km in zip(kept, kept_masks):
            if not boxes_overlap(instances[i].box, instances[j].box):
                continue
            iou, contain = mask_overlap(m, km)
            if iou >= iou_thresh or contain >= contain_thresh:
                dup = True
                break
        if not dup:
            kept.append(i)
            kept_masks.append(m)
    kept.sort()                                        # back to emission order for a stable result
    return [instances[i] for i in kept], len(instances) - len(kept)


def soft_nms(instances: list[Instance], sigma: float, score_floor: float, contain_thresh: float
             ) -> tuple[list[Instance], int]:
    """Soft suppression: decay an overlapping detection's score instead of deleting it.

    Gaussian decay ``s ← s · exp(−o² / σ)`` where ``o`` is the larger of IoU and containment against
    an already-kept detection, applied in descending score order; a detection is dropped only once
    its score falls below ``score_floor``. Two genuinely distinct instances that merely touch are
    demoted rather than removed, so they still rank — which is where hard NMS costs recall in a
    crowded scene. Containment above ``contain_thresh`` is still treated as a hard duplicate: a mask
    fully inside a kept one carries no pixels of its own.

    Kept instances carry the decayed score (a :class:`~foveate.types.Instance` is immutable in
    practice, so a demoted one is rebuilt), because the decayed value is what AP must rank by.
    """
    order = sorted(range(len(instances)), key=lambda i: instances[i].score, reverse=True)
    scores = {i: float(instances[i].score) for i in order}
    kept: list[int] = []
    kept_masks: list[np.ndarray] = []
    for i in order:
        m = instances[i].mask.astype(bool)
        for j, km in zip(kept, kept_masks):
            if not boxes_overlap(instances[i].box, instances[j].box):
                continue
            iou, contain = mask_overlap(m, km)
            if contain >= contain_thresh:
                scores[i] = 0.0
                break
            o = max(iou, contain)
            if o > 0.0:
                scores[i] *= float(np.exp(-(o * o) / max(sigma, 1e-9)))
        if scores[i] >= score_floor:
            kept.append(i)
            kept_masks.append(m)
    kept.sort()                                        # back to emission order for a stable result
    out = [Instance(instances[i].mask, instances[i].box, instances[i].depth, scores[i])
           for i in kept]
    return out, len(instances) - len(out)


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------
@runtime_checkable
class MergeRule(Protocol):
    """Strategy interface for the Merge stage."""

    def merge(self, instances: list[Instance]) -> tuple[list[Instance], int, int]:
        """``instances`` → ``(kept, n_merged, n_suppressed)``, in emission order."""
        ...


class NoMerge:
    """``none`` — emit the recursion's leaves untouched."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def merge(self, instances):
        return instances, 0, 0


class _FragmentMerging:
    """The optional fragment union that runs before any suppression rule.

    Ordering matters: a crop boundary can cut one object into disjoint pieces that suppression is
    blind to (IoU ~ 0, no containment), and unioning them *first* means the union — not an arbitrary
    piece — is what competes for survival.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def _fragments(self, instances):
        if not self.cfg.merge_fragments:
            return instances, 0
        return merge_fragments(instances, self.cfg.merge_fragment_gap)


class NmsMerge(_FragmentMerging):
    """``nms`` (default) — greedy score-ranked suppression by mask IoU and containment.

    Set both ``nms_iou`` and ``nms_containment`` to 1.0 to keep the fragment union but suppress
    nothing (``merge_rule: none`` skips both).
    """

    def merge(self, instances):
        instances, n_merged = self._fragments(instances)
        if self.cfg.nms_iou >= 1.0 and self.cfg.nms_containment >= 1.0:
            return instances, n_merged, 0
        kept, n_suppressed = nms(instances, self.cfg.nms_iou, self.cfg.nms_containment)
        return kept, n_merged, n_suppressed


class SoftMerge(_FragmentMerging):
    """``soft`` — soft-NMS: decay overlapping scores instead of deleting them."""

    def merge(self, instances):
        instances, n_merged = self._fragments(instances)
        kept, n_suppressed = soft_nms(instances, self.cfg.merge_soft_sigma,
                                      self.cfg.merge_soft_score_floor, self.cfg.nms_containment)
        return kept, n_merged, n_suppressed


#: ``cfg.merge_rule`` → implementation.
_MERGE_RULES = {
    "nms": NmsMerge,
    "soft": SoftMerge,
    "none": NoMerge,
}


def build_merge_rule(cfg) -> MergeRule:
    """Build the Merge strategy named by ``cfg.merge_rule``."""
    name = str(cfg.merge_rule)
    try:
        return _MERGE_RULES[name](cfg)
    except KeyError:
        raise ValueError(
            f"Unknown merge_rule {name!r}; expected one of {sorted(_MERGE_RULES)}."
        ) from None
