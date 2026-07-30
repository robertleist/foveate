"""STOP slot (:mod:`foveate.stop`) — the survivor rule, the floor, the retry gate, the oracle."""

import numpy as np
import pytest

from foveate import Config, cascade
from foveate.stop import (
    OracleStopRule,
    ReidStopRule,
    build_stop_rule,
    reid_survivors,
    survivors,
)

# ---------------------------------------------------------------------------
# The rule as pure functions (behaviour of record — unchanged by the slot refactor)
# ---------------------------------------------------------------------------


def test_reid_survivors_stop_rule():
    # No child beats the parent → empty → the parent is the CLS peak (reid-stop fires).
    assert reid_survivors(0.8, [0.7, 0.8, 0.75]) == []      # ties (0.8) do NOT survive
    # Only children strictly above the parent continue; the lower-sim sibling is dropped.
    assert reid_survivors(0.6, [0.7, 0.5, 0.9]) == [0, 2]
    # A single worse child (the plain zoom case) → stop and emit the predecessor.
    assert reid_survivors(0.75, [0.6]) == []
    # A single better child → keep zooming, parent superseded.
    assert reid_survivors(0.5, [0.55]) == [0]


def test_survivors_single_child_is_strict_zoom_guard():
    """One child (zoom) uses the strict beat-the-parent peak guard — the floor is irrelevant."""
    assert survivors(0.8, [0.7], crop_sim_floor=0.5) == []      # over-zoomed → stop
    assert survivors(0.5, [0.55], crop_sim_floor=0.5) == [0]    # improved → keep zooming


def test_survivors_keeps_novel_sibling_below_biased_parent():
    """A crop holds the exemplar AND a novel instance, so its parent CLS (0.85) is inflated by the
    exemplar but still diluted below the *isolated* exemplar sub-crop (0.90). The novel sub-crop
    (0.62) scores below the parent but above the class floor — it must be KEPT, not discarded."""
    keep = survivors(0.85, [0.90, 0.62], crop_sim_floor=0.5)
    assert keep == [0, 1]                       # child 0 (>parent) confirms; child 1 clears the floor


def test_survivors_drops_below_floor_sibling():
    """A confirmed split still drops any sub-crop that fails the class floor (not the class)."""
    assert survivors(0.85, [0.90, 0.30], crop_sim_floor=0.5) == [0]


def test_survivors_keeps_improved_child_below_floor():
    """Regression: a child that IMPROVED on its (low) parent must survive even when it is still
    below the class floor — it found a better crop and keeps zooming toward the floor. Parent 0.081,
    floor 0.30: the 0.154 child beat the parent, so it must be PURSUED, not pruned for being below
    the floor (the old rule kept only ``s >= floor`` and wrongly dropped it)."""
    assert survivors(0.081, [0.154, 0.05], crop_sim_floor=0.30) == [0]   # 0.05 below both → pruned
    # a strong sibling (0.40) confirms the split; the weaker 0.154 still improved on the parent → kept
    assert survivors(0.081, [0.40, 0.154], crop_sim_floor=0.30) == [0, 1]


def test_survivors_rejects_fragmented_single_object():
    """Splitting a single object → every half is a weaker partial view (best <= parent) → the split
    is not confirmed → keep nothing so the caller emits the parent."""
    assert survivors(0.90, [0.80, 0.75], crop_sim_floor=0.5) == []    # partial halves, both worse
    assert survivors(0.90, [0.90, 0.90], crop_sim_floor=0.5) == []    # flat/tied CLS never confirms


# ---------------------------------------------------------------------------
# ReidStopRule — the same rule behind the slot interface
# ---------------------------------------------------------------------------
_BOX = (0, 10, 0, 10)


def test_reid_rule_matches_the_pure_functions():
    rule = build_stop_rule(Config(crop_sim_floor=0.5))
    assert isinstance(rule, ReidStopRule)
    boxes = [_BOX, _BOX]
    assert rule.survivors(0.85, _BOX, [0.90, 0.62], boxes) == survivors(
        0.85, [0.90, 0.62], crop_sim_floor=0.5)
    assert rule.survivors(0.8, _BOX, [0.7], [_BOX]) == []


def test_reid_rule_accept_is_the_floor():
    rule = build_stop_rule(Config(crop_sim_floor=0.5))
    assert rule.accept(0.5, _BOX) and rule.accept(0.9, _BOX)   # >= floor
    assert not rule.accept(0.49, _BOX)


def test_reid_rule_retry_gate_only_fires_on_a_hair_thin_peak():
    rule = build_stop_rule(Config(zoom_split_retry_eps=0.01))
    assert rule.allow_retry(0.80, 0.795)          # parent beat the child by 0.005 < eps → retry
    assert not rule.allow_retry(0.80, 0.70)       # a clear peak → no retry
    assert not rule.allow_retry(0.80, 0.85)       # the child WON (negative margin) → not this path
    assert not build_stop_rule(Config(zoom_split_retry_eps=0.0)).allow_retry(0.80, 0.795)  # disabled


# ---------------------------------------------------------------------------
# OracleStopRule — same rule, GT isolation score in place of g
# ---------------------------------------------------------------------------
def _labels(shape=(40, 40)):
    """Two GT instances: a 10x10 square at (5,5) and another at (25,25)."""
    lab = np.zeros(shape, dtype=np.int32)
    lab[5:15, 5:15] = 1
    lab[25:35, 25:35] = 2
    return lab


def test_oracle_isolation_peaks_on_the_tight_single_instance_box():
    rule = build_stop_rule(Config(stop_rule="oracle"))
    assert isinstance(rule, OracleStopRule)
    rule.set_target_instances(_labels())
    whole = rule._isolation((0, 40, 0, 40))          # both instances + background
    loose = rule._isolation((0, 20, 0, 20))          # instance 1 with slack
    tight = rule._isolation((5, 15, 5, 15))          # exactly instance 1
    assert tight == pytest.approx(1.0)
    assert tight > loose > whole                     # rises monotonically as the box isolates one GT


def test_oracle_survivors_ignore_g_and_follow_the_gt():
    """The rule's *shape* is held fixed and only the signal is swapped, so the oracle descends by
    GT isolation no matter what ``g`` says."""
    rule = build_stop_rule(Config(stop_rule="oracle"))
    rule.set_target_instances(_labels())
    parent = (0, 20, 0, 20)              # frames GT 1 loosely      (isolation 0.25)
    tight = (5, 15, 5, 15)               # exactly GT 1             (1.0)
    other = (25, 40, 25, 40)             # frames GT 2              (0.44)
    empty = (36, 40, 0, 4)               # frames nothing           (0.0)
    # g (0.1 against a 0.9 parent) would drop BOTH children; the oracle keeps both, because each
    # frames a GT instance better than the parent does — the confirmed-split / novel-sibling case,
    # oracle-signalled.
    assert rule.survivors(0.9, parent, [0.1, 0.1], [tight, other]) == [0, 1]
    # ...and it still prunes a child that frames nothing (below both the parent and the floor).
    assert rule.survivors(0.9, parent, [0.9, 0.9], [tight, empty]) == [0]
    # A single child that frames its instance *worse* than the parent → stop and emit the parent.
    assert rule.survivors(0.1, tight, [0.9], [parent]) == []


def test_oracle_accept_rejects_a_box_with_no_gt():
    rule = build_stop_rule(Config(stop_rule="oracle"))
    rule.set_target_instances(_labels())
    assert rule.accept(0.0, (5, 15, 5, 15))          # contains GT 1 → the concept, whatever g says
    assert not rule.accept(1.0, (36, 40, 0, 4))      # empty corner → not the concept


def test_oracle_without_gt_is_inert():
    """No GT injected (a misconfigured run) → isolation is 0 everywhere, so nothing is accepted;
    it must not raise."""
    rule = build_stop_rule(Config(stop_rule="oracle"))
    assert rule._isolation((0, 10, 0, 10)) == 0.0
    assert not rule.accept(0.9, (0, 10, 0, 10))
    assert rule.survivors(0.5, _BOX, [0.9], [_BOX]) == []


def test_oracle_never_retries():
    rule = build_stop_rule(Config(stop_rule="oracle", zoom_split_retry_eps=1.0))
    assert not rule.allow_retry(0.80, 0.795)


def test_registry_rejects_unknown_stop_rule():
    with pytest.raises(ValueError, match="Unknown stop_rule"):
        build_stop_rule(Config(stop_rule="does-not-exist"))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_cascade_runs_with_oracle_stop(backbone, two_squares):
    """The oracle Stop rule drives the cascade off the injected GT instance map: the emitted masks
    land on the GT, and the leaf scores are still the *re-id* scores (the oracle replaces the
    decision, never the confidence)."""
    img, ex = two_squares
    labels = np.zeros(img.shape[:2], dtype=np.int32)
    labels[20:40, 20:40] = 1
    labels[80:100, 80:100] = 2
    cfg = Config(stop_rule="oracle", min_crop=24, cascade_min_instance_area=4)
    instances, stats = cascade(backbone, img, ex, config=cfg, gt_foreground=labels)
    assert stats.n_embeds > 0
    gt = labels > 0
    for inst in instances:
        m = inst.mask.astype(bool)
        assert (m & gt).sum() > 0                       # every detection sits on a GT instance
        assert np.isfinite(inst.score) and abs(inst.score) <= 1.0 + 1e-5   # a cosine, not an IoU
