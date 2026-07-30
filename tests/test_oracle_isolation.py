"""The oracle Stop rule's isolation score — "am I framing exactly one instance?"."""

from __future__ import annotations

import numpy as np
import pytest

from foveate.config import Config
from foveate.stop import build_stop_rule


def _rule(gt, **kw):
    r = build_stop_rule(Config(stop_rule="oracle", **kw))
    r.set_target_instances(gt)
    return r


def _square(hw=(100, 100), box=(20, 40, 20, 40), label=1, base=None):
    gt = np.zeros(hw, np.int32) if base is None else base
    y0, y1, x0, x1 = box
    gt[y0:y1, x0:x1] = label
    return gt


def _diagonal(hw=(100, 100), start=20, length=20):
    """A thin diagonal instance: its bbox is mostly background (fill ratio ~1/length)."""
    gt = np.zeros(hw, np.int32)
    for i in range(length):
        gt[start + i, start + i] = 1
    return gt


# --------------------------------------------------------------------------- the contract
def test_score_is_one_exactly_at_the_instance_bbox():
    r = _rule(_square())
    assert r._isolation((20, 40, 20, 40)) == pytest.approx(1.0)


def test_score_is_zero_when_no_instance_is_in_the_crop():
    r = _rule(_square())
    assert r._isolation((60, 90, 60, 90)) == 0.0


def test_score_falls_off_as_the_crop_leaves_the_instance():
    r = _rule(_square())
    exact = r._isolation((20, 40, 20, 40))
    loose = r._isolation((10, 50, 10, 50))
    whole = r._isolation((0, 100, 0, 100))
    assert exact > loose > whole > 0


def test_a_second_visible_instance_does_not_lower_the_score():
    """The question is 'am I framing one instance', not 'am I seeing only one'."""
    gt = _square()
    gt = _square(box=(20, 40, 42, 62), label=2, base=gt)       # neighbour just outside the crop
    r = _rule(gt)
    assert r._isolation((20, 40, 20, 40)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- bbox vs mask
def test_shape_independence_is_the_reason_bbox_replaced_mask():
    """On a thin diagonal instance the mask score at the CORRECT crop is near zero."""
    gt = _diagonal()
    bbox = _rule(gt, oracle_isolation="bbox")._isolation((20, 40, 20, 40))
    mask = _rule(gt, oracle_isolation="mask")._isolation((20, 40, 20, 40))
    assert bbox == pytest.approx(1.0)
    assert mask < 0.1                       # ~= the instance's fill ratio, not its correctness


def test_the_two_agree_on_a_box_shaped_instance():
    gt = _square()
    bbox = _rule(gt, oracle_isolation="bbox")._isolation((20, 40, 20, 40))
    mask = _rule(gt, oracle_isolation="mask")._isolation((20, 40, 20, 40))
    assert bbox == pytest.approx(mask)


def test_mask_score_can_be_raised_by_shrinking_into_the_instance():
    """The over-zoom mechanism: for a non-convex shape the mask score peaks inside the object."""
    gt = np.zeros((100, 100), np.int32)
    gt[20:40, 20:24] = 1                    # a thin vertical bar
    gt[20:24, 20:60] = 1                    # ...with a long horizontal arm (an L)
    r = _rule(gt, oracle_isolation="mask")
    at_bbox = r._isolation((20, 40, 20, 60))
    inside = r._isolation((20, 24, 20, 60))  # a sub-box hugging the arm only
    assert inside > at_bbox                  # peak is NOT at the true bbox


def test_bbox_score_peaks_at_the_true_bbox_for_the_same_shape():
    gt = np.zeros((100, 100), np.int32)
    gt[20:40, 20:24] = 1
    gt[20:24, 20:60] = 1
    r = _rule(gt, oracle_isolation="bbox")
    assert r._isolation((20, 40, 20, 60)) > r._isolation((20, 24, 20, 60))


# --------------------------------------------------------------------------- target selection
def test_max_selection_reports_the_best_available_instance():
    gt = _square(box=(20, 40, 20, 40))
    gt = _square(box=(50, 70, 50, 70), label=2, base=gt)
    r = _rule(gt, oracle_isolation_select="max")
    assert r._isolation((50, 70, 50, 70)) == pytest.approx(1.0)


def test_center_selection_scores_the_instance_nearest_the_crop_centre():
    gt = _square(box=(20, 40, 20, 40))                          # instance 1
    gt = _square(box=(20, 30, 45, 55), label=2, base=gt)        # instance 2, smaller, to the right
    crop = (20, 40, 20, 40)
    by_max = _rule(gt, oracle_isolation_select="max")._isolation(crop)
    by_center = _rule(gt, oracle_isolation_select="center")._isolation(crop)
    assert by_max == pytest.approx(1.0)                          # instance 1 fills the crop
    assert by_center == pytest.approx(1.0)                       # ...and is also the nearest


def test_center_selection_ignores_instances_the_crop_does_not_touch():
    gt = _square(box=(20, 40, 20, 40))
    gt = _square(box=(80, 95, 80, 95), label=2, base=gt)        # far away, no overlap
    r = _rule(gt, oracle_isolation_select="center")
    assert r._isolation((20, 40, 20, 40)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- plumbing
def test_accept_is_true_exactly_when_some_instance_is_visible():
    r = _rule(_square())
    assert r.accept(0.0, (20, 40, 20, 40))
    assert not r.accept(1.0, (60, 90, 60, 90))


def test_bool_gt_degrades_to_one_instance_over_the_union():
    r = _rule(np.zeros((50, 50), bool) | _square((50, 50), (10, 20, 10, 20)).astype(bool))
    assert r._isolation((10, 20, 10, 20)) == pytest.approx(1.0)


def test_no_gt_injected_scores_zero_rather_than_crashing():
    r = build_stop_rule(Config(stop_rule="oracle"))
    assert r._isolation((0, 10, 0, 10)) == 0.0
