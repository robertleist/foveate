"""Classification of undetected GT instances into search / emit / ranking failures."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.miss_diagnostics import MissBreakdown, classify_misses


def _mask(box, hw=(100, 100)):
    m = np.zeros(hw, dtype=bool)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = True
    return m


INST = (20, 40, 20, 40)


def test_detected_when_a_surviving_mask_matches():
    d = classify_misses([_mask(INST)], [_mask(INST)], [_mask(INST)], [INST])
    assert (d.detected, d.suppressed, d.not_emitted, d.never_framed) == (1, 0, 0, 0)


def test_suppressed_when_the_match_exists_only_before_nms():
    """Emitted correctly, then removed by the score-ranked NMS — a ranking problem."""
    d = classify_misses([_mask(INST)], pre_masks=[_mask(INST)], post_masks=[], visited_boxes=[INST])
    assert (d.detected, d.suppressed, d.not_emitted, d.never_framed) == (0, 1, 0, 0)


def test_not_emitted_when_a_crop_framed_it_but_no_mask_matches():
    """The recursion looked in the right place; the emit path produced nothing usable."""
    d = classify_misses([_mask(INST)], pre_masks=[], post_masks=[], visited_boxes=[INST])
    assert (d.detected, d.suppressed, d.not_emitted, d.never_framed) == (0, 0, 1, 0)


def test_never_framed_when_no_visited_crop_matches_the_instance():
    """The recursion never looked there — a search problem, not an emit problem."""
    d = classify_misses([_mask(INST)], pre_masks=[], post_masks=[],
                        visited_boxes=[(70, 90, 70, 90)])
    assert (d.detected, d.suppressed, d.not_emitted, d.never_framed) == (0, 0, 0, 1)


def test_a_loose_crop_containing_the_instance_is_not_counted_as_framed():
    """A whole-image crop 'contains' everything; framing means the crop is ABOUT the instance."""
    d = classify_misses([_mask(INST)], pre_masks=[], post_masks=[],
                        visited_boxes=[(0, 100, 0, 100)])
    assert d.never_framed == 1
    assert d.seen_but_never_framed == 1        # ...but recorded as 'the search passed over it'


def test_seen_but_never_framed_stays_zero_when_no_crop_contained_it():
    d = classify_misses([_mask(INST)], pre_masks=[], post_masks=[],
                        visited_boxes=[(70, 90, 70, 90)])
    assert d.never_framed == 1 and d.seen_but_never_framed == 0


def test_a_poor_overlapping_mask_does_not_count_as_a_detection():
    """Below the AP match threshold, so the instance is still undetected."""
    sliver = _mask((20, 24, 20, 40))            # 20 % of the instance
    d = classify_misses([_mask(INST)], pre_masks=[sliver], post_masks=[sliver],
                        visited_boxes=[INST])
    assert d.detected == 0 and d.not_emitted == 1


def test_categories_are_exclusive_and_sum_to_the_gt_count():
    gt = [_mask(INST), _mask((60, 80, 60, 80)), _mask((10, 15, 80, 90))]
    d = classify_misses(gt, pre_masks=[_mask(INST), _mask((60, 80, 60, 80))],
                        post_masks=[_mask(INST)], visited_boxes=[INST, (60, 80, 60, 80)])
    assert d.n_gt == 3
    assert d.detected + d.suppressed + d.not_emitted + d.never_framed == 3
    assert (d.detected, d.suppressed, d.not_emitted, d.never_framed) == (1, 1, 0, 1)


def test_merge_pools_counts_across_images():
    a = classify_misses([_mask(INST)], [_mask(INST)], [_mask(INST)], [INST])
    b = classify_misses([_mask(INST)], [], [], [(70, 90, 70, 90)])
    pooled = a.merge(b)
    assert pooled.n_gt == 2 and pooled.detected == 1 and pooled.never_framed == 1
    assert len(pooled.per_image) == 2


def test_metrics_are_fractions_of_the_gt_count():
    d = classify_misses([_mask(INST), _mask((60, 80, 60, 80))],
                        [_mask(INST)], [_mask(INST)], [INST])
    m = d.to_metrics("intra")
    assert m["intra_miss_detected_frac"] == pytest.approx(0.5)
    assert m["intra_miss_n_gt"] == 2.0


def test_no_gt_is_empty_not_an_error():
    d = classify_misses([], [], [], [])
    assert d.n_gt == 0 and isinstance(d, MissBreakdown)
    assert "0 GT instances" in d.report()


def test_report_mentions_every_category():
    d = classify_misses([_mask(INST)], [], [], [(70, 90, 70, 90)])
    text = d.report()
    for name in ("detected", "suppressed by NMS", "framed but not emitted", "never framed"):
        assert name in text
