"""MERGE slot (:mod:`foveate.merge_rule`) — nms, soft, none."""

import numpy as np
import pytest

from foveate import Config, cascade
from foveate.merge_rule import (
    NmsMerge,
    NoMerge,
    SoftMerge,
    build_merge_rule,
    mask_overlap,
    nms,
)
from foveate.types import Instance


def _inst(mask: np.ndarray, score: float) -> Instance:
    ys, xs = np.where(mask)
    box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    return Instance(mask.astype(np.uint8), box, 0, score)


def _half_overlapping_pair():
    """Two equal-area detections at exactly the NMS IoU threshold, neither contained in the other.

    Equal areas 300, intersection 200 → IoU 0.50 (hard NMS deletes) and containment 0.67 (below the
    0.7 containment threshold, so it is a genuine partial overlap and not a nested duplicate).
    """
    H = W = 24
    a = np.zeros((H, W), bool); a[0:20, 0:15] = True
    b = np.zeros((H, W), bool); b[0:20, 5:20] = True
    return a, b


def _nested_trio():
    """A tight detection inside a looser one, plus a disjoint third."""
    H = W = 24
    loose = np.zeros((H, W), bool); loose[2:22, 2:22] = True
    tight = np.zeros((H, W), bool); tight[7:17, 7:17] = True    # nested inside loose
    far = np.zeros((H, W), bool); far[0:4, 20:24] = True        # disjoint
    return loose, tight, far


def test_mask_overlap_iou_and_containment():
    H = W = 20
    big = np.zeros((H, W), bool); big[2:18, 2:18] = True        # area 256
    small = np.zeros((H, W), bool); small[6:14, 6:14] = True    # area 64, fully inside big
    iou, contain = mask_overlap(big, small)
    assert abs(contain - 1.0) < 1e-9                            # the smaller is fully contained
    assert iou < 0.5                                            # ...but IoU is low (nested)
    disjoint = np.zeros((H, W), bool); disjoint[0:2, 0:2] = True
    assert mask_overlap(big, disjoint) == (0.0, 0.0)


# --------------------------------------------------------------------------- nms
def test_nms_suppresses_nested_duplicate_keeps_higher_score():
    """The nested-split failure mode: two detections of one object, one tight (higher g) inside a
    looser one. NMS keeps the higher-scoring (tighter) detection and drops the nested duplicate."""
    loose, tight, far = _nested_trio()
    kept, n = nms([_inst(loose, 0.7), _inst(tight, 0.9), _inst(far, 0.6)], 0.5, 0.7)
    assert n == 1
    assert {int(k.mask.sum()) for k in kept} == {int(tight.sum()), int(far.sum())}


def test_nms_keeps_distinct_instances():
    """Disjoint instances must never be merged (the clean two-instance case)."""
    H = W = 24
    a = np.zeros((H, W), bool); a[2:8, 2:8] = True
    b = np.zeros((H, W), bool); b[16:22, 16:22] = True
    kept, n = nms([_inst(a, 0.8), _inst(b, 0.7)], 0.5, 0.7)
    assert n == 0 and len(kept) == 2


def test_nms_rule_reports_what_it_removed():
    loose, tight, far = _nested_trio()
    rule = build_merge_rule(Config())
    assert isinstance(rule, NmsMerge)
    kept, merged, suppressed = rule.merge([_inst(loose, 0.7), _inst(tight, 0.9), _inst(far, 0.6)])
    assert (len(kept), merged, suppressed) == (2, 0, 1)


def test_nms_thresholds_at_one_suppress_nothing():
    loose, tight, far = _nested_trio()
    rule = build_merge_rule(Config(nms_iou=1.0, nms_containment=1.0))
    kept, _, suppressed = rule.merge([_inst(loose, 0.7), _inst(tight, 0.9), _inst(far, 0.6)])
    assert len(kept) == 3 and suppressed == 0


# --------------------------------------------------------------------------- soft
def test_soft_keeps_a_partial_overlap_that_nms_deletes_but_demotes_it():
    """Where hard suppression costs recall: two objects overlapping above the IoU threshold. Soft
    keeps the weaker one and ranks it below the stronger, rather than removing it."""
    a, b = _half_overlapping_pair()
    pair = [_inst(a, 0.9), _inst(b, 0.6)]

    hard, _, n_hard = build_merge_rule(Config()).merge(pair)
    soft, _, n_soft = build_merge_rule(Config(merge_rule="soft")).merge(pair)
    assert isinstance(build_merge_rule(Config(merge_rule="soft")), SoftMerge)
    assert len(hard) == 1 and n_hard == 1
    assert len(soft) == 2 and n_soft == 0
    assert soft[0].score == pytest.approx(0.9)              # the winner is untouched
    assert 0.0 < soft[1].score < 0.6                        # the overlapping one is decayed


def test_soft_still_deletes_a_fully_contained_duplicate():
    """A mask entirely inside a kept one carries no pixels of its own — decaying it is not enough."""
    loose, tight, far = _nested_trio()
    kept, _, suppressed = build_merge_rule(Config(merge_rule="soft")).merge(
        [_inst(loose, 0.7), _inst(tight, 0.9), _inst(far, 0.6)])
    assert suppressed == 1
    assert {int(k.mask.sum()) for k in kept} == {int(tight.sum()), int(far.sum())}


def test_soft_sigma_controls_how_hard_the_decay_is():
    a, b = _half_overlapping_pair()
    pair = [_inst(a, 0.9), _inst(b, 0.6)]
    gentle, _, _ = build_merge_rule(Config(merge_rule="soft", merge_soft_sigma=2.0)).merge(pair)
    harsh, _, _ = build_merge_rule(Config(merge_rule="soft", merge_soft_sigma=0.1)).merge(pair)
    assert gentle[1].score > harsh[1].score


# --------------------------------------------------------------------------- none / registry
def test_none_is_the_identity():
    loose, tight, far = _nested_trio()
    rule = build_merge_rule(Config(merge_rule="none"))
    assert isinstance(rule, NoMerge)
    given = [_inst(loose, 0.7), _inst(tight, 0.9), _inst(far, 0.6)]
    assert rule.merge(given) == (given, 0, 0)


def test_registry_rejects_unknown_merge_rule():
    with pytest.raises(ValueError, match="Unknown merge_rule"):
        build_merge_rule(Config(merge_rule="does-not-exist"))


# --------------------------------------------------------------------------- end to end
def test_merge_none_never_suppresses_in_the_cascade(backbone, two_squares):
    img, ex = two_squares
    _, stats = cascade(backbone, img, ex,
                       config=Config(merge_rule="none", min_crop=24,
                                     cascade_min_instance_area=4))
    assert stats.suppressed == 0 and stats.merged == 0


@pytest.mark.parametrize("name", ["nms", "soft", "none"])
def test_every_merge_rule_drives_the_cascade(backbone, two_squares, name):
    img, ex = two_squares
    instances, stats = cascade(backbone, img, ex,
                               config=Config(merge_rule=name, min_crop=24,
                                             cascade_min_instance_area=4))
    assert stats.n_embeds > 0
    assert all(inst.mask.shape == img.shape[:2] for inst in instances)


# --------------------------------------------------------------------------- oracle
def test_oracle_merge_keeps_one_best_detection_per_instance():
    """The ceiling of any merge rule: duplicates, fragments and background masks all go."""
    from foveate.merge_rule import OracleMerge

    H = W = 40
    labels = np.zeros((H, W), np.int32)
    labels[2:18, 2:18] = 1
    labels[22:38, 22:38] = 2

    exact1 = labels == 1
    loose1 = np.zeros((H, W), bool); loose1[0:20, 0:20] = True    # same object, worse IoU
    exact2 = labels == 2
    bg = np.zeros((H, W), bool); bg[0:3, 36:40] = True            # touches no instance

    rule = build_merge_rule(Config(merge_rule="oracle"))
    assert isinstance(rule, OracleMerge)
    rule.set_target_instances(labels)
    kept, _, suppressed = rule.merge(
        [_inst(loose1, 0.9), _inst(exact1, 0.2), _inst(exact2, 0.5), _inst(bg, 0.99)])

    assert suppressed == 2                                        # the loose duplicate and the bg one
    sums = {int(k.mask.sum()) for k in kept}
    assert sums == {int(exact1.sum()), int(exact2.sum())}         # best per instance, score ignored


def test_oracle_merge_without_gt_is_an_error_not_silent_nonsense():
    rule = build_merge_rule(Config(merge_rule="oracle"))
    a = np.zeros((8, 8), bool); a[1:4, 1:4] = True
    with pytest.raises(RuntimeError, match="set_target_instances"):
        rule.merge([_inst(a, 0.5)])


def test_oracle_merge_drives_the_cascade(backbone, two_squares):
    img, ex = two_squares
    labels = np.zeros(img.shape[:2], dtype=np.int32)
    labels[20:40, 20:40] = 1
    labels[80:100, 80:100] = 2
    instances, stats = cascade(backbone, img, ex, gt_foreground=labels,
                               config=Config(merge_rule="oracle", min_crop=24,
                                             cascade_min_instance_area=4))
    assert stats.n_embeds > 0
    assert len(instances) <= 2                                    # at most one per GT instance


# --------------------------------------------------------------------------- clipped fragments
def _crop_inst(mask, box, score):
    """An Instance carrying the CROP box it was found in (not its own tight box)."""
    return Instance(mask.astype(np.uint8), box, 0, score)


def test_clipped_detection_is_recognised_by_its_crop_border():
    from foveate.merge_rule import is_clipped

    H = W = 40
    whole = np.zeros((H, W), bool); whole[12:18, 12:18] = True     # interior to the crop
    piece = np.zeros((H, W), bool); piece[10:18, 12:18] = True     # reaches the crop's top edge
    crop = (10, 30, 10, 30)
    assert not is_clipped(_crop_inst(whole, crop, 0.9))
    assert is_clipped(_crop_inst(piece, crop, 0.9))


def test_an_image_border_is_not_a_clip():
    """A crop edge that IS an image edge is where the object genuinely ends."""
    from foveate.merge_rule import is_clipped

    H = W = 40
    at_edge = np.zeros((H, W), bool); at_edge[0:8, 12:18] = True
    assert not is_clipped(_crop_inst(at_edge, (0, 30, 10, 30), 0.9))   # top edge == image edge
    assert is_clipped(_crop_inst(at_edge, (0, 30, 10, 18), 0.9))       # ...but the right edge is not


def test_drop_clipped_runs_before_suppression_and_is_counted():
    H = W = 40
    whole = np.zeros((H, W), bool); whole[20:26, 12:18] = True      # interior to the crop
    piece = np.zeros((H, W), bool); piece[10:14, 12:18] = True      # reaches the crop's top edge
    crop = (10, 30, 10, 30)                                         # disjoint, so NMS never fires
    given = [_crop_inst(whole, crop, 0.9), _crop_inst(piece, crop, 0.8)]

    off, merged_off, _ = build_merge_rule(Config()).merge(given)
    on, merged_on, _ = build_merge_rule(Config(merge_drop_clipped=True)).merge(given)
    assert len(off) == 2 and merged_off == 0
    assert len(on) == 1 and merged_on == 1
    assert int(on[0].mask.sum()) == int(whole.sum())
