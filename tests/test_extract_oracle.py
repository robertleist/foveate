"""The monolithic Extract-slot oracle and the mask refinement step."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from foveate.config import Config
from foveate.extract import OracleExtractor, build_extractor
from foveate.mask_refine import refine_mask


def _labels(hw, boxes):
    """GT instance-label map: label i+1 on box i."""
    out = np.zeros(hw, dtype=np.int32)
    for i, (y0, y1, x0, x1) in enumerate(boxes, start=1):
        out[y0:y1, x0:x1] = i
    return out


def _oracle(labels=None, **cfg):
    ex = build_extractor(Config(extractor="oracle", **cfg))
    if labels is not None:
        ex.set_target_instances(labels)
    return ex


def _extract(ex, grid_hw, box):
    """Run the oracle on a dummy patch grid of ``grid_hw`` over ``box`` → the instance list."""
    return ex.extract(torch.zeros(*grid_hw, 1), box=box).instances


# --------------------------------------------------------------------------- registry
def test_oracle_is_registered():
    assert isinstance(_oracle(), OracleExtractor)


def test_using_the_oracle_without_gt_is_an_error_not_silent_nonsense():
    with pytest.raises(RuntimeError, match="set_target_foreground"):
        _extract(_oracle(), (4, 4), (0, 8, 0, 8))


# --------------------------------------------------------------------------- instances
def test_returns_touching_instances_that_connected_components_would_merge():
    """Two adjacent instances form ONE connected component; the oracle still returns two."""
    ex = _oracle(_labels((8, 8), [(0, 8, 0, 4), (0, 8, 4, 8)]))
    parts = _extract(ex, (4, 4), (0, 8, 0, 8))
    assert len(parts) == 2
    assert not (parts[0] & parts[1]).any()                 # disjoint
    assert (parts[0] | parts[1]).all()                     # and together they cover the region


def test_the_region_is_exactly_the_union_of_the_instances():
    ex = _oracle(_labels((8, 8), [(0, 4, 0, 4), (4, 8, 4, 8)]))
    res = ex.extract(torch.zeros(4, 4, 1), box=(0, 8, 0, 8))
    assert np.array_equal(np.logical_or.reduce(res.instances), res.foreground)


def test_background_only_crop_is_returned_whole_for_stop_to_reject():
    ex = _oracle(np.zeros((8, 8), dtype=np.int32))         # no instances anywhere
    # The GT is empty, so the region is empty too and there is nothing to propose.
    assert _extract(ex, (4, 4), (0, 8, 0, 8)) == []


def test_a_crop_with_no_gt_yields_no_instances():
    ex = _oracle(_labels((16, 16), [(0, 4, 0, 4)]))
    assert _extract(ex, (4, 4), (8, 16, 8, 16)) == []


def test_box_selects_the_region_of_the_gt_that_is_looked_up():
    """A crop of the right half must see only the instance living there."""
    ex = _oracle(_labels((8, 8), [(0, 8, 0, 4), (0, 8, 4, 8)]))
    assert len(_extract(ex, (4, 4), (0, 8, 4, 8))) == 1


def test_bool_gt_degrades_to_a_single_instance():
    ex = _oracle(np.ones((8, 8), dtype=bool))
    assert len(_extract(ex, (4, 4), (0, 8, 0, 8))) == 1


# --------------------------------------------------------------------------- coverage
def test_any_coverage_keeps_an_instance_smaller_than_a_patch():
    """The §A0.4 point: extraction is a PROPOSAL, so a sub-patch object must still mark a patch.

    Centre sampling deletes it before the recursion can see it — no patch, no crop, no instance.
    """
    labels = _labels((64, 64), [(0, 60, 0, 60)])           # a big one...
    labels[62:64, 62:64] = 2                               # ...and one smaller than a patch (8x8 px)
    assert len(_extract(_oracle(labels, oracle_coverage="any"), (8, 8), (0, 64, 0, 64))) == 2
    assert len(_extract(_oracle(labels, oracle_coverage="center"), (8, 8), (0, 64, 0, 64))) == 1


def test_any_coverage_instances_may_overlap_on_a_shared_patch():
    """Two objects sharing a patch both claim it — they genuinely need two crops."""
    # 4 px patches; the boundary at x=9 falls inside patch column 2 (x 8..11), so both instances
    # hold pixels in it.
    labels = _labels((16, 16), [(0, 16, 0, 9), (0, 16, 9, 16)])
    parts = _extract(_oracle(labels, oracle_coverage="any"), (4, 4), (0, 16, 0, 16))
    assert len(parts) == 2
    assert (parts[0] & parts[1]).any()


# --------------------------------------------------------------------------- mask refinement
def _image_with_square(hw, box, fg=220, bg=30):
    img = np.full((*hw, 3), bg, dtype=np.uint8)
    y0, y1, x0, x1 = box
    img[y0:y1, x0:x1] = fg
    return img


def test_refine_none_is_the_identity():
    mask = np.zeros((40, 40), dtype=bool); mask[10:30, 10:30] = True
    img = _image_with_square((40, 40), (12, 28, 12, 28))
    assert np.array_equal(refine_mask(mask, img, Config()), mask)


def test_grabcut_pulls_a_coarse_mask_toward_the_image_boundary():
    """A patch-grid mask overshooting the object should shrink onto it."""
    truth = (16, 32, 16, 32)
    img = _image_with_square((48, 48), truth)
    coarse = np.zeros((48, 48), dtype=bool)
    coarse[12:36, 12:36] = True                            # 4 px of slop on every side

    out = refine_mask(coarse, img, Config(mask_refine="grabcut", mask_refine_band=2))
    gt = np.zeros((48, 48), dtype=bool); gt[16:32, 16:32] = True

    def iou(a, b):
        return (a & b).sum() / (a | b).sum()

    assert iou(out, gt) > iou(coarse, gt)


def test_grabcut_keeps_the_input_when_the_mask_is_empty():
    img = _image_with_square((32, 32), (8, 24, 8, 24))
    empty = np.zeros((32, 32), dtype=bool)
    assert not refine_mask(empty, img, Config(mask_refine="grabcut")).any()


def test_grabcut_rejects_a_refinement_that_changes_the_area_drastically():
    """A uniform image gives GrabCut no signal; the guard must return the input unchanged."""
    img = np.full((40, 40, 3), 128, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool); mask[10:30, 10:30] = True
    out = refine_mask(mask, img, Config(mask_refine="grabcut", mask_refine_max_change=0.2))
    assert abs(out.sum() - mask.sum()) / mask.sum() <= 0.2


def test_unknown_refine_method_is_rejected():
    img = _image_with_square((16, 16), (4, 12, 4, 12))
    mask = np.zeros((16, 16), dtype=bool); mask[4:12, 4:12] = True
    with pytest.raises(ValueError, match="Unknown mask_refine"):
        refine_mask(mask, img, Config(mask_refine="crf"))
