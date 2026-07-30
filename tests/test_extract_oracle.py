"""The Extract-slot oracle and the mask refinement step."""

from __future__ import annotations

import numpy as np
import pytest

from foveate.config import Config
from foveate.extract import OracleInstanceExtractor, build_instance_extractor
from foveate.mask_refine import refine_mask


def _labels(hw, boxes):
    """GT instance-label map: label i+1 on box i."""
    out = np.zeros(hw, dtype=np.int32)
    for i, (y0, y1, x0, x1) in enumerate(boxes, start=1):
        out[y0:y1, x0:x1] = i
    return out


# --------------------------------------------------------------------------- registry
def test_oracle_is_registered_and_can_split():
    ex = build_instance_extractor(Config(instance_extractor="oracle"))
    assert isinstance(ex, OracleInstanceExtractor)
    assert ex.can_split


def test_using_the_oracle_without_gt_is_an_error_not_silent_nonsense():
    ex = build_instance_extractor(Config(instance_extractor="oracle"))
    with pytest.raises(RuntimeError, match="set_target_instances"):
        ex.components(np.ones((4, 4), dtype=bool), box=(0, 8, 0, 8))


# --------------------------------------------------------------------------- components
def test_components_separates_touching_instances_that_cc_would_merge():
    """Two adjacent instances form ONE connected component; the oracle still returns two."""
    ex = build_instance_extractor(Config(instance_extractor="oracle"))
    ex.set_target_instances(_labels((8, 8), [(0, 8, 0, 4), (0, 8, 4, 8)]))
    fg = np.ones((4, 4), dtype=bool)                       # foreground covers both

    cc = build_instance_extractor(Config(instance_extractor="cc")).components(fg)
    assert len(cc) == 1                                    # connected components cannot cut inside

    parts = ex.components(fg, box=(0, 8, 0, 8))
    assert len(parts) == 2
    assert not (parts[0] & parts[1]).any()                 # disjoint
    assert (parts[0] | parts[1]).all()                     # and together they cover the foreground


def test_components_never_adds_patches_the_where_stage_missed():
    """The Extract slot decomposes a foreground; it must not repair one."""
    ex = build_instance_extractor(Config(instance_extractor="oracle"))
    ex.set_target_instances(_labels((8, 8), [(0, 8, 0, 4), (0, 8, 4, 8)]))
    fg = np.zeros((4, 4), dtype=bool)
    fg[:, :2] = True                                       # Where found only the left instance

    parts = ex.components(fg, box=(0, 8, 0, 8))
    assert len(parts) == 1
    assert np.array_equal(parts[0], fg)                    # exactly what it was given, no more


def test_background_only_foreground_is_returned_whole_for_stop_to_reject():
    ex = build_instance_extractor(Config(instance_extractor="oracle"))
    ex.set_target_instances(np.zeros((8, 8), dtype=np.int32))   # no instances anywhere
    fg = np.ones((4, 4), dtype=bool)
    parts = ex.components(fg, box=(0, 8, 0, 8))
    assert len(parts) == 1 and parts[0].all()


def test_empty_foreground_yields_no_components():
    ex = build_instance_extractor(Config(instance_extractor="oracle"))
    ex.set_target_instances(_labels((8, 8), [(0, 4, 0, 4)]))
    assert ex.components(np.zeros((4, 4), dtype=bool), box=(0, 8, 0, 8)) == []


def test_box_selects_the_region_of_the_gt_that_is_looked_up():
    """A crop of the right half must see only the instance living there."""
    ex = build_instance_extractor(Config(instance_extractor="oracle"))
    ex.set_target_instances(_labels((8, 8), [(0, 8, 0, 4), (0, 8, 4, 8)]))
    parts = ex.components(np.ones((4, 4), dtype=bool), box=(0, 8, 4, 8))
    assert len(parts) == 1


# --------------------------------------------------------------------------- split
def test_split_returns_one_grid_per_instance_in_the_clump():
    ex = build_instance_extractor(Config(instance_extractor="oracle"))
    ex.set_target_instances(_labels((8, 8), [(0, 8, 0, 4), (0, 8, 4, 8)]))
    subs = ex.split(None, np.ones((4, 4), dtype=bool), box=(0, 8, 0, 8))
    assert len(subs) == 2


def test_split_of_a_single_instance_reports_unsplittable():
    ex = build_instance_extractor(Config(instance_extractor="oracle"))
    ex.set_target_instances(_labels((8, 8), [(0, 8, 0, 8)]))
    comp = np.ones((4, 4), dtype=bool)
    subs = ex.split(None, comp, box=(0, 8, 0, 8))
    assert len(subs) == 1 and np.array_equal(subs[0], comp)


def test_bool_gt_degrades_to_a_single_instance():
    ex = build_instance_extractor(Config(instance_extractor="oracle"))
    ex.set_target_instances(np.ones((8, 8), dtype=bool))
    assert len(ex.components(np.ones((4, 4), dtype=bool), box=(0, 8, 0, 8))) == 1


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
