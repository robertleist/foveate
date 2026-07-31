"""MASK slot (:mod:`foveate.upsample`) — nearest, bilinear, and the shape-only oracle."""

import numpy as np
import pytest

from foveate import Config, cascade
from foveate.upsample import (
    BilinearUpsampler,
    NearestUpsampler,
    OracleUpsampler,
    build_mask_upsampler,
)

_BOX = (0, 40, 0, 40)


def _grid(shape=(8, 8), y=slice(2, 6), x=slice(2, 6)):
    g = np.zeros(shape, dtype=bool)
    g[y, x] = True
    return g


def test_nearest_keeps_the_patch_staircase():
    up = build_mask_upsampler(Config())
    assert isinstance(up, NearestUpsampler) and not isinstance(up, BilinearUpsampler)
    out = up.upsample(_grid(), _BOX)
    assert out.shape == (40, 40)
    assert set(np.unique(out)) <= {0, 1}


def test_bilinear_smooths_the_boundary_without_moving_the_object():
    near = build_mask_upsampler(Config(mask_upsample="nearest")).upsample(_grid(), _BOX)
    bil = build_mask_upsampler(Config(mask_upsample="bilinear")).upsample(_grid(), _BOX)
    assert isinstance(build_mask_upsampler(Config(mask_upsample="bilinear")), BilinearUpsampler)
    # Same object, different edge: overlapping heavily but not identical.
    inter = np.logical_and(near.astype(bool), bil.astype(bool)).sum()
    assert inter / max(near.sum(), 1) > 0.7
    assert not np.array_equal(near, bil)


# --------------------------------------------------------------------------- the oracle
def _labels(shape=(40, 40)):
    lab = np.zeros(shape, np.int32)
    lab[10:30, 10:30] = 1                      # a square, deliberately not patch-aligned below
    return lab


def test_oracle_replaces_the_shape_of_a_detection_that_already_matches():
    up = build_mask_upsampler(Config(mask_upsample="oracle"))
    assert isinstance(up, OracleUpsampler)
    up.set_target_instances(_labels())
    # A grid whose 5x5-patch block lands on the GT square → base mask already matches.
    out = up.upsample(_grid(y=slice(2, 6), x=slice(2, 6)), _BOX).astype(bool)
    assert np.array_equal(out, _labels() == 1)  # snapped exactly to the instance


def test_oracle_is_shape_only_and_never_rescues_a_non_match():
    """The discipline that keeps the measurement honest: it may not do the Merge slot's job.

    A clipped fragment overlapping the object below the gate must come back unchanged — otherwise a
    *search* gain would be booked as boundary quality.
    """
    up = build_mask_upsampler(Config(mask_upsample="oracle", oracle_upsample_iou=0.5))
    up.set_target_instances(_labels())
    sliver = _grid(y=slice(2, 3), x=slice(2, 3))          # a corner of the object → IoU well under 0.5
    base = build_mask_upsampler(Config(mask_upsample="bilinear")).upsample(sliver, _BOX)
    assert np.array_equal(up.upsample(sliver, _BOX), base)
    # ...and lowering the gate lets the same fragment snap, which is why the gate exists.
    loose = build_mask_upsampler(Config(mask_upsample="oracle", oracle_upsample_iou=0.0))
    loose.set_target_instances(_labels())
    assert np.array_equal(loose.upsample(sliver, _BOX).astype(bool), _labels() == 1)


def test_oracle_leaves_a_background_detection_alone():
    up = build_mask_upsampler(Config(mask_upsample="oracle"))
    up.set_target_instances(_labels())
    bg = _grid(y=slice(6, 8), x=slice(0, 2))              # touches no instance
    base = build_mask_upsampler(Config(mask_upsample="bilinear")).upsample(bg, _BOX)
    assert np.array_equal(up.upsample(bg, _BOX), base)


def test_oracle_without_gt_is_an_error_not_silent_nonsense():
    up = build_mask_upsampler(Config(mask_upsample="oracle"))
    with pytest.raises(RuntimeError, match="set_target_instances"):
        up.upsample(_grid(), _BOX)


def test_registry_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown mask_upsample"):
        build_mask_upsampler(Config(mask_upsample="does-not-exist"))


@pytest.mark.parametrize("name", ["nearest", "bilinear", "oracle"])
def test_every_arm_drives_the_cascade(backbone, two_squares, name):
    img, ex = two_squares
    labels = np.zeros(img.shape[:2], dtype=np.int32)
    labels[20:40, 20:40] = 1
    labels[80:100, 80:100] = 2
    instances, stats = cascade(backbone, img, ex, gt_foreground=labels,
                               config=Config(mask_upsample=name, min_crop=24,
                                             cascade_min_instance_area=4))
    assert stats.n_embeds > 0
    assert all(inst.mask.shape == img.shape[:2] for inst in instances)
