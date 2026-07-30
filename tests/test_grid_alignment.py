"""Patch-grid ↔ pixel alignment, and the end-to-end oracle sanity check.

Two bugs lived here and both were silent — no exception, no warning, just instances that quietly
never got detected. These tests exist so they cannot come back.

1. **Sampling point.** A patch covers pixels ``[j·P, (j+1)·P)`` and every consumer maps it back to
   that interval, so a mask must be sampled at the interval's **centre**. ``cv2.INTER_NEAREST``
   samples the **left edge**, which offsets every derived crop box by half a patch.
2. **Two conventions in one pipeline.** The Where stage and the Extract oracle sampled the ground
   truth at points half a patch apart, so patches Where called foreground read as background in
   Extract and whole instances vanished from the decomposition. There is one oracle now (the Extract
   slot is one slot), so the two cannot diverge — but the invariant is still pinned below: its
   foreground and its instances must be the same region, decomposed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from foveate import features as featlib
from foveate.backbones.mock import MockBackbone
from foveate.cascade import _child_box, cascade
from foveate.config import Config
from foveate.extract import build_extractor


# --------------------------------------------------------------------------- sampling point
def test_grid_sampling_uses_the_patch_centre():
    """A 1-D ramp makes the sample position directly visible."""
    labels = np.arange(64, dtype=np.int32)[None, :].repeat(2, axis=0)
    grid = featlib.resize_labels_to_grid(labels, (2, 8))
    # 8 cells over 64 px → cell width 8, centres at 4, 12, 20, ...
    assert list(grid[0]) == [4, 12, 20, 28, 36, 44, 52, 60]


def test_mask_and_label_resize_agree():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=(97, 61)).astype(np.int32)
    grid_labels = featlib.resize_labels_to_grid(labels, (12, 9))
    grid_mask = featlib.resize_mask_to_grid(labels > 0, (12, 9))
    assert np.array_equal(grid_labels > 0, grid_mask)


def test_a_patch_marked_foreground_maps_back_onto_the_object():
    """The round trip mask → grid → child box must land ON the object, not beside it."""
    n = 512
    mask = np.zeros((n, n), dtype=bool)
    mask[200:260, 200:260] = True                       # a 60 px square

    grid = featlib.resize_mask_to_grid(mask, (32, 32))
    box = _child_box(grid, (0, n, 0, n), pad_frac=0.0, dilate=0.0)
    y0, y1, x0, x1 = box
    covered = mask[y0:y1, x0:x1].sum() / mask.sum()
    assert covered > 0.5, f"child box {box} covers only {covered:.0%} of the object"


def test_left_edge_sampling_loses_object_coverage():
    """Pin the defect itself: the old convention lands the crop off-centre by half a patch.

    The damage scales with patch size relative to the object — an instance a couple of patches wide
    (the regime the cascade is *for*) loses most of itself.
    """
    import cv2

    n = 512
    for side in (24, 40, 60):
        mask = np.zeros((n, n), dtype=bool)
        mask[200:200 + side, 200:200 + side] = True

        old = cv2.resize(mask.astype(np.uint8), (32, 32),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        new = featlib.resize_mask_to_grid(mask, (32, 32))

        def covered(grid):
            y0, y1, x0, x1 = _child_box(grid, (0, n, 0, n), pad_frac=0.0, dilate=0.0)
            return mask[y0:y1, x0:x1].sum() / mask.sum()

        assert covered(new) > covered(old), f"side={side}px: alignment fix did not improve coverage"
    # the smallest object — ~1.5 patches — is where the old convention does real damage
    mask = np.zeros((n, n), dtype=bool)
    mask[200:224, 200:224] = True
    old = cv2.resize(mask.astype(np.uint8), (32, 32), interpolation=cv2.INTER_NEAREST).astype(bool)
    y0, y1, x0, x1 = _child_box(old, (0, n, 0, n), pad_frac=0.0, dilate=0.0)
    assert mask[y0:y1, x0:x1].sum() / mask.sum() < 0.6


# ------------------------------------------------------------ foreground and instances agree
def test_oracle_foreground_and_instances_are_the_same_region():
    """The bug that dropped instances: the region says N objects, the decomposition must return N.

    It used to be two objects sampling the GT half a patch apart; it is one object now, so what is
    pinned is that its own two outputs still agree — and with the shared ``resize_mask_to_grid``.
    """
    n = 512
    labels = np.zeros((n, n), np.int32)
    for i, (y, x) in enumerate([(40, 40), (40, 200), (40, 380), (300, 100), (300, 300)], start=1):
        labels[y:y + 50, x:x + 50] = i

    ex = build_extractor(Config(extractor="oracle"))
    ex.set_target_instances(labels)
    res = ex.extract(torch.zeros(14, 14, 1), box=(0, n, 0, n))
    assert len(res.instances) == 5
    assert np.array_equal(np.logical_or.reduce(res.instances), res.foreground)
    assert np.array_equal(res.foreground,
                          featlib.resize_mask_to_grid(labels > 0, (14, 14), mode="any"))


# --------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("grid", [24, 48])
def test_all_oracle_cascade_recovers_every_separated_instance(grid):
    """With a perfect Where, Extract and Stop, five separated squares must all come back."""
    n = 512
    image = np.full((n, n, 3), 30, np.uint8)
    labels = np.zeros((n, n), np.int32)
    boxes = [(40, 40), (40, 200), (40, 380), (300, 100), (300, 300)]
    for i, (y, x) in enumerate(boxes, start=1):
        image[y:y + 50, x:x + 50] = (200, 60, 60)
        labels[y:y + 50, x:x + 50] = i
    masks = [labels == i for i in range(1, 6)]

    instances, _ = cascade(
        MockBackbone(image_size=grid * 16, patch_size=16), image, [masks[0]],
        config=Config(foreground_extractor="oracle", instance_extractor="oracle",
                      stop_rule="oracle", crop_dilate=0.0, min_crop=8,
                      cascade_min_instance_area=4),
        gt_foreground=labels,
    )
    assert len(instances) == 5

    def iou(a, b):
        a, b = np.asarray(a, bool), np.asarray(b, bool)
        return (a & b).sum() / max((a | b).sum(), 1)

    for m in masks:
        assert max(iou(inst.mask, m) for inst in instances) > 0.75


def test_all_oracle_needs_no_crop_padding():
    """crop_dilate compensated for the alignment bug; with it fixed, 0 must be fine."""
    n = 512
    image = np.full((n, n, 3), 30, np.uint8)
    labels = np.zeros((n, n), np.int32)
    for i, (y, x) in enumerate([(40, 40), (300, 300)], start=1):
        image[y:y + 50, x:x + 50] = (200, 60, 60)
        labels[y:y + 50, x:x + 50] = i

    instances, _ = cascade(
        MockBackbone(image_size=768, patch_size=16), image, [labels == 1],
        config=Config(foreground_extractor="oracle", instance_extractor="oracle",
                      stop_rule="oracle", crop_dilate=0.0, min_crop=8,
                      cascade_min_instance_area=4),
        gt_foreground=labels,
    )
    assert len(instances) == 2
