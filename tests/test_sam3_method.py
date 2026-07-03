"""Unit tests for the SAM 3 baseline's pure geometry (no model download required).

The mask->box, concat-canvas, and crop-back functions are module-level so they can be tested
in isolation — this is the highest-risk code (coordinate translation must be exact). Real-model
tests are skipped unless the transformers SAM 3 classes are importable.
"""

import numpy as np
import pytest

# The geometry helpers live in a module that imports transformers.models.sam3 at top level;
# skip everything if SAM 3 isn't installed (the module would raise ImportError on import).
sam3 = pytest.importorskip("experiments.methods.sam3")


# ---------------------------------------------------------------------------
# mask -> bbox
# ---------------------------------------------------------------------------
def test_masks_to_boxes_tight_and_skips_empty():
    m = np.zeros((10, 12), dtype=bool)
    m[2:5, 3:8] = True  # ys 2..4, xs 3..7
    empty = np.zeros((10, 12), dtype=bool)

    boxes = sam3.masks_to_boxes([m, empty, m])
    # xyxy, inclusive-min / exclusive-max; the empty mask is dropped.
    assert boxes == [[3, 2, 8, 5], [3, 2, 8, 5]]


def test_masks_to_boxes_single_pixel():
    m = np.zeros((5, 5), dtype=bool)
    m[4, 1] = True
    assert sam3.masks_to_boxes([m]) == [[1, 4, 2, 5]]


# ---------------------------------------------------------------------------
# concat canvas construction
# ---------------------------------------------------------------------------
def test_build_concat_canvas_offset_and_placement():
    # Differing H and W: exemplar 4x3, target 6x5 -> canvas 6x8, target offset = 3.
    exemplar = np.full((4, 3, 3), 7, dtype=np.uint8)
    target = np.full((6, 5, 3), 9, dtype=np.uint8)

    canvas, x_offset = sam3.build_concat_canvas(exemplar, target)
    assert canvas.shape == (6, 8, 3)
    assert x_offset == 3
    # Exemplar at origin (top-left 4x3), target offset right by exemplar width.
    assert np.all(canvas[:4, :3] == 7)
    assert np.all(canvas[:6, 3:8] == 9)
    # Padding under the shorter exemplar (rows 4..5, cols 0..2) stays zero.
    assert np.all(canvas[4:6, :3] == 0)


def test_build_concat_canvas_exemplar_taller():
    exemplar = np.full((7, 2, 3), 1, dtype=np.uint8)
    target = np.full((3, 4, 3), 2, dtype=np.uint8)
    canvas, x_offset = sam3.build_concat_canvas(exemplar, target)
    assert canvas.shape == (7, 6, 3)
    assert x_offset == 2
    # Target is top-aligned; rows 3..6 of the target column band are zero-padded.
    assert np.all(canvas[3:7, 2:6] == 0)


# ---------------------------------------------------------------------------
# crop canvas masks back to the target region  (the round-trip)
# ---------------------------------------------------------------------------
def test_crop_canvas_masks_round_trip():
    # Canvas geometry: exemplar 4x3, target 6x5 -> canvas 6x8, x_offset=3.
    exemplar = np.zeros((4, 3, 3), dtype=np.uint8)
    target = np.zeros((6, 5, 3), dtype=np.uint8)
    _, x_offset = sam3.build_concat_canvas(exemplar, target)
    th, tw = 6, 5

    # Detection A: a blob inside the target region (canvas cols 3..7, rows 0..5).
    a = np.zeros((6, 8), dtype=bool)
    a[1:3, 4:6] = True  # target-local rows 1..2, cols 1..2
    # Detection B: entirely in the exemplar half (cols 0..2) -> must be dropped.
    b = np.zeros((6, 8), dtype=bool)
    b[0:2, 0:2] = True

    canvas_masks = np.stack([a, b])
    canvas_scores = np.array([0.8, 0.4])

    masks, scores = sam3.crop_canvas_masks_to_target(
        canvas_masks, canvas_scores, x_offset, (th, tw)
    )
    # Only A survives; its cropped mask is (th, tw) with the blob at target-local coords.
    assert masks.shape == (1, th, tw)
    assert scores.tolist() == [0.8]
    expected = np.zeros((th, tw), dtype=bool)
    expected[1:3, 1:3] = True
    assert np.array_equal(masks[0], expected)


def test_crop_canvas_masks_drops_below_min_area():
    exemplar = np.zeros((4, 3, 3), dtype=np.uint8)
    target = np.zeros((6, 5, 3), dtype=np.uint8)
    _, x_offset = sam3.build_concat_canvas(exemplar, target)

    # A single target pixel is below min_target_area=2 -> dropped, leaving nothing.
    m = np.zeros((6, 8), dtype=bool)
    m[0, 4] = True  # one pixel in the target region
    masks, scores = sam3.crop_canvas_masks_to_target(
        np.stack([m]), np.array([0.9]), x_offset, (6, 5), min_target_area=2
    )
    assert masks.shape == (0, 6, 5)
    assert scores.shape == (0,)


def test_crop_canvas_masks_empty_input():
    masks, scores = sam3.crop_canvas_masks_to_target(
        np.zeros((0, 6, 8), dtype=bool), np.zeros((0,)), 3, (6, 5)
    )
    assert masks.shape == (0, 6, 5)
    assert scores.shape == (0,)


# ---------------------------------------------------------------------------
# real-model smoke test (only when SAM 3 + weights are actually usable)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="SAM 3 weights are heavy; skip real-model test without CUDA.",
)
def test_sam3_registered():
    # Registration works purely from importing the module (no model load here).
    from experiments.methods.base import _METHODS

    assert "sam3" in _METHODS
