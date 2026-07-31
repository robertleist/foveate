"""The monolithic segmenter arms of the Extract slot (:mod:`foveate.sam3`, :mod:`foveate.ntt`).

Only the pure geometry / matching logic is exercised here — no SAM weights are downloaded. The
model-touching classes are covered by the end-to-end runs recorded in the roadmap.
"""

import numpy as np
import pytest
import torch

from foveate import Config
from foveate.extract import ExtractResult, resolve_extractor_name
from foveate.sam3 import build_concat_canvas, crop_canvas_masks_to_target, masks_to_boxes


def test_registry_knows_the_segmenter_arms():
    for name in ("sam3", "ntt"):
        assert resolve_extractor_name(Config(extractor=name)) == name
    # ...and they are NOT reachable by accident from the legacy key pair.
    assert resolve_extractor_name(Config()) == "composite"


def test_extract_result_carries_optional_pixel_masks():
    """A segmenter's pixel masks must survive the patch grid, or the slot is lossy again."""
    grid = np.zeros((4, 4), dtype=bool); grid[1:3, 1:3] = True
    px = np.zeros((40, 40), dtype=bool); px[10:30, 10:30] = True
    res = ExtractResult([grid], grid, grid.astype(np.float32), masks=[px])
    assert res.masks is not None and res.masks[0].shape == (40, 40)
    assert ExtractResult([grid], grid, grid.astype(np.float32)).masks is None


# --------------------------------------------------------------------------- SAM 3 geometry
def test_masks_to_boxes_skips_empty_masks():
    a = np.zeros((10, 10), bool); a[2:5, 3:7] = True
    assert masks_to_boxes([a, np.zeros((10, 10), bool)]) == [[3, 2, 7, 5]]


def test_concat_canvas_leaves_the_exemplar_coordinates_alone():
    """The exemplar goes left at the origin, so its prompt boxes need no transform."""
    left = np.full((6, 4, 3), 7, np.uint8)
    right = np.full((8, 5, 3), 9, np.uint8)
    canvas, x_offset = build_concat_canvas(left, right)
    assert canvas.shape == (8, 9, 3) and x_offset == 4
    assert np.array_equal(canvas[:6, :4], left)
    assert np.array_equal(canvas[:8, 4:9], right)


def test_canvas_masks_are_cut_back_to_the_target_half():
    """A mask that landed on the exemplar is a detection of the prompt, not of the target."""
    x_offset, (h, w) = 4, (8, 5)
    on_target = np.zeros((8, 9), bool); on_target[1:4, 5:8] = True
    on_exemplar = np.zeros((8, 9), bool); on_exemplar[1:4, 0:3] = True
    masks, scores = crop_canvas_masks_to_target(
        np.stack([on_target, on_exemplar]), np.array([0.9, 0.8]), x_offset, (h, w))
    assert masks.shape == (1, h, w) and scores.tolist() == [0.9]
    assert masks[0][1:4, 1:4].all()                       # shifted into target coordinates


def test_canvas_masks_handle_the_empty_case():
    masks, scores = crop_canvas_masks_to_target(np.zeros((0, 8, 9), bool), np.zeros((0,)), 4, (8, 5))
    assert masks.shape == (0, 8, 5) and scores.shape == (0,)


# --------------------------------------------------------------------------- NTT matching
def test_memory_and_heatmap_find_the_concept():
    from foveate.ntt import similarity_heatmap, spherical_kmeans

    feats = torch.zeros(20, 4)
    feats[:10, 0] = 1.0                                   # two well-separated modes
    feats[10:, 1] = 1.0
    centers = spherical_kmeans(feats, 2)
    assert centers.shape == (2, 4)

    grid = torch.zeros(3, 3, 4)
    grid[0, 0, 0] = 1.0                                   # matches mode one
    heat = similarity_heatmap(grid, centers)
    assert heat.shape == (3, 3)
    assert heat[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert heat[2, 2] < heat[0, 0]


def test_empty_memory_yields_no_prompts():
    from foveate.ntt import grid_topk_points, similarity_heatmap

    grid = torch.zeros(3, 3, 4)
    heat = similarity_heatmap(grid, torch.zeros(0, 4))
    assert float(heat.min()) == -1.0
    assert grid_topk_points(heat, 8, thr=0.5).shape == (0, 2)


def test_grid_points_map_to_patch_centres():
    """Same convention as the rest of the grid <-> pixel code: a cell is addressed at its middle."""
    from foveate.ntt import grid_points_to_pixels

    pts = grid_points_to_pixels(np.array([[0, 0], [1, 3]]), (2, 4), (20, 40))
    assert pts.tolist() == [[5.0, 5.0], [35.0, 15.0]]
