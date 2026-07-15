"""Unit tests for the "No Time to Train!" baseline's pure matching logic (no weights needed).

The memory / heatmap / point-mapping / scoring / suppression functions are module-level so the
whole matching pipeline is testable without downloading SAM 2 or DINO. The module imports
transformers.models.sam2 at top level, so the suite skips if SAM 2 isn't installed.
"""

import numpy as np
import pytest
import torch

ntt = pytest.importorskip("experiments.methods.no_time_to_train")


def _grid(vectors: list[list[float]], hw: tuple[int, int]) -> torch.Tensor:
    """Build an ``(Hp, Wp, D)`` L2-normalized grid from a flat list of row-major patch vectors."""
    hp, wp = hw
    t = torch.tensor(vectors, dtype=torch.float32).reshape(hp, wp, -1)
    return t / (t.norm(dim=-1, keepdim=True) + 1e-8)


# ---------------------------------------------------------------------------
# spherical_kmeans
# ---------------------------------------------------------------------------
def test_spherical_kmeans_clamps_k_and_normalizes():
    feats = torch.nn.functional.normalize(torch.randn(3, 5), dim=-1)
    centers = ntt.spherical_kmeans(feats, k=8)          # k clamped to M=3
    assert centers.shape == (3, 5)
    assert torch.allclose(centers.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_spherical_kmeans_recovers_two_clusters():
    # Two tight antipodal blobs -> two centers pointing along +x and -x.
    a = torch.tensor([[1.0, 0.0], [0.98, 0.02], [0.99, -0.01]])
    b = torch.tensor([[-1.0, 0.0], [-0.97, 0.03], [-0.99, 0.01]])
    feats = torch.nn.functional.normalize(torch.cat([a, b]), dim=-1)
    centers = ntt.spherical_kmeans(feats, k=2, seed=0)
    xs = sorted(c[0].item() for c in centers)
    assert xs[0] < -0.9 and xs[1] > 0.9


# ---------------------------------------------------------------------------
# build_memory
# ---------------------------------------------------------------------------
def test_build_memory_empty_masks_returns_no_centers():
    grid = _grid([[1, 0], [0, 1], [1, 1], [0, 0]], (2, 2))
    centers = ntt.build_memory(grid, [np.zeros((8, 8), bool)], kmeans_k=4)
    assert centers.shape == (0, 2)


def test_build_memory_uses_only_foreground_patches():
    # Foreground (top-left patch) is +x; rest is +y. One center -> must be +x, not +y.
    grid = _grid([[1, 0], [0, 1], [0, 1], [0, 1]], (2, 2))
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True                                    # covers only the top-left patch
    centers = ntt.build_memory(grid, [mask], kmeans_k=1)
    assert centers.shape == (1, 2)
    assert centers[0, 0].item() > 0.99                   # aligned with +x foreground


# ---------------------------------------------------------------------------
# similarity_heatmap
# ---------------------------------------------------------------------------
def test_similarity_heatmap_matches_aligned_patch():
    grid = _grid([[1, 0], [0, 1], [-1, 0], [0, -1]], (2, 2))
    centers = torch.tensor([[1.0, 0.0]])                 # look for +x
    heat = ntt.similarity_heatmap(grid, centers)
    assert heat.shape == (2, 2)
    assert heat[0, 0].item() == pytest.approx(1.0, abs=1e-5)     # +x patch
    assert heat[1, 0].item() == pytest.approx(-1.0, abs=1e-5)    # -x patch


def test_similarity_heatmap_empty_memory_is_minus_one():
    grid = _grid([[1, 0], [0, 1]], (1, 2))
    heat = ntt.similarity_heatmap(grid, torch.zeros((0, 2)))
    assert torch.all(heat == -1.0)


# ---------------------------------------------------------------------------
# grid_topk_points  +  grid_points_to_pixels
# ---------------------------------------------------------------------------
def test_grid_topk_points_orders_and_thresholds():
    heat = torch.tensor([[0.1, 0.9], [0.6, 0.3]])
    rc = ntt.grid_topk_points(heat, num_points=3, thr=0.5)
    # Above-thr cells by descending sim: 0.9 at (0,1), 0.6 at (1,0). 0.3/0.1 pruned.
    assert rc.tolist() == [[0, 1], [1, 0]]


def test_grid_topk_points_empty_when_all_below_thr():
    heat = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    rc = ntt.grid_topk_points(heat, num_points=4, thr=0.9)
    assert rc.shape == (0, 2)


def test_grid_points_to_pixels_cell_centers():
    rc = np.array([[0, 0], [1, 2]])                      # (row, col)
    px = ntt.grid_points_to_pixels(rc, grid_hw=(2, 3), img_hw=(100, 300))
    # (0,0) -> x=(0.5/3)*300=50, y=(0.5/2)*100=25 ; (1,2) -> x=(2.5/3)*300=250, y=(1.5/2)*100=75
    assert px.tolist() == [[50.0, 25.0], [250.0, 75.0]]


def test_grid_points_to_pixels_empty():
    px = ntt.grid_points_to_pixels(np.zeros((0, 2), np.int64), (2, 2), (10, 10))
    assert px.shape == (0, 2)


# ---------------------------------------------------------------------------
# score_masks
# ---------------------------------------------------------------------------
def test_score_masks_high_for_aligned_low_for_opposed():
    grid = _grid([[1, 0], [1, 0], [-1, 0], [-1, 0]], (2, 2))
    centers = torch.tensor([[1.0, 0.0]])
    aligned = torch.tensor([[True, True], [False, False]])   # the +x patches
    opposed = torch.tensor([[False, False], [True, True]])   # the -x patches
    scores = ntt.score_masks(grid, torch.stack([aligned, opposed]), centers)
    assert scores[0].item() == pytest.approx(1.0, abs=1e-5)  # (cos=+1 -> 1)
    assert scores[1].item() == pytest.approx(0.0, abs=1e-5)  # (cos=-1 -> 0)


def test_score_masks_empty_foreground_scores_zero():
    grid = _grid([[1, 0], [0, 1]], (1, 2))
    empty = torch.zeros((1, 1, 2), dtype=torch.bool)
    scores = ntt.score_masks(grid, empty, torch.tensor([[1.0, 0.0]]))
    assert scores.tolist() == [0.0]


# ---------------------------------------------------------------------------
# suppress_masks
# ---------------------------------------------------------------------------
def test_suppress_masks_drops_duplicate_keeps_higher_score():
    a = np.zeros((6, 6), dtype=bool); a[0:3, 0:3] = True
    dup = a.copy()                                       # identical -> IoU 1
    far = np.zeros((6, 6), dtype=bool); far[4:6, 4:6] = True
    masks = np.stack([a, dup, far])
    scores = np.array([0.9, 0.7, 0.8])
    kept = ntt.suppress_masks(masks, scores, iou_thr=0.8, containment_thr=0.9)
    # Duplicate of the top mask is dropped; the disjoint mask survives. Order is by score desc.
    assert kept.tolist() == [0, 2]


def test_suppress_masks_drops_nested_fragment():
    big = np.zeros((6, 6), dtype=bool); big[0:6, 0:6] = True
    small = np.zeros((6, 6), dtype=bool); small[1:3, 1:3] = True   # fully inside big, low IoU
    masks = np.stack([big, small])
    scores = np.array([0.9, 0.6])
    kept = ntt.suppress_masks(masks, scores, iou_thr=0.9, containment_thr=0.9)
    assert kept.tolist() == [0]                          # small is contained -> dropped


def test_suppress_masks_empty():
    kept = ntt.suppress_masks(np.zeros((0, 5, 5), bool), np.zeros((0,)),
                              iou_thr=0.5, containment_thr=0.5)
    assert kept.shape == (0,)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
def test_registered():
    from experiments.methods.base import _METHODS

    assert "no_time_to_train" in _METHODS
