"""Tests for the semantic + connected-components baseline (experiments/methods/semantic_cc.py).

The deterministic core — connected-components labelling + scoring — is unit-tested directly on
hand-built arrays via :func:`foreground_to_instances`. One end-to-end test drives ``predict`` with
the weightless MockBackbone (colour features, no DINOv3 weights) so the real single-pass INSID3
WHERE step is exercised too.
"""

import numpy as np
import pytest

from experiments.datasets import EvalItem
from experiments.methods import build_method
from experiments.methods.semantic_cc import SemanticCCMethod, foreground_to_instances


# ---------------------------------------------------------------------------
# foreground_to_instances — the deterministic connected-components core
# ---------------------------------------------------------------------------
def test_two_separated_blobs_give_two_components():
    fg = np.zeros((10, 10), dtype=bool)
    fg[1:4, 1:4] = True      # blob A
    fg[6:9, 6:9] = True      # blob B, well separated
    masks, scores = foreground_to_instances(
        fg, None, min_area=1, connectivity=8, score_mode="constant",
    )
    assert masks.shape == (2, 10, 10)
    assert scores.shape == (2,)
    # Each component recovers exactly one blob (9 px), disjoint from the other.
    assert {int(m.sum()) for m in masks} == {9}
    assert not np.logical_and(masks[0], masks[1]).any()


def test_min_area_drops_tiny_blob():
    fg = np.zeros((10, 10), dtype=bool)
    fg[1:5, 1:5] = True      # 16 px — kept
    fg[8, 8] = True          # 1 px — dropped by min_area
    masks, _ = foreground_to_instances(
        fg, None, min_area=5, connectivity=8, score_mode="constant",
    )
    assert masks.shape[0] == 1
    assert int(masks[0].sum()) == 16


def test_empty_foreground_is_zero_instances():
    fg = np.zeros((7, 5), dtype=bool)
    masks, scores = foreground_to_instances(
        fg, None, min_area=1, connectivity=8, score_mode="mean_prob",
    )
    assert masks.shape == (0, 7, 5)
    assert scores.shape == (0,)


def test_connectivity_splits_diagonal_touch():
    # Two 1-px blobs touching only at a corner: 8-connectivity merges them, 4-connectivity keeps
    # them separate.
    fg = np.zeros((4, 4), dtype=bool)
    fg[1, 1] = True
    fg[2, 2] = True
    masks8, _ = foreground_to_instances(fg, None, min_area=1, connectivity=8, score_mode="constant")
    masks4, _ = foreground_to_instances(fg, None, min_area=1, connectivity=4, score_mode="constant")
    assert masks8.shape[0] == 1
    assert masks4.shape[0] == 2


def test_mean_prob_scoring():
    fg = np.zeros((4, 4), dtype=bool)
    fg[0:2, 0:2] = True
    score_map = np.zeros((4, 4), dtype=np.float64)
    score_map[0:2, 0:2] = [[0.2, 0.4], [0.6, 0.8]]   # mean = 0.5
    _, scores = foreground_to_instances(
        fg, score_map, min_area=1, connectivity=8, score_mode="mean_prob",
    )
    assert scores == pytest.approx([0.5])


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------
def _config():
    return {
        "backbone": {"type": "mock", "image_size": 224, "patch_size": 14},
        "foveate": {"standardize": False, "insid3_tau_fg": 0.6,
                    "insid3_aggt": 0.2},
        "method": {"type": "semantic_cc", "min_area": 4, "connectivity": 8,
                   "score_mode": "mean_prob"},
    }


def test_registered_and_dispatched():
    method = build_method(_config())
    assert isinstance(method, SemanticCCMethod)
    assert method.min_area == 4 and method.connectivity == 8


def test_param_blocks_includes_shared_foveate():
    method = build_method(_config())
    blocks = method.param_blocks()
    assert blocks["semantic_cc"]["min_area"] == 4
    assert "foveate" in blocks and blocks["foveate"]["foreground_extractor"] == "insid3"


def test_invalid_connectivity_raises():
    cfg = _config()
    cfg["method"]["connectivity"] = 6
    with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
        build_method(cfg)


# ---------------------------------------------------------------------------
# End-to-end predict() through the real single-pass INSID3 gate + MockBackbone
# ---------------------------------------------------------------------------
def _blob_image(size: int, colour, boxes) -> np.ndarray:
    """A black image with coloured square blobs at the given (y0, y1, x0, x1) boxes."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y0, y1, x0, x1 in boxes:
        img[y0:y1, x0:x1] = colour
    return img


def test_predict_end_to_end_intra():
    method = build_method(_config())
    size = 96
    # Two well-separated red blobs; prompt with one of them (intra: exemplar on the target image).
    boxes = [(10, 30, 10, 30), (60, 85, 60, 85)]
    img = _blob_image(size, (220, 30, 30), boxes)
    exemplar = np.zeros((size, size), dtype=bool)
    exemplar[10:30, 10:30] = True
    item = EvalItem(
        image_id="t", image=img, exemplar_masks=[exemplar],
        gt_masks=np.zeros((1, size, size), dtype=bool), class_id=0, exemplar_image=None,
    )
    pred = method.predict(item)
    assert pred.masks.dtype == bool
    assert pred.masks.shape[1:] == (size, size)
    assert pred.masks.shape[0] == pred.scores.shape[0]
    assert pred.n_embeds == 1
    # The two red blobs are distinct colour regions, so the INSID3 foreground + connected
    # components must find at least the two of them (individuation-free split).
    assert pred.masks.shape[0] >= 2


def test_predict_empty_foreground_shape():
    # A monochrome image with no colour matching the exemplar → INSID3 finds no distinct class
    # region → 0 instances with the correct (0, H, W) shape (no crash).
    method = build_method(_config())
    size = 64
    img = _blob_image(size, (200, 20, 20), [(5, 20, 5, 20)])
    # Exemplar mask over the black background (a different "colour" than any red blob elsewhere).
    exemplar = np.zeros((size, size), dtype=bool)
    exemplar[40:55, 40:55] = True
    item = EvalItem(
        image_id="t", image=img, exemplar_masks=[exemplar],
        gt_masks=np.zeros((1, size, size), dtype=bool), class_id=0, exemplar_image=None,
    )
    pred = method.predict(item)
    assert pred.masks.shape[1:] == (size, size)
    assert pred.masks.shape[0] == pred.scores.shape[0]


def test_predict_cross_image_inter():
    # Inter: exemplar lives on a SEPARATE support image, foreground predicted on the target.
    method = build_method(_config())
    size = 96
    target = _blob_image(size, (30, 200, 40), [(10, 35, 10, 35), (55, 85, 55, 85)])
    support = _blob_image(size, (30, 200, 40), [(20, 50, 20, 50)])
    exemplar = np.zeros((size, size), dtype=bool)
    exemplar[20:50, 20:50] = True                     # in SUPPORT-image coords
    item = EvalItem(
        image_id="t", image=target, exemplar_masks=[exemplar],
        gt_masks=np.zeros((1, size, size), dtype=bool), class_id=0, exemplar_image=support,
    )
    pred = method.predict(item)
    assert pred.masks.shape[1:] == (size, size)
    assert pred.masks.shape[0] == pred.scores.shape[0]
    assert pred.n_embeds == 1
