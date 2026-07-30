"""GT-aware over/under-zoom diagnostics (roadmap A1.1)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.zoom_diagnostics import ZoomDiagnostics, zoom_diagnostics


def _mask(hw, box):
    y0, y1, x0, x1 = box
    m = np.zeros(hw, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _event(box, decision, *, fg=None, children=()):
    return dict(box=box, decision=decision, fg=fg, children=list(children))


HW = (200, 200)


def test_whole_instance_emitted_is_not_overzoom():
    gt = [_mask(HW, (50, 100, 50, 100))]
    d = zoom_diagnostics([_event((45, 105, 45, 105), "leaf")], gt, [], HW)
    assert d.n_terminal == 1
    assert d.n_overzoom == 0
    assert d.mean_retained == pytest.approx(1.0)


def test_clipped_instance_is_overzoom():
    gt = [_mask(HW, (50, 100, 50, 100))]                 # 50x50 instance
    d = zoom_diagnostics([_event((50, 100, 50, 75), "leaf")], gt, [], HW)   # right half cut off
    assert d.n_overzoom == 1
    assert d.overzoom_rate == pytest.approx(1.0)
    assert d.mean_retained == pytest.approx(0.5, abs=0.02)


def test_two_instances_in_one_leaf_is_underzoom():
    gt = [_mask(HW, (10, 40, 10, 40)), _mask(HW, (10, 40, 60, 90))]
    d = zoom_diagnostics([_event((5, 45, 5, 95), "leaf")], gt, [], HW)
    assert d.n_underzoom == 1
    assert d.n_overzoom == 0                              # its dominant instance is whole


def test_slow_zoom_counts_only_descend_steps():
    gt = [_mask(HW, (50, 100, 50, 100))]
    barely = _event((0, 200, 0, 200), "zoom", children=[(0, 195, 0, 195)])   # ~0.95 area ratio
    real = _event((0, 200, 0, 200), "zoom", children=[(0, 100, 0, 100)])     # 0.25
    d = zoom_diagnostics([barely, real], gt, [], HW)
    assert d.n_descend == 2
    assert d.n_slow_zoom == 1
    assert d.n_terminal == 0


def test_foreground_precision_recall_binned_by_crop_size():
    gt = [_mask(HW, (0, 100, 0, 100))]                    # top-left quadrant is the concept
    fg = np.zeros((4, 4), dtype=bool)
    fg[:2, :2] = True                                     # exactly the same quadrant on the grid
    # A 200 px crop against a 200 px backbone input → the "1-2x" bin.
    d = zoom_diagnostics([_event((0, 200, 0, 200), "leaf", fg=fg)], gt, [], HW, image_size=200)
    acc = d.bins["1-2x"]
    assert acc.precision == pytest.approx(1.0)
    assert acc.recall == pytest.approx(1.0)


def test_foreground_recall_drops_when_fg_is_too_tight():
    gt = [_mask(HW, (0, 100, 0, 100))]
    fg = np.zeros((4, 4), dtype=bool)
    fg[0, 0] = True                                       # a quarter of the true region
    d = zoom_diagnostics([_event((0, 200, 0, 200), "leaf", fg=fg)], gt, [], HW, image_size=200)
    acc = d.bins["1-2x"]
    assert acc.precision == pytest.approx(1.0)            # what it kept is right
    assert acc.recall == pytest.approx(0.25)              # ...but it kept too little (over-zoom)


def test_exemplar_masks_count_as_the_concept():
    """A crop landing on a *prompt* instance is doing its job — not a false positive."""
    fg = np.ones((2, 2), dtype=bool)
    ex = [_mask(HW, (0, 200, 0, 200))]
    empty = zoom_diagnostics([_event((0, 200, 0, 200), "leaf", fg=fg)], [], ex, HW, image_size=200)
    assert empty.bins["1-2x"].precision == pytest.approx(1.0)


def test_emit_side_histogram_and_metrics_are_flat_floats():
    gt = [_mask(HW, (50, 100, 50, 100))]
    d = zoom_diagnostics([_event((0, 80, 0, 80), "leaf"), _event((0, 200, 0, 200), "leaf-cap")],
                         gt, [], HW, image_size=100)
    assert sorted(d.emit_sides) == [80.0, 200.0]
    m = d.to_metrics("intra")
    assert m["intra_zoom_emit_frac_below_input"] == pytest.approx(0.5)
    assert all(isinstance(v, float) for v in m.values())
    assert "intra_zoom_overzoom_rate" in m


def test_merge_pools_micro_averages():
    gt = [_mask(HW, (50, 100, 50, 100))]
    a = zoom_diagnostics([_event((50, 100, 50, 75), "leaf")], gt, [], HW)      # over-zoomed
    b = zoom_diagnostics([_event((45, 105, 45, 105), "leaf")], gt, [], HW)     # clean
    pooled = a.merge(b)
    assert pooled.n_terminal == 2
    assert pooled.overzoom_rate == pytest.approx(0.5)


def test_no_masks_or_no_events_is_empty_not_an_error():
    assert zoom_diagnostics([], [], [], HW).n_events == 0
    assert isinstance(zoom_diagnostics([_event((0, 10, 0, 10), "leaf")], [], [], HW),
                      ZoomDiagnostics)


def test_ignores_non_cascade_events_without_a_box():
    gt = [_mask(HW, (50, 100, 50, 100))]
    d = zoom_diagnostics([{"decision": "leaf"}], gt, [], HW)
    assert d.n_events == 0
