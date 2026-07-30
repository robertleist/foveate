"""The A1.2 over-zoom guards: separability, hysteresis, and the crop dilation.

Each is off by default, so the first thing every group asserts is that the default path is
unchanged — the refactor's byte-identical guarantee must survive new knobs.
"""

from __future__ import annotations

import numpy as np
import pytest

from foveate import thresholding
from foveate.cascade import _child_box
from foveate.config import Config
from foveate.otsu import OtsuExtractor


# --------------------------------------------------------------------------- separability
def test_separability_is_high_for_a_clean_two_mode_map():
    values = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])
    tau = thresholding.otsu(values)
    assert thresholding.separability(values, tau) > 0.9


def test_separability_is_low_for_a_unimodal_map():
    """The over-zoom regime: the concept fills the crop, so there is no background mode left."""
    rng = np.random.default_rng(0)
    values = rng.normal(0.8, 0.02, size=400)
    tau = thresholding.otsu(values)
    assert thresholding.separability(values, tau) < 0.75      # ~0.65, the unimodal fixed point


def test_separability_separates_structureless_from_real_foreground():
    """The calibration the guard's default rests on — see :func:`thresholding.separability`."""
    rng = np.random.default_rng(0)
    def eta(v):
        return thresholding.separability(v, thresholding.otsu(v))

    structureless = [eta(rng.normal(0.8, 0.02, 4000)), eta(rng.uniform(0, 1, 4000))]
    # A real map, even with the foreground down to 10 % of the patches.
    real = [eta(np.concatenate([rng.normal(0.2, 0.08, 3000), rng.normal(0.8, 0.08, 1000)])),
            eta(np.concatenate([rng.normal(0.2, 0.05, 3600), rng.normal(0.75, 0.05, 400)]))]
    assert max(structureless) < 0.8 < min(real)


def test_separability_of_a_constant_map_is_zero():
    values = np.full(64, 0.5)
    assert thresholding.separability(values, thresholding.otsu(values)) == 0.0


# --------------------------------------------------------------------------- hysteresis
def test_hysteresis_grows_a_connected_rim_but_not_a_free_floating_blob():
    grid = np.zeros((7, 7))
    grid[3, 3] = 0.9                      # seed
    grid[3, 4] = 0.6                      # dim rim, attached  -> joins
    grid[0, 0] = 0.6                      # dim, detached      -> stays out
    out = thresholding.hysteresis(grid, hi=0.8, lo=0.5)
    assert out[3, 3] and out[3, 4]
    assert not out[0, 0]


def test_hysteresis_without_a_seed_is_empty():
    grid = np.full((4, 4), 0.3)
    assert not thresholding.hysteresis(grid, hi=0.8, lo=0.1).any()


def test_hysteresis_collapses_to_the_single_cut_when_lo_is_not_lower():
    grid = np.linspace(0, 1, 16).reshape(4, 4)
    assert np.array_equal(thresholding.hysteresis(grid, hi=0.5, lo=0.5), grid >= 0.5)


# --------------------------------------------------------------------------- crop dilation
def _comp(shape, box):
    r0, r1, c0, c1 = box
    g = np.zeros(shape, dtype=bool)
    g[r0:r1, c0:c1] = True
    return g


def test_crop_dilate_defaults_to_no_change():
    comp = _comp((10, 10), (4, 6, 4, 6))
    assert _child_box(comp, (0, 100, 0, 100), 0.0) == _child_box(comp, (0, 100, 0, 100), 0.0, 0)


def test_crop_dilate_widens_the_box_by_whole_patches():
    comp = _comp((10, 10), (4, 6, 4, 6))                   # patches 4-5 → 40..60 px of a 100 px crop
    tight = _child_box(comp, (0, 100, 0, 100), 0.0, 0)
    wide = _child_box(comp, (0, 100, 0, 100), 0.0, 1)
    assert tight == (40, 60, 40, 60)
    assert wide == (30, 70, 30, 70)                         # one patch = 10 px on each side


def test_crop_dilate_is_clipped_to_the_parent_box():
    comp = _comp((10, 10), (0, 2, 0, 2))                    # already at the grid edge
    box = (0, 100, 0, 100)
    assert _child_box(comp, box, 0.0, 3) == (0, 50, 0, 50)  # clipped at 0, grown at the far side


# --------------------------------------------------------------------------- Otsu extractor wiring
class _StubOtsu(OtsuExtractor):
    """Drive ``_binarize``/``_grow`` directly — the reference/backbone half is not under test."""

    def __init__(self, cfg):
        super().__init__(cfg)


def _binarize(cfg, sim_grid):
    ext = _StubOtsu(cfg)
    score = np.clip((sim_grid - sim_grid.min()) / max(float(np.ptp(sim_grid)), 1e-12), 0, 1)
    return ext._binarize(sim_grid, score, thresholding.otsu(sim_grid))


def test_default_config_is_the_plain_otsu_cut():
    sim = np.array([[0.1, 0.1], [0.9, 0.9]])
    fg, mode, _ = _binarize(Config(), sim)
    assert mode == "otsu"
    assert np.array_equal(fg, sim >= thresholding.otsu(sim))


def test_unimodal_map_accepts_the_whole_crop_when_guarded():
    rng = np.random.default_rng(1)
    sim = rng.normal(0.8, 0.02, size=(8, 8))
    cfg = Config(otsu_min_separability=0.8, otsu_unimodal_fallback="accept_all")
    fg, mode, eta = _binarize(cfg, sim)
    assert mode == "unimodal:accept_all"
    assert fg.all()                        # ...instead of slicing the object in half
    assert eta < 0.8


def test_guard_leaves_a_genuinely_bimodal_map_alone():
    sim = np.concatenate([np.full(32, 0.1), np.full(32, 0.9)]).reshape(8, 8)
    cfg = Config(otsu_min_separability=0.8, otsu_unimodal_fallback="accept_all")
    fg, mode, _ = _binarize(cfg, sim)
    assert mode == "otsu"
    assert fg.sum() == 32


def test_unimodal_percentile_fallback_keeps_a_fraction():
    rng = np.random.default_rng(2)
    sim = rng.normal(0.8, 0.02, size=(10, 10))
    cfg = Config(otsu_min_separability=0.8, otsu_unimodal_fallback="percentile",
                 gate_percentile=80.0)
    fg, mode, _ = _binarize(cfg, sim)
    assert mode == "unimodal:percentile"
    assert 15 <= fg.sum() <= 25            # ~20 % of 100 patches


def test_unimodal_otsu_fallback_is_the_control_arm():
    """Detects the regime, acts exactly as before — so the ablation can separate the two."""
    rng = np.random.default_rng(3)
    sim = rng.normal(0.8, 0.02, size=(8, 8))
    cfg = Config(otsu_min_separability=0.8, otsu_unimodal_fallback="otsu")
    fg, mode, _ = _binarize(cfg, sim)
    assert mode == "unimodal:otsu"
    assert np.array_equal(fg, sim >= thresholding.otsu(sim))


def test_hysteresis_recovers_the_rim_the_single_cut_shaved_off():
    """The over-zoom shape: Otsu keeps only the object's bright core, shaving its dim rim.

    The rim scores no better than a patch of mildly similar background elsewhere in the crop, so no
    *single* cut can keep one and drop the other — only connectivity separates them.
    """
    sim = np.full((10, 10), 0.15)          # background
    sim[3:8, 3:8] = 0.45                   # the object's dim rim
    sim[4:7, 4:7] = 0.95                   # ...and its bright core
    sim[0:2, 0:2] = 0.45                   # a detached patch of look-alike background

    plain, _, _ = _binarize(Config(), sim)
    grown, _, _ = _binarize(Config(otsu_hysteresis_lo=0.35), sim)

    assert plain.sum() == 9                # the single cut keeps the core only
    assert grown[3:8, 3:8].all()           # rim recovered (connected to the core)
    assert not grown[0:2, 0:2].any()       # look-alike background still rejected (detached)


def test_hysteresis_off_by_default():
    sim = np.full((5, 5), 0.05)
    sim[2, 2] = 0.9
    default, _, _ = _binarize(Config(), sim)
    assert default.sum() == 1
