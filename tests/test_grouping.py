"""The grouping half of the composite extractor (:mod:`foveate.grouping`)."""

import numpy as np
import pytest
import torch

from foveate import Config
from foveate.grouping import (
    _CONN4,
    _CONN8,
    ConnectedComponents,
    KMeansGrouping,
    _extract_structure,
    build_grouping,
    split_kmeans,
)


def test_extract_structure_selects_connectivity():
    assert _extract_structure(4) is _CONN4
    assert _extract_structure(8) is _CONN8
    assert _extract_structure(99) is _CONN8              # anything else → 8-connectivity default


def test_components_separates_two_blobs_and_honours_connectivity():
    """Two diagonally-touching blobs: one instance under 8-connectivity, two under 4."""
    fg = np.zeros((6, 6), dtype=bool)
    fg[1:3, 1:3] = True
    fg[3:5, 3:5] = True                                  # touches the first only diagonally
    feat = torch.zeros(6, 6, 3)
    assert len(build_grouping(Config(instance_extractor="cc",
                                     extract_connectivity=8)).group(feat, fg)) == 1
    assert len(build_grouping(Config(instance_extractor="cc",
                                     extract_connectivity=4)).group(feat, fg)) == 2


def test_empty_foreground_yields_no_instances():
    g = build_grouping(Config(instance_extractor="cc"))
    assert g.group(torch.zeros(4, 4, 3), np.zeros((4, 4), dtype=bool)) == []


def test_cc_never_cuts_a_component_even_when_it_fills_the_crop():
    """``cc`` (≡ the legacy ``split_mode: none``) is the no-cut control arm."""
    g = build_grouping(Config(instance_extractor="cc"))
    assert isinstance(g, ConnectedComponents)
    out = g.group(torch.zeros(4, 4, 3), np.ones((4, 4), dtype=bool))
    assert len(out) == 1 and out[0].all()


# --------------------------------------------------------------------------- the k=2 cut
def test_split_kmeans_partitions_two_feature_clusters():
    """k=2 on features splits a component into two disjoint sub-masks that cover it exactly."""
    hp = wp = 6
    feat = torch.zeros(hp, wp, 4)
    feat[:, :3, 0] = 1.0                 # left half → one feature axis
    feat[:, 3:, 1] = 1.0                 # right half → another
    comp = np.ones((hp, wp), dtype=bool)
    subs = split_kmeans(feat, comp)
    assert len(subs) == 2
    assert not (subs[0] & subs[1]).any()                 # disjoint
    assert (subs[0] | subs[1]).sum() == comp.sum()       # partition the whole component
    assert {int(subs[0].sum()), int(subs[1].sum())} == {hp * 3}  # the two halves


def test_split_kmeans_single_patch_is_one_instance():
    feat = torch.zeros(4, 4, 3); feat[0, 0, 0] = 1.0
    comp = np.zeros((4, 4), dtype=bool); comp[0, 0] = True
    subs = split_kmeans(feat, comp)
    assert len(subs) == 1 and bool(subs[0][0, 0])


def test_kmeans_cuts_a_component_that_fills_the_crop():
    g = build_grouping(Config())                         # kmeans is the default
    assert isinstance(g, KMeansGrouping)
    feat = torch.zeros(6, 6, 4)
    feat[:, :3, 0] = 1.0
    feat[:, 3:, 1] = 1.0
    assert len(g.group(feat, np.ones((6, 6), dtype=bool))) == 2


def test_kmeans_leaves_a_component_the_cascade_can_still_zoom_into_whole():
    """The cut is conditional: below the fill threshold the zoom asks again at a finer scale, and
    cutting now would replace the tightened whole object with two halves of it."""
    g = build_grouping(Config())
    feat = torch.zeros(12, 12, 4)
    feat[:, :6, 0] = 1.0
    feat[:, 6:, 1] = 1.0
    comp = np.zeros((12, 12), dtype=bool)
    comp[3:7, 3:7] = True                                # a small blob in a large crop
    out = g.group(feat, comp)
    assert len(out) == 1 and np.array_equal(out[0], comp)


def test_fill_threshold_is_the_cascades_own_convergence_test():
    """The gate is ``features.child_box`` + ``shrink_stop`` — the same computation the cascade runs
    one step later, so the two cannot drift apart."""
    feat = torch.zeros(8, 8, 4)
    feat[:, :4, 0] = 1.0
    feat[:, 4:, 1] = 1.0
    comp = np.zeros((8, 8), dtype=bool)
    comp[:, :7] = True
    box = (0, 80, 0, 80)                                 # child box 80 x 76 of 80 x 80 → 0.95
    from foveate import features as featlib
    child = featlib.child_box(comp, box, Config().pad_frac, Config().crop_dilate)
    assert (child[1] - child[0]) * (child[3] - child[2]) / 6400 == pytest.approx(0.95)
    assert len(build_grouping(Config(shrink_stop=0.9)).group(feat, comp, box=box)) == 2
    assert len(build_grouping(Config(shrink_stop=0.99)).group(feat, comp, box=box)) == 1


# --------------------------------------------------------------------------- registry
@pytest.mark.parametrize("name", ["cc", "none", "kmeans", "agglomerative", "watershed"])
def test_registry_builds_every_name(name):
    g = build_grouping(Config(instance_extractor=name))
    assert isinstance(g, ConnectedComponents)                     # all share the CC proposal
    assert isinstance(g, KMeansGrouping) == (name == "kmeans")


def test_registry_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown grouping"):
        build_grouping(Config(instance_extractor="does-not-exist"))


def test_legacy_and_short_keys_both_reach_the_grouping():
    """Every existing YAML says ``split_mode``; new ones may say ``grouping``. Both must resolve."""
    assert Config.from_dict({"split_mode": "watershed"}).instance_extractor == "watershed"
    assert Config.from_dict({"split_mode": "none"}).instance_extractor == "none"
    assert Config.from_dict({"grouping": "cc"}).instance_extractor == "cc"
    assert Config.from_dict({"where": "otsu"}).foreground_extractor == "otsu"
    assert Config().instance_extractor == "kmeans"                # unchanged default
