"""EXTRACT slot (:mod:`foveate.extract`) — components, splitters and the registry."""

import numpy as np
import pytest
import torch

from foveate import Config, cascade
from foveate.extract import (
    _CONN4,
    _CONN8,
    ConnectedComponentsExtractor,
    KMeansExtractor,
    _extract_structure,
    _split_kmeans,
    build_instance_extractor,
    component_label_map,
)


def test_extract_structure_selects_connectivity():
    """The Extract stage maps connectivity 4/8 to the right structuring element."""
    assert _extract_structure(4) is _CONN4
    assert _extract_structure(8) is _CONN8
    assert _extract_structure(99) is _CONN8              # anything else → 8-connectivity default


def test_components_separates_two_blobs_and_honours_connectivity():
    """Two diagonally-touching blobs: one component under 8-connectivity, two under 4."""
    fg = np.zeros((6, 6), dtype=bool)
    fg[1:3, 1:3] = True
    fg[3:5, 3:5] = True                                  # touches the first only diagonally
    assert len(build_instance_extractor(Config(extract_connectivity=8)).components(fg)) == 1
    assert len(build_instance_extractor(Config(extract_connectivity=4)).components(fg)) == 2


def test_components_of_empty_foreground_is_empty():
    ie = build_instance_extractor(Config())
    assert ie.components(np.zeros((4, 4), dtype=bool)) == []


def test_component_label_map_round_trips():
    comps = [np.zeros((3, 3), dtype=bool), np.zeros((3, 3), dtype=bool)]
    comps[0][0, 0] = True
    comps[1][2, 2] = True
    lab = component_label_map(comps, (3, 3))
    assert lab[0, 0] == 1 and lab[2, 2] == 2 and lab.sum() == 3   # 0 elsewhere


def test_cc_extractor_never_splits():
    """``cc`` (≡ the legacy ``split_mode: none``) advertises no split and returns the component whole."""
    ie = build_instance_extractor(Config(instance_extractor="cc"))
    assert isinstance(ie, ConnectedComponentsExtractor) and not ie.can_split
    comp = np.ones((4, 4), dtype=bool)
    subs = ie.split(torch.zeros(4, 4, 3), comp)
    assert len(subs) == 1 and subs[0].all()


def test_split_kmeans_partitions_two_feature_clusters():
    """k=2 on features splits a component into two disjoint sub-masks that cover it exactly."""
    hp = wp = 6
    feat = torch.zeros(hp, wp, 4)
    feat[:, :3, 0] = 1.0                 # left half → one feature axis
    feat[:, 3:, 1] = 1.0                 # right half → another
    comp = np.ones((hp, wp), dtype=bool)
    subs = _split_kmeans(feat, comp)
    assert len(subs) == 2
    assert not (subs[0] & subs[1]).any()                 # disjoint
    assert (subs[0] | subs[1]).sum() == comp.sum()       # partition the whole component
    assert {int(subs[0].sum()), int(subs[1].sum())} == {hp * 3}  # the two halves


def test_split_kmeans_single_patch_unsplittable():
    feat = torch.zeros(4, 4, 3); feat[0, 0, 0] = 1.0
    comp = np.zeros((4, 4), dtype=bool); comp[0, 0] = True
    subs = _split_kmeans(feat, comp)
    assert len(subs) == 1 and bool(subs[0][0, 0])


def test_kmeans_extractor_delegates_to_split_kmeans():
    """The registry's default is the k=2 splitter, and it splits (``can_split`` is not a lie)."""
    ie = build_instance_extractor(Config())
    assert isinstance(ie, KMeansExtractor) and ie.can_split
    feat = torch.zeros(4, 4, 3)
    feat[:, :2, 0] = 1.0
    feat[:, 2:, 1] = 1.0
    assert len(ie.split(feat, np.ones((4, 4), dtype=bool))) == 2


@pytest.mark.parametrize("name", ["cc", "none", "kmeans", "agglomerative", "watershed"])
def test_registry_builds_every_name(name):
    ie = build_instance_extractor(Config(instance_extractor=name))
    assert isinstance(ie, ConnectedComponentsExtractor)               # all share the CC proposal
    assert ie.can_split == (name not in ("cc", "none"))


def test_registry_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown instance_extractor"):
        build_instance_extractor(Config(instance_extractor="does-not-exist"))


def test_legacy_split_mode_key_maps_to_the_extract_slot():
    """Every existing YAML says ``split_mode``; the alias must keep working, ``none`` included."""
    assert Config.from_dict({"split_mode": "watershed"}).instance_extractor == "watershed"
    assert Config.from_dict({"split_mode": "none"}).instance_extractor == "none"
    assert Config().instance_extractor == "kmeans"                    # unchanged default


@pytest.mark.parametrize("name", ["kmeans", "agglomerative", "cc"])
def test_cascade_runs_with_each_extractor(backbone, two_squares, name):
    """Each Extract slot value drives the whole cascade end-to-end."""
    img, ex = two_squares
    cfg = Config(instance_extractor=name, min_crop=24, cascade_min_instance_area=4)
    instances, stats = cascade(backbone, img, ex, config=cfg)
    assert stats.n_embeds > 0
    assert all(inst.mask.shape == img.shape[:2] for inst in instances)


def test_cc_extractor_never_attempts_a_split(backbone, two_squares):
    """``can_split=False`` must short-circuit BOTH split paths — the convergence split and the
    marginal-peak retry — while the cascade still discovers instances. The retry is what fires on
    this fixture (``zoom_split_retry_eps=1.0``), so it is the sharper of the two to gate on."""
    img, ex = two_squares
    base = dict(min_crop=24, cascade_min_instance_area=4, zoom_split_retry_eps=1.0)
    ev_km, ev_cc = [], []
    cascade(backbone, img, ex, config=Config(instance_extractor="kmeans", **base),
            observer=ev_km.append)
    inst_cc, _ = cascade(backbone, img, ex, config=Config(instance_extractor="cc", **base),
                         observer=ev_cc.append)
    assert any(e["decision"] == "clump-split" for e in ev_km)      # the splitter does fire...
    assert not any(e["decision"] == "clump-split" for e in ev_cc)  # ...and ``cc`` never does
    assert inst_cc                                                 # still discovers instances
