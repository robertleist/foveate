"""EXTRACT slot (:mod:`foveate.extract`) — the contract, the composite, and the registry.

The grouping half lives in tests/test_grouping.py, the oracle in tests/test_extract_oracle.py.
"""

import numpy as np
import pytest
import torch

from foveate import Config, cascade
from foveate.extract import (
    CompositeExtractor,
    ExtractResult,
    OracleExtractor,
    build_extractor,
    component_label_map,
    resolve_extractor_name,
)


def test_component_label_map_round_trips():
    comps = [np.zeros((3, 3), dtype=bool), np.zeros((3, 3), dtype=bool)]
    comps[0][0, 0] = True
    comps[1][2, 2] = True
    lab = component_label_map(comps, (3, 3))
    assert lab[0, 0] == 1 and lab[2, 2] == 2 and lab.sum() == 3   # 0 elsewhere


# --------------------------------------------------------------------------- the contract
def test_extract_returns_instances_and_the_region_they_decompose(backbone, two_squares):
    """One call, one answer: instance grids plus the region the extractor calls the concept."""
    img, ex = two_squares
    cfg = Config(foreground_extractor="otsu", debias=False)
    extractor = build_extractor(cfg)
    extractor.set_reference(backbone, img, ex, None, cfg)

    from foveate import features as featlib
    (feat, cls), = featlib.embed_batch(backbone, [img], standardize=cfg.standardize)
    res = extractor.extract(feat, cls=cls, box=(0, 128, 0, 128))

    assert isinstance(res, ExtractResult)
    assert res.foreground.shape == feat.shape[:2] and res.foreground.dtype == bool
    assert res.score_map.shape == res.foreground.shape
    assert all(g.shape == res.foreground.shape and g.dtype == bool for g in res.instances)
    # Every instance lies inside the region — the composite decomposes a foreground, never repairs it.
    for g in res.instances:
        assert not (g & ~res.foreground).any()


def test_internals_are_only_paid_for_when_someone_is_watching(backbone, two_squares):
    img, ex = two_squares
    cfg = Config(debias=False)
    extractor = build_extractor(cfg)
    extractor.set_reference(backbone, img, ex, None, cfg)
    from foveate import features as featlib
    (feat, cls), = featlib.embed_batch(backbone, [img], standardize=cfg.standardize)
    assert extractor.extract(feat, cls=cls, box=(0, 128, 0, 128)).internals == {}
    assert extractor.extract(feat, cls=cls, box=(0, 128, 0, 128),
                             return_internals=True).internals


# --------------------------------------------------------------------------- composite
def test_composite_exposes_both_halves_for_the_ablation():
    ex = build_extractor(Config(foreground_extractor="otsu", instance_extractor="cc"))
    assert isinstance(ex, CompositeExtractor)
    from foveate.grouping import ConnectedComponents
    from foveate.otsu import OtsuExtractor
    assert isinstance(ex.where, OtsuExtractor)
    assert isinstance(ex.grouping, ConnectedComponents)


def test_composite_forwards_the_exemplar_bank(backbone, two_squares):
    """``g`` is built off the extractor's bank, so the composite has to hand its Where half's up."""
    img, ex = two_squares
    cfg = Config(debias=False)
    extractor = build_extractor(cfg)
    extractor.set_reference(backbone, img, ex, None, cfg)
    assert extractor.exemplar_cls is extractor.where.exemplar_cls
    assert extractor.exemplar_cls.shape[0] == 1


# --------------------------------------------------------------------------- registry / resolution
def test_registry_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown extractor"):
        build_extractor(Config(extractor="does-not-exist"))


def test_explicit_extractor_key_wins():
    assert resolve_extractor_name(Config(extractor="oracle")) == "oracle"
    assert isinstance(build_extractor(Config(extractor="oracle")), OracleExtractor)
    # ...even against a legacy pair that would otherwise resolve the other way
    assert resolve_extractor_name(
        Config(extractor="composite", instance_extractor="oracle")) == "composite"


@pytest.mark.parametrize("legacy, expected", [
    # An oracular DECOMPOSITION collapses into the one monolithic oracle...
    (dict(instance_extractor="oracle"), "oracle"),
    (dict(foreground_extractor="oracle", instance_extractor="oracle"), "oracle"),
    (dict(foreground_extractor="oracle_cc"), "oracle"),          # itself a Where+group method
    # ...while a perfect Where paired with a real grouping stays the Where-headroom ablation.
    (dict(foreground_extractor="oracle", instance_extractor="kmeans"), "composite"),
    (dict(foreground_extractor="insid3", instance_extractor="kmeans"), "composite"),
    (dict(), "composite"),
])
def test_legacy_key_pair_resolves_without_changing_meaning(legacy, expected):
    assert resolve_extractor_name(Config(**legacy)) == expected


@pytest.mark.parametrize("name", ["kmeans", "agglomerative", "cc"])
def test_cascade_runs_with_each_grouping(backbone, two_squares, name):
    """Each grouping value drives the whole cascade end-to-end."""
    img, ex = two_squares
    cfg = Config(instance_extractor=name, min_crop=24, cascade_min_instance_area=4)
    instances, stats = cascade(backbone, img, ex, config=cfg)
    assert stats.n_embeds > 0
    assert all(inst.mask.shape == img.shape[:2] for inst in instances)


def test_cc_grouping_never_produces_more_instances_than_components(backbone, two_squares):
    """``cc`` cannot invent an internal boundary, so a crop never yields more instances than the
    foreground has connected components — while the cascade still discovers both squares."""
    img, ex = two_squares
    events = []
    inst, _ = cascade(backbone, img, ex, observer=events.append,
                      config=Config(instance_extractor="cc", min_crop=24,
                                    cascade_min_instance_area=4))
    assert inst
    from scipy.ndimage import label
    for e in events:
        fg = e.get("fg")
        if fg is None or not np.size(fg):
            continue
        assert e["n_components"] <= label(np.asarray(fg, bool))[1]
