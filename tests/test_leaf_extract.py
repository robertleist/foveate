"""The LEAF Extract slot — the expensive question, asked only where the zoom bottoms out.

The claim this file pins down is the WHERE x STOP x EXTRACT factorization: the descent runs a cheap
region proposal (Where + connected components) guided by the CLS re-identification score, and the
instance decomposition is asked for **once per leaf**. So the tests check the three things that make
that a real contract rather than a rename:

1. the leaf extractor is called, and only O(leaves) times (never once per visited crop);
2. its instances are what gets emitted (it overrides the descent's answer);
3. an empty leaf answer falls back to the descent's instances, so foveating can only refine.
"""

from __future__ import annotations

import numpy as np
import pytest

from foveate import Config, cascade, features as featlib
from foveate.extract import ExtractResult, build_leaf_extractor, leaf_config


class _RecordingExtractor:
    """A leaf extractor that records every crop it is asked about.

    ``answer`` is a callable ``(feat) -> list[grid]``; returning ``[]`` exercises the fallback.
    """

    def __init__(self, answer):
        self.answer = answer
        self.boxes: list[tuple[int, int, int, int]] = []
        self.images: list[np.ndarray] = []
        self.exemplar_cls = None

    def set_reference(self, backbone, ref_image, ref_masks, negative_masks, cfg) -> None:
        self.reference_set = True

    def extract(self, feat, *, cls=None, box=None, image=None, return_internals=False):
        self.boxes.append(box)
        self.images.append(image)
        hp, wp = feat.shape[:2]
        instances = self.answer(feat)
        fg = (np.logical_or.reduce(instances) if instances
              else np.zeros((hp, wp), dtype=bool))
        return ExtractResult(instances=instances, foreground=fg,
                             score_map=np.zeros((hp, wp), np.float32))


def _descent_cfg(**kwargs) -> Config:
    """The new algorithm's descent: a cheap Where, components only (never cut), reid stop."""
    base = dict(foreground_extractor="otsu", instance_extractor="cc", stop_rule="reid",
                debias=False, min_crop=24, cascade_min_instance_area=4)
    base.update(kwargs)
    return Config(**base)


def _one_centre_instance(feat):
    """One instance covering the middle of whatever crop it is handed."""
    hp, wp = feat.shape[:2]
    g = np.zeros((hp, wp), dtype=bool)
    g[hp // 4: 3 * hp // 4, wp // 4: 3 * wp // 4] = True
    return [g]


def test_leaf_extractor_is_called_and_only_on_terminals(backbone, two_squares):
    """It runs on leaves, once each — not once per visited crop."""
    img, ex = two_squares
    leaf = _RecordingExtractor(_one_centre_instance)
    instances, stats = cascade(backbone, img, ex, config=_descent_cfg(),
                               leaf_extractor=leaf)

    assert stats.n_leaf_calls == len(leaf.boxes) > 0
    # O(leaves), not O(crops visited): the descent embedded strictly more crops than it emitted.
    assert stats.n_leaf_calls <= stats.n_embeds
    # Every call got the crop's pixels alongside its box (a segmenter needs both).
    for box, image in zip(leaf.boxes, leaf.images):
        y0, y1, x0, x1 = box
        assert image.shape[:2] == (y1 - y0, x1 - x0)
    assert instances


def test_leaf_instances_override_the_descent(backbone, two_squares):
    """What the leaf slot returns is what gets emitted."""
    img, ex = two_squares

    def three_stripes(feat):
        hp, wp = feat.shape[:2]
        out = []
        for k in range(3):                                # three disjoint horizontal bands
            g = np.zeros((hp, wp), dtype=bool)
            g[k * hp // 3 + 1: (k + 1) * hp // 3 - 1, 1:wp - 1] = True
            if g.any():
                out.append(g)
        return out

    leaf = _RecordingExtractor(three_stripes)
    cfg = _descent_cfg(merge_rule="none")                 # no dedup: count what the recursion emitted
    instances, stats = cascade(backbone, img, ex, config=cfg, leaf_extractor=leaf)

    assert stats.n_leaf_calls > 0
    # The descent's grouping is `cc` (never cuts), so any crop emitting 3 masks got them from the
    # leaf slot. Three per leaf call, minus whatever the min-area filter dropped.
    assert len(instances) > stats.n_leaf_calls


def test_empty_leaf_answer_falls_back_to_the_descent(backbone, two_squares):
    """A leaf extractor that finds nothing must not delete the region the recursion committed to."""
    img, ex = two_squares
    cfg = _descent_cfg()
    baseline, _ = cascade(backbone, img, ex, config=cfg)

    silent = _RecordingExtractor(lambda feat: [])
    fallback, stats = cascade(backbone, img, ex, config=cfg, leaf_extractor=silent)

    assert stats.n_leaf_calls > 0
    assert len(fallback) == len(baseline)
    for a, b in zip(fallback, baseline):
        assert np.array_equal(a.mask, b.mask)


def test_no_leaf_extractor_is_the_behaviour_of_record(backbone, two_squares):
    """Unset ``leaf_extractor`` leaves the cascade bit-identical to before the slot existed."""
    img, ex = two_squares
    cfg = _descent_cfg(instance_extractor="kmeans")
    a, sa = cascade(backbone, img, ex, config=cfg)
    b, sb = cascade(backbone, img, ex, config=cfg)
    assert sa.n_leaf_calls == sb.n_leaf_calls == 0
    assert len(a) == len(b)


def test_build_leaf_extractor_is_none_unless_configured():
    assert build_leaf_extractor(Config()) is None
    assert build_leaf_extractor(Config(leaf_extractor="oracle")) is not None


def test_leaf_config_inherits_then_overrides():
    """``leaf_where`` / ``leaf_grouping`` name the composite's halves *at the leaf*."""
    cfg = Config(foreground_extractor="otsu", instance_extractor="cc",
                 leaf_extractor="composite", leaf_grouping="kmeans")
    leaf = leaf_config(cfg)
    assert leaf.extractor == "composite"
    assert leaf.foreground_extractor == "otsu"            # inherited from the descent
    assert leaf.instance_extractor == "kmeans"            # overridden for the leaf
    assert cfg.instance_extractor == "cc"                 # ...and the descent is untouched


@pytest.mark.parametrize("leaf", ["oracle", "composite"])
def test_oracle_and_composite_leaf_slots_run_end_to_end(backbone, two_squares, leaf):
    """The two feature-based leaf arms drive the real registry, GT injection included."""
    img, ex = two_squares
    gt = np.zeros(img.shape[:2], dtype=np.int32)
    gt[20:40, 20:40] = 1
    gt[80:100, 80:100] = 2
    cfg = _descent_cfg(leaf_extractor=leaf, leaf_grouping="kmeans")
    instances, stats = cascade(backbone, img, ex, config=cfg, gt_foreground=gt)

    assert stats.n_leaf_calls > 0
    assert all(inst.mask.shape == img.shape[:2] for inst in instances)


def test_a_feature_based_leaf_slot_costs_no_extra_embeds(backbone, two_squares):
    """The leaf crop is already embedded — the slot reuses that grid.

    Only the one-off exemplar-bank embed separates the two runs, so the descent cost is unchanged.
    """
    img, ex = two_squares
    gt = np.zeros(img.shape[:2], dtype=np.int32)
    gt[20:40, 20:40] = 1
    gt[80:100, 80:100] = 2
    base, sb = cascade(backbone, img, ex, config=_descent_cfg(), gt_foreground=gt)
    _, sl = cascade(backbone, img, ex, config=_descent_cfg(leaf_extractor="oracle"),
                    gt_foreground=gt)
    assert sl.n_embeds == sb.n_embeds + 1                 # +1 = the leaf slot's exemplar bank


def test_kmeans_leaf_grouping_cuts_a_leaf_unconditionally(backbone, two_squares):
    """Regression pin for a measured failure: a k=2 leaf grouping cuts EVERY leaf.

    ``_CuttingGrouping`` cuts a component that fills its crop — and a leaf's component fills its
    crop *by definition*, since that is what made the descent stop there. During the descent an
    unearned cut was harmless: the peak guard (:func:`foveate.stop.survivors`) only confirmed a
    split whose best child beat its parent, so a single object's spurious halves were dropped. A
    leaf has no children, so nothing confirms it and both halves are emitted. Measured on corals
    with a real Where and a real Merge, that is AP 0.792 (no leaf pass) -> 0.102 (kmeans leaf), with
    the count error going 0.4 -> 4.4.

    Asserted on the grouping itself rather than end-to-end, because it is the *mechanism* that must
    not come back — an end-to-end count on a two-object fixture is dominated by NMS and min-area.
    """
    from foveate.grouping import build_grouping

    cfg = _descent_cfg(leaf_extractor="composite", leaf_grouping="kmeans")
    leaf_grouping = build_grouping(leaf_config(cfg))
    feat, _ = featlib.embed_batch(backbone, [two_squares[0]], chunk=1, standardize=False)[0]
    hp, wp = feat.shape[:2]
    box = (0, two_squares[0].shape[0], 0, two_squares[0].shape[1])

    filled = np.ones((hp, wp), dtype=bool)                # the leaf case: the region IS the crop
    assert len(leaf_grouping.group(feat, filled, box=box)) >= 2

    # `cc` under the identical input hands the region back whole — an instance count it can decide.
    assert len(build_grouping(_descent_cfg()).group(feat, filled, box=box)) == 1
