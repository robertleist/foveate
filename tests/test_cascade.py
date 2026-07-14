import numpy as np
import pytest

import torch

from foveate import Config, cascade
from foveate.cascade import (
    _CONN4, _CONN8, _component_scores, _extract_structure, _reid_survivors, _mask_overlap, _nms,
    _split_kmeans, _survivors,
)
from foveate.types import Instance


def test_extract_structure_selects_connectivity():
    """The Extract stage maps connectivity 4/8 to the right structuring element."""
    assert _extract_structure(4) is _CONN4
    assert _extract_structure(8) is _CONN8
    assert _extract_structure(99) is _CONN8              # anything else → 8-connectivity default


def test_cascade_runs_with_otsu_where(backbone, two_squares):
    """The Otsu Where extractor drives the full cascade end-to-end."""
    img, ex = two_squares
    cfg = Config(foreground_extractor="otsu", debias=False, min_crop=24,
                 cascade_min_instance_area=4)
    instances, stats = cascade(backbone, img, ex, config=cfg)
    assert stats.n_embeds > 0
    assert all(inst.mask.shape == img.shape[:2] for inst in instances)


def test_cascade_runs_with_oracle_where(backbone, two_squares):
    """The oracle Where extractor drives the full cascade off the injected GT foreground."""
    img, ex = two_squares
    gt = np.zeros(img.shape[:2], dtype=bool)              # both red squares are the class GT
    gt[20:40, 20:40] = True
    gt[80:100, 80:100] = True
    cfg = Config(foreground_extractor="oracle", debias=False, min_crop=24,
                 cascade_min_instance_area=4)
    instances, stats = cascade(backbone, img, ex, config=cfg, gt_foreground=gt)
    assert stats.n_embeds > 0
    assert all(inst.mask.shape == img.shape[:2] for inst in instances)
    # Perfect Where: every emitted mask overlaps the GT foreground and is dominated by it (patch-grid
    # quantization on upsampling can bleed a few pixels past the exact GT boundary, so allow slack).
    for inst in instances:
        m = inst.mask.astype(bool)
        assert (m & gt).sum() > (m & ~gt).sum()


def test_cascade_runs_with_both_connectivities(backbone, two_squares):
    """Extract connectivity is an ablatable axis: 4 and 8 both run the cascade."""
    img, ex = two_squares
    for conn in (4, 8):
        cfg = Config(extract_connectivity=conn, min_crop=24, cascade_min_instance_area=4)
        instances, stats = cascade(backbone, img, ex, config=cfg)
        assert stats.n_embeds > 0


def _inst(mask: np.ndarray, score: float) -> Instance:
    ys, xs = np.where(mask)
    box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    return Instance(mask.astype(np.uint8), box, 0, score)


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


def test_cls_survivors_stop_rule():
    # No child beats the parent → empty → the parent is the CLS peak (reid-stop fires).
    assert _reid_survivors(0.8, [0.7, 0.8, 0.75]) == []      # ties (0.8) do NOT survive
    # Only children strictly above the parent continue; the lower-sim sibling is dropped.
    assert _reid_survivors(0.6, [0.7, 0.5, 0.9]) == [0, 2]
    # A single worse child (the plain zoom case) → stop and emit the predecessor.
    assert _reid_survivors(0.75, [0.6]) == []
    # A single better child → keep zooming, parent superseded.
    assert _reid_survivors(0.5, [0.55]) == [0]


def test_survivors_single_child_is_strict_zoom_guard():
    """One child (zoom) uses the strict beat-the-parent peak guard — the floor is irrelevant."""
    assert _survivors(0.8, [0.7], crop_sim_floor=0.5) == []      # over-zoomed → stop
    assert _survivors(0.5, [0.55], crop_sim_floor=0.5) == [0]    # improved → keep zooming


def test_survivors_keeps_novel_sibling_below_biased_parent():
    """A crop holds the exemplar AND a novel instance, so its parent CLS (0.85) is inflated by the
    exemplar but still diluted below the *isolated* exemplar sub-crop (0.90). The novel sub-crop
    (0.62) scores below the parent but above the class floor — it must be KEPT, not discarded."""
    keep = _survivors(0.85, [0.90, 0.62], crop_sim_floor=0.5)
    assert keep == [0, 1]                       # child 0 (>parent) confirms; child 1 clears the floor


def test_survivors_drops_below_floor_sibling():
    """A confirmed split still drops any sub-crop that fails the class floor (not the class)."""
    assert _survivors(0.85, [0.90, 0.30], crop_sim_floor=0.5) == [0]


def test_survivors_keeps_improved_child_below_floor():
    """Regression: a child that IMPROVED on its (low) parent must survive even when it is still
    below the class floor — it found a better crop and keeps zooming toward the floor. Parent 0.081,
    floor 0.30: the 0.154 child beat the parent, so it must be PURSUED, not pruned for being below
    the floor (the old rule kept only ``s >= floor`` and wrongly dropped it)."""
    assert _survivors(0.081, [0.154, 0.05], crop_sim_floor=0.30) == [0]   # 0.05 below both → pruned
    # a strong sibling (0.40) confirms the split; the weaker 0.154 still improved on the parent → kept
    assert _survivors(0.081, [0.40, 0.154], crop_sim_floor=0.30) == [0, 1]


def test_survivors_rejects_fragmented_single_object():
    """Splitting a single object → every half is a weaker partial view (best <= parent) → the split
    is not confirmed → keep nothing so the caller emits the parent."""
    assert _survivors(0.90, [0.80, 0.75], crop_sim_floor=0.5) == []    # partial halves, both worse
    assert _survivors(0.90, [0.90, 0.90], crop_sim_floor=0.5) == []    # flat/tied CLS never confirms


def test_cls_worse_children_are_traced(backbone, two_squares, monkeypatch):
    """Dropped (reid-worse) children must be emitted as terminal ``reid-worse`` events so the
    trajectory / tool can show why the cascade stopped."""
    import importlib
    C = importlib.import_module("foveate.cascade")  # module handle (the `cascade` attr on the
                                                    # package is the function, so fetch the module)

    # Force "no child ever survives" → the reid-stop path fires and every child is dropped.
    monkeypatch.setattr(C, "_survivors",
                        lambda parent_reid, child_scores, *, crop_sim_floor: [])
    img, ex = two_squares
    events = []
    C.cascade(backbone, img, ex,
              config=Config(min_crop=24, cascade_min_instance_area=4),
              observer=events.append)
    worse = [e for e in events if e["decision"] == "reid-worse"]
    assert worse, "expected reid-worse events for dropped children"
    for e in worse:                                          # well-formed terminal trace nodes
        assert "parent_reid" in e and e["children"] == [] and e["instance_grids"] == []
        assert "box" in e and "reid_score" in e
    assert any(e["decision"] == "reid-stop" for e in events)  # the predecessor was emitted instead


def test_mask_overlap_iou_and_containment():
    H = W = 20
    big = np.zeros((H, W), bool); big[2:18, 2:18] = True        # area 256
    small = np.zeros((H, W), bool); small[6:14, 6:14] = True    # area 64, fully inside big
    iou, contain = _mask_overlap(big, small)
    assert abs(contain - 1.0) < 1e-9                            # the smaller is fully contained
    assert iou < 0.5                                            # ...but IoU is low (nested) → needs contain
    disjoint = np.zeros((H, W), bool); disjoint[0:2, 0:2] = True
    assert _mask_overlap(big, disjoint) == (0.0, 0.0)


def test_nms_suppresses_nested_duplicate_keeps_higher_score():
    """The nested-split failure mode: two detections of one object, one tight (higher CLS) inside a
    looser one. NMS keeps the higher-scoring (tighter) detection and drops the nested duplicate."""
    H = W = 24
    loose = np.zeros((H, W), bool); loose[2:22, 2:22] = True
    tight = np.zeros((H, W), bool); tight[7:17, 7:17] = True    # nested inside loose
    far = np.zeros((H, W), bool); far[0:4, 20:24] = True        # disjoint → survives
    kept, n = _nms([_inst(loose, 0.7), _inst(tight, 0.9), _inst(far, 0.6)], 0.5, 0.7)
    assert n == 1
    masks = {int(k.mask.sum()) for k in kept}
    assert masks == {int(tight.sum()), int(far.sum())}         # loose (nested dup) suppressed


def test_nms_keeps_distinct_instances():
    """Disjoint instances must never be merged (the clean two-instance case)."""
    H = W = 24
    a = np.zeros((H, W), bool); a[2:8, 2:8] = True
    b = np.zeros((H, W), bool); b[16:22, 16:22] = True
    kept, n = _nms([_inst(a, 0.8), _inst(b, 0.7)], 0.5, 0.7)
    assert n == 0 and len(kept) == 2


def test_nms_disabled_when_thresholds_are_one(backbone, two_squares):
    """Both thresholds at 1.0 turns NMS off — no leaf is suppressed."""
    img, ex = two_squares
    _, stats = cascade(backbone, img, ex,
                       config=Config(min_crop=24, cascade_min_instance_area=4,
                                                nms_iou=1.0, nms_containment=1.0))
    assert stats.suppressed == 0


def test_new_acceptance_config_defaults():
    cfg = Config()
    assert cfg.split_aggregate == "mean"
    assert cfg.split_margin == 0.0
    assert cfg.boundary_smooth_sigma == 0.0


def test_converged_split_does_not_regress_two_instances(backbone, two_squares):
    """The always-split-at-convergence + CLS-survivor rule must not regress the clean
    two-instance case (the two squares are separate components, so they split cleanly)."""
    img, ex = two_squares
    cfg = Config(min_crop=24, cascade_min_instance_area=4)
    instances, stats = cascade(backbone, img, ex, config=cfg)
    assert len(instances) == 2


def test_zoom_split_retry_attempts_split_on_marginal_peak(backbone, two_squares):
    """A zoom that peaks by <= eps tries a k=2 split of the parent before emitting. With a large
    eps the retry fires (a ``clump-split`` appears where a plain ``reid-stop`` would be), yet the
    strict split-confirm still falls back to emitting the parent — so no over-split / duplicates."""
    img, ex = two_squares
    off = Config(min_crop=24, cascade_min_instance_area=4, zoom_split_retry_eps=0.0)
    ev0 = []; inst0, st0 = cascade(backbone, img, ex, config=off, observer=ev0.append)
    on = Config(min_crop=24, cascade_min_instance_area=4, zoom_split_retry_eps=1.0)
    ev1 = []; inst1, st1 = cascade(backbone, img, ex, config=on, observer=ev1.append)

    assert not any(e["decision"] == "clump-split" for e in ev0)   # disabled → no retry split
    assert any(e["decision"] == "clump-split" for e in ev1)       # enabled → a marginal peak retried
    assert st1.n_embeds > st0.n_embeds                            # the retry paid for 2 sub-crops
    assert len(inst0) == 2 and len(inst1) == 2                    # correctness preserved either way


def test_emit_components_splits_multi_component_parent(backbone, two_squares):
    """``emit_components`` splits an emitted parent's OR-merged foreground into connected
    components. Here the embed budget is cut off (``max_total_embeds=2``) right after the root
    splits into the two red squares but before their child crops are processed, so the flush
    falls back to emitting the parent — whose two components don't touch. Default fuses them into
    one instance; ``emit_components`` recovers both."""
    img, ex = two_squares
    # crop_sim_floor=-1 so the parent always clears the floor regardless of MockBackbone cosines.
    base = dict(min_crop=24, cascade_min_instance_area=4, crop_sim_floor=-1.0, max_total_embeds=2)
    merged, _ = cascade(backbone, img, ex, config=Config(emit_components=False, **base))
    split, _ = cascade(backbone, img, ex, config=Config(emit_components=True, **base))
    assert len(merged) == 1                      # two non-touching squares OR-merged into one
    assert len(split) == 2                       # ... recovered as two separate instances


def test_component_scores_downweight_cutoff_sliver():
    """A border-touching component (a cut-off sliver) is scaled by its area fraction of the
    dominant component; interior components keep the crop's g. This is what stops the sliver — which
    carries the exemplar's inflated g — from outranking a neighbour's full detection in NMS."""
    shape = (8, 8)
    whole = np.zeros(shape, dtype=bool); whole[2:6, 2:6] = True          # interior, 16 px, largest
    sliver = np.zeros(shape, dtype=bool); sliver[0, 3:5] = True          # touches top border, 2 px
    scores = _component_scores([whole, sliver], base_score=0.9, grid_shape=shape)
    assert scores[0] == 0.9                                              # whole → g unchanged
    assert scores[1] == pytest.approx(0.9 * 2 / 16)                      # sliver → area-scaled down
    assert scores[1] < scores[0]                                         # so the full detection wins


def test_component_scores_interior_components_keep_g():
    """Two non-touching interior instances both keep the full g (no false area penalty)."""
    shape = (10, 10)
    a = np.zeros(shape, dtype=bool); a[2:4, 2:4] = True                  # interior, small
    b = np.zeros(shape, dtype=bool); b[5:9, 5:9] = True                  # interior, large
    scores = _component_scores([a, b], base_score=0.7, grid_shape=shape)
    assert scores == [0.7, 0.7]


@pytest.mark.parametrize("mode", ["mean", "kmeans", "full"])
def test_reid_masked_modes_run_and_discover_targets(backbone, two_squares, mode):
    """Each masked g mode drives the full cascade end-to-end (scoring the extracted foreground).
    crop_sim_floor is lowered because masked-cosine g lives on a different scale than CLS cosine."""
    img, ex = two_squares
    cfg = Config(reid_mode=mode, reid_kmeans_k=2, crop_sim_floor=0.0,
                 min_crop=24, cascade_min_instance_area=4)
    instances, stats = cascade(backbone, img, ex, config=cfg)
    assert stats.n_embeds > 0
    assert len(instances) == 2                               # the two red squares, not the distractor
    for inst in instances:
        ys, xs = np.where(inst.mask)
        cy, cx = ys.mean(), xs.mean()
        assert not (cy < 60 and cx > 60), "discovered the blue distractor"


def test_reid_mode_masked_is_independent_of_where_extractor(backbone, two_squares):
    """The masked re-id score g is decoupled from the Where stage: it drives the cascade even with
    a non-INSID3 foreground extractor (here otsu), because build_reid_scorer builds its own
    exemplar bank rather than borrowing the extractor's."""
    img, ex = two_squares
    cfg = Config(reid_mode="mean", foreground_extractor="otsu", debias=False,
                 crop_sim_floor=0.0, min_crop=24, cascade_min_instance_area=4)
    instances, stats = cascade(backbone, img, ex, config=cfg)
    assert stats.n_embeds > 0
    assert len(instances) >= 1                               # runs end-to-end, no coupling error


def test_max_depth_is_not_a_stopping_signal(backbone, two_squares):
    """Depth is no longer a stopping signal — even ``max_depth=0`` must not force a leaf-cap at
    the root; only ``min_crop`` (size) and CLS halt the cascade, so the zoom goes past depth 0."""
    img, ex = two_squares
    events = []
    cascade(backbone, img, ex,
            config=Config(max_depth=0, min_crop=24, cascade_min_instance_area=4),
            observer=events.append)
    assert max(e["depth"] for e in events) > 0, "max_depth=0 stopped the cascade at the root"


def test_discovers_both_targets_not_distractor(backbone, two_squares):
    """Default extractor (INSID3) finds the two red squares, not the blue distractor."""
    img, ex = two_squares
    # standardize=True (default): MockBackbone black-background patches become non-zero,
    # so INSID3's cluster_all over the full grid stays well-defined.
    cfg = Config(min_crop=24, cascade_min_instance_area=4)
    instances, stats = cascade(backbone, img, ex, config=cfg)

    assert len(instances) == 2
    assert stats.leaves == 2
    # Each discovered mask should sit on a red square, not the blue distractor at (20:40, 80:100).
    for inst in instances:
        ys, xs = np.where(inst.mask)
        cy, cx = ys.mean(), xs.mean()
        assert not (cy < 60 and cx > 60), "discovered the blue distractor"


def test_bank_extractor_discovers_both_targets(backbone, two_squares):
    """The 'bank' foreground strategy stays covered: two red squares, no distractor."""
    img, ex = two_squares
    cfg = Config(foreground_extractor="bank", gate_threshold=0.8,
                 min_crop=24, cascade_min_instance_area=4)
    instances, stats = cascade(backbone, img, ex, config=cfg)

    assert len(instances) == 2
    assert stats.leaves == 2
    for inst in instances:
        ys, xs = np.where(inst.mask)
        cy, cx = ys.mean(), xs.mean()
        assert not (cy < 60 and cx > 60), "discovered the blue distractor"


def test_config_from_dict_overrides():
    cfg = Config.from_dict({"gate_threshold": 0.7, "prototype_budget": 8, "bogus": 1})
    assert cfg.gate_threshold == 0.7
    assert cfg.prototype_budget == 8
    assert not hasattr(cfg, "bogus")


def test_empty_exemplar_raises(backbone, two_squares):
    img, _ = two_squares
    empty = [np.zeros((128, 128), np.uint8)]
    try:
        cascade(backbone, img, empty, config=Config())
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty exemplar masks")


def test_observer_is_called(backbone, two_squares):
    img, ex = two_squares
    seen = []
    cascade(backbone, img, ex, config=Config(gate_threshold=0.4, min_crop=24),
            observer=lambda info: seen.append(info["decision"]))
    assert seen and all("decision" for _ in seen)


def test_observer_events_carry_internals_and_instance_grids(backbone, two_squares):
    img, ex = two_squares
    events = []
    cascade(backbone, img, ex,
            config=Config(min_crop=24, cascade_min_instance_area=4),
            observer=events.append)
    assert events
    for e in events:
        assert "internals" in e and isinstance(e["internals"], dict)
        assert "instance_grids" in e and isinstance(e["instance_grids"], list)
    # at least one region produced instances (a split or a leaf)
    assert any(e["instance_grids"] for e in events)
    # INSID3 internals expose the per-crop selection + params on non-empty events
    sel = [e for e in events if e["internals"]]
    assert sel and "tau_used" in sel[0]["internals"]
