import numpy as np
import pytest

from foveate import Config, cascade
from foveate.cascade import _component_scores

# The three slots are covered by tests/test_extract.py, tests/test_stop.py and
# tests/test_merge_rule.py; what is tested here is the cascade *loop* — how it drives them.


def test_cascade_runs_with_otsu_where(backbone, two_squares):
    """The Otsu Where extractor drives the full cascade end-to-end."""
    img, ex = two_squares
    cfg = Config(foreground_extractor="otsu", debias=False, min_crop=24,
                 cascade_min_instance_area=4)
    instances, stats = cascade(backbone, img, ex, config=cfg)
    assert stats.n_embeds > 0
    assert all(inst.mask.shape == img.shape[:2] for inst in instances)


def test_cascade_masked_confidence_scores_each_instance_on_its_own_foreground(backbone, two_squares):
    """With confidence_reid_mode set, leaves are scored by the masked confidence scorer (per
    instance), not by the shared recursion CLS score. The run stays valid and scores finite."""
    img, ex = two_squares
    gt = np.zeros(img.shape[:2], dtype=bool)
    gt[20:40, 20:40] = True
    gt[80:100, 80:100] = True
    cfg = Config(foreground_extractor="oracle", debias=False, min_crop=24,
                 cascade_min_instance_area=4, confidence_reid_mode="full")
    instances, stats = cascade(backbone, img, ex, config=cfg, gt_foreground=gt)
    assert stats.n_embeds > 0
    # Masked cosine scores, finite and within [-1, 1] up to float32 slack.
    assert all(np.isfinite(inst.score) and abs(inst.score) <= 1.0 + 1e-5 for inst in instances)


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


@pytest.mark.parametrize("cfg_kwargs", [
    dict(extractor="oracle"),                            # the monolithic GT extractor
    dict(foreground_extractor="oracle_cc"),              # the legacy spelling that resolves to it
    dict(instance_extractor="oracle"),                   # ...and the other one
])
def test_cascade_runs_with_the_monolithic_oracle(backbone, two_squares, cfg_kwargs):
    """Every spelling of "the decomposition is oracular" drives the cascade off the injected GT."""
    img, ex = two_squares
    labels = np.zeros(img.shape[:2], dtype=np.int32)     # two GT instances (the red squares)
    labels[20:40, 20:40] = 1
    labels[80:100, 80:100] = 2
    cfg = Config(debias=False, min_crop=24, cascade_min_instance_area=4, **cfg_kwargs)
    instances, stats = cascade(backbone, img, ex, config=cfg, gt_foreground=labels)
    assert stats.n_embeds > 0
    assert all(inst.mask.shape == img.shape[:2] for inst in instances)


def test_cascade_runs_with_both_connectivities(backbone, two_squares):
    """Extract connectivity is an ablatable axis: 4 and 8 both run the cascade."""
    img, ex = two_squares
    for conn in (4, 8):
        cfg = Config(extract_connectivity=conn, min_crop=24, cascade_min_instance_area=4)
        instances, stats = cascade(backbone, img, ex, config=cfg)
        assert stats.n_embeds > 0


def test_cls_worse_children_are_traced(backbone, two_squares, monkeypatch):
    """Dropped (reid-worse) children must be emitted as terminal ``reid-worse`` events so the
    trajectory / tool can show why the cascade stopped."""
    import importlib
    C = importlib.import_module("foveate.cascade")  # module handle (the `cascade` attr on the
                                                    # package is the function, so fetch the module)
    import foveate.stop as stop_mod

    # Force "no child ever survives" → the reid-stop path fires and every child is dropped. Patching
    # the Stop slot's rule (which ReidStopRule resolves at call time) is the seam now.
    monkeypatch.setattr(stop_mod, "survivors",
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


def test_two_separate_instances_are_not_regressed(backbone, two_squares):
    """The clean case the whole cascade exists for: two separate components, two instances out."""
    img, ex = two_squares
    cfg = Config(min_crop=24, cascade_min_instance_area=4)
    instances, stats = cascade(backbone, img, ex, config=cfg)
    assert len(instances) == 2


def test_the_forced_split_retry_is_gone(backbone, two_squares):
    """``zoom_split_retry_eps`` hedged the always-split-then-confirm dance, which the two-slot
    contract removed: the extractor decides the instance count, so there is nothing to retry. The key
    keeps resolving (every frozen config sets it) but must no longer change the run."""
    img, ex = two_squares
    base = dict(min_crop=24, cascade_min_instance_area=4)
    ev0 = []; inst0, st0 = cascade(backbone, img, ex, observer=ev0.append,
                                   config=Config(zoom_split_retry_eps=0.0, **base))
    ev1 = []; inst1, st1 = cascade(backbone, img, ex, observer=ev1.append,
                                   config=Config(zoom_split_retry_eps=1.0, **base))
    assert st0.n_embeds == st1.n_embeds and len(inst0) == len(inst1) == 2
    assert [e["decision"] for e in ev0] == [e["decision"] for e in ev1]
    assert not any(e["decision"] == "clump-split" for e in ev0 + ev1)


def test_the_fixed_point_ends_a_branch_that_stopped_changing(backbone, two_squares):
    """A crop whose extraction reproduces the instance it was cropped for is emitted there.

    Tightening the tolerance to 1.0 effectively disables the fixed point, so the descent then runs
    on until the geometric shrink or the peak guard stops it — strictly more crops, same instances.
    """
    img, ex = two_squares
    base = dict(min_crop=8, cascade_min_instance_area=4)
    on, st_on = cascade(backbone, img, ex,
                        config=Config(stop_fixed_point_iou=0.5, **base))
    off, st_off = cascade(backbone, img, ex,
                          config=Config(stop_fixed_point_iou=1.0, **base))
    assert st_on.n_embeds < st_off.n_embeds
    assert len(on) == len(off) == 2


def test_emit_components_splits_multi_component_parent(backbone, two_squares):
    """``emit_components`` emits an emitted parent's instances separately instead of OR-merging
    them into one mask. Here the embed budget is cut off (``max_total_embeds=2``) right after the root
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


def test_reid_mode_masked_is_independent_of_the_extractor(backbone, two_squares):
    """The masked re-id score g is decoupled from the Extract slot: it drives the cascade even with
    a non-INSID3 Where half (here otsu), because build_reid_scorer builds its own exemplar bank
    rather than borrowing the extractor's."""
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
