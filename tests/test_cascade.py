import numpy as np

from foveate import Config, discover_instances
from foveate.cascade import _aggregate


def test_aggregate_modes():
    scores = [0.2, 0.6, 0.4]
    assert _aggregate(scores, "max") == 0.6
    assert _aggregate(scores, "min") == 0.2
    assert abs(_aggregate(scores, "mean") - 0.4) < 1e-9
    assert abs(_aggregate(scores, "anything-else") - 0.4) < 1e-9  # defaults to mean


def test_new_acceptance_config_defaults():
    cfg = Config()
    assert cfg.split_aggregate == "mean"
    assert cfg.split_margin == 0.0
    assert cfg.boundary_smooth_sigma == 0.0


def test_split_aggregate_modes_still_discover_both(backbone, two_squares):
    """The CLS-subsplit acceptance must not regress the clean two-instance case."""
    img, ex = two_squares
    for mode in ("mean", "max", "min"):
        cfg = Config(min_crop=24, cascade_min_instance_area=4, split_aggregate=mode)
        instances, stats = discover_instances(backbone, img, ex, config=cfg)
        assert len(instances) == 2, mode


def test_discovers_both_targets_not_distractor(backbone, two_squares):
    """Default extractor (INSID3) finds the two red squares, not the blue distractor."""
    img, ex = two_squares
    # standardize=True (default): MockBackbone black-background patches become non-zero,
    # so INSID3's cluster_all over the full grid stays well-defined.
    cfg = Config(min_crop=24, cascade_min_instance_area=4)
    instances, stats = discover_instances(backbone, img, ex, config=cfg)

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
    instances, stats = discover_instances(backbone, img, ex, config=cfg)

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
        discover_instances(backbone, img, empty, config=Config())
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty exemplar masks")


def test_observer_is_called(backbone, two_squares):
    img, ex = two_squares
    seen = []
    discover_instances(backbone, img, ex, config=Config(gate_threshold=0.4, min_crop=24),
                       observer=lambda info: seen.append(info["decision"]))
    assert seen and all("decision" for _ in seen)
