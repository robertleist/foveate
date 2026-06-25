import numpy as np

from foveate import Config, discover_instances


def test_discovers_both_targets_not_distractor(backbone, two_squares):
    img, ex = two_squares
    cfg = Config(gate_threshold=0.4, min_crop=24, cascade_min_instance_area=4)
    instances, stats = discover_instances(backbone, img, ex, config=cfg)

    assert len(instances) == 2
    assert stats.leaves == 2
    # Each discovered mask should sit on a red square, not the blue distractor at (20:40, 80:100).
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
