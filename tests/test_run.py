import math

import numpy as np
import pytest

from experiments.datasets import build_datasets, iter_inter_items, iter_intra_items
from experiments.methods import Method, MethodPrediction, build_method, register_method
from experiments.methods.foveate_method import FoveateMethod
from experiments.run import run_experiment
from data import DataConfig


def _config(targets):
    return {
        "run_name": "test",
        "backbone": {"type": "mock", "image_size": 224, "patch_size": 14},
        "data": {
            "name": "synthetic", "categories": [], "min_instances": 2,
            "train_images": 4, "interval_images": 2, "known_ratio": 0.5, "seed": 0,
            "options": {"n_images": 6, "image_size": 96, "n_instances": 3, "n_categories": 2},
        },
        # standardize defaults to True: under the MockBackbone, z-scoring lifts the
        # all-black background patches off zero, so INSID3's cluster_all (cosine) over
        # the full target grid stays well-defined.
        "foveate": {"gate_threshold": 0.4, "min_crop": 24,
                    "cascade_min_instance_area": 4},
        "eval": {"targets": targets, "max_exemplars": 2},
        "mlflow": {"enabled": False},
    }


def test_run_intra():
    result = run_experiment(_config(["intra"]))
    assert "intra_ap" in result and "intra_pq" in result and "intra_mean_iou" in result
    assert result["intra_n_images"] >= 1


def test_run_intra_and_inter():
    result = run_experiment(_config(["intra", "inter"]))
    assert "intra_ap" in result and "inter_ap" in result
    assert result["inter_n_images"] >= 1


# ---------------------------------------------------------------------------
# Method abstraction
# ---------------------------------------------------------------------------
def _results_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        # Wall-clock / throughput — never bit-identical across runs.
        if k.endswith(("_runtime_s", "_throughput_img_s")):
            continue
        va, vb = a[k], b[k]
        if isinstance(va, float) and math.isnan(va):
            if not (isinstance(vb, float) and math.isnan(vb)):
                return False
        elif va != vb:
            return False
    return True


def test_default_method_is_foveate():
    # No `method:` block => FoveateMethod, so every existing config keeps working.
    assert isinstance(build_method(_config(["intra"])), FoveateMethod)


def test_explicit_foveate_method_is_identical():
    # `method: {type: foveate}` must behave exactly like the implicit default.
    base = run_experiment(_config(["intra"]))
    cfg = _config(["intra"])
    cfg["method"] = {"type": "foveate"}
    assert _results_equal(run_experiment(cfg), base)


def test_build_method_unknown_type_raises():
    cfg = _config(["intra"])
    cfg["method"] = {"type": "does-not-exist"}
    with pytest.raises(ValueError, match="unknown method type 'does-not-exist'"):
        build_method(cfg)


@register_method("_test_gt_oracle")
class _GTOracleMethod(Method):
    """Returns the GT masks verbatim; records the items it saw (registry is process-global,
    so the name is test-prefixed to avoid colliding with real methods)."""

    seen: list = []

    def predict(self, item, observer=None):
        _GTOracleMethod.seen.append(item)
        masks = item.gt_masks.astype(bool)
        return MethodPrediction(masks=masks, scores=np.ones(masks.shape[0]), n_embeds=0)


def test_custom_registered_method_is_dispatched():
    _GTOracleMethod.seen.clear()
    cfg = _config(["intra"])
    cfg["method"] = {"type": "_test_gt_oracle"}
    result = run_experiment(cfg)
    assert _GTOracleMethod.seen, "run_experiment never called the registered method"
    # A GT oracle scores perfectly and spends no embeds.
    assert result["intra_ap"] == pytest.approx(1.0)
    assert result["intra_mean_embeds"] == 0
    # class_name is plumbed through EvalItem for text-prompted baselines.
    assert all(item.class_name and item.class_name.startswith("colour")
               for item in _GTOracleMethod.seen)


def test_eval_items_carry_class_names():
    intra_ds, inter_ds = build_datasets(DataConfig.from_dict(_config(["intra"])["data"]))
    names = intra_ds.class_names
    assert names and all(v == f"colour{k}" for k, v in names.items())
    intra = list(iter_intra_items(intra_ds, max_exemplars=2))
    assert intra and all(it.class_name == names[it.class_id] for it in intra)
    from experiments.datasets import build_support_index
    inter = list(iter_inter_items(inter_ds, build_support_index(intra_ds), max_exemplars=2))
    assert inter and all(it.class_name == names[it.class_id] for it in inter)


def test_run_interval_minus_one_evaluates_all_non_train():
    # interval_images=-1 through the full runner: the inter (cross-image) eval covers every
    # qualifying image not in the train pool. With n_images=6 / train_images=4 that's 2 images.
    cfg = _config(["intra", "inter"])
    cfg["data"]["train_images"] = 4
    cfg["data"]["interval_images"] = -1
    result = run_experiment(cfg)
    assert "inter_ap" in result and result["inter_n_images"] >= 1
