from experiments.run import run_experiment


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


def test_run_interval_minus_one_evaluates_all_non_train():
    # interval_images=-1 through the full runner: the inter (cross-image) eval covers every
    # qualifying image not in the train pool. With n_images=6 / train_images=4 that's 2 images.
    cfg = _config(["intra", "inter"])
    cfg["data"]["train_images"] = 4
    cfg["data"]["interval_images"] = -1
    result = run_experiment(cfg)
    assert "inter_ap" in result and result["inter_n_images"] >= 1
