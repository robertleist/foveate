from experiments.run import run_experiment


def _config(tmp_path, targets):
    return {
        "run_name": "test",
        "output_dir": str(tmp_path / "out"),
        "backbone": {"type": "mock", "image_size": 224, "patch_size": 14},
        "data": {
            "name": "synthetic", "categories": [], "min_instances": 2,
            "train_images": 4, "interval_images": 2, "known_ratio": 0.5, "seed": 0,
            "options": {"n_images": 6, "image_size": 96, "n_instances": 3, "n_categories": 2},
        },
        "foveate": {"standardize": False, "gate_threshold": 0.4, "min_crop": 24,
                    "cascade_min_instance_area": 4},
        "eval": {"targets": targets, "max_exemplars": 2},
        "mlflow": {"enabled": False},
    }


def test_run_intra(tmp_path):
    result = run_experiment(_config(tmp_path, ["intra"]))
    assert "intra_ap" in result and "intra_pq" in result and "intra_mean_iou" in result
    assert result["intra_n_images"] >= 1
    assert (tmp_path / "out" / "metrics.json").exists()


def test_run_intra_and_inter(tmp_path):
    result = run_experiment(_config(tmp_path, ["intra", "inter"]))
    assert "intra_ap" in result and "inter_ap" in result
    assert result["inter_n_images"] >= 1
    # Cross-image masks land in their own subdir.
    assert (tmp_path / "out" / "inter").exists()
