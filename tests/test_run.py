from experiments.run import run_experiment


def test_run_experiment_offline(tmp_path):
    config = {
        "run_name": "test",
        "output_dir": str(tmp_path / "out"),
        "backbone": {"type": "mock", "image_size": 224, "patch_size": 14},
        "data": {
            "name": "synthetic", "categories": [], "min_instances": 2,
            "train_images": 3, "known_ratio": 0.5, "seed": 0,
            "options": {"n_images": 3, "image_size": 96, "n_instances": 3, "n_categories": 2},
        },
        "foveate": {"gate_threshold": 0.4, "min_crop": 24, "cascade_min_instance_area": 4},
        "eval": {"max_exemplars": 2, "exemplar_source": "auto"},
        "mlflow": {"enabled": False},
    }
    result = run_experiment(config)
    assert "ap" in result and "pq" in result and "mean_iou" in result
    assert result["n_images"] >= 1
    assert (tmp_path / "out" / "metrics.json").exists()
