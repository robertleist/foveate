from foveate import Config, run


def test_single_pass_pipeline_runs(backbone, two_squares):
    img, ex = two_squares
    res = run(backbone, img, ex, params=Config(gate_threshold=0.4))
    assert res.instance_masks.ndim == 3
    assert res.instance_masks.shape[1:] == img.shape[:2]
    assert res.instance_scores.shape[0] == res.instance_masks.shape[0]
    assert res.stats["foreground_patches"] > 0
