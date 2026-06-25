import torch

from data import DataConfig, InstanceDataset
from experiments.datasets import iter_eval_items


def _cfg(**kw):
    base = dict(
        name="synthetic", categories=[], min_instances=2, train_images=4,
        known_ratio=0.5, seed=0,
        options=dict(n_images=4, image_size=96, n_instances=4, n_categories=2),
    )
    base.update(kw)
    return DataConfig.from_dict(base)


def test_sample_is_mask_based():
    ds = InstanceDataset.from_config(_cfg())
    sample = ds[0]
    assert not hasattr(sample, "flow_map")
    assert sample.semantic_masks.dtype == torch.bool
    assert sample.semantic_masks.shape[0] == ds.num_classes
    assert sample.semantic_masks.shape[1:] == sample.image.shape[1:]
    assert len(sample.instances) > 0


def test_pu_split_partitions_all_instances():
    ds = InstanceDataset.from_config(_cfg())
    sample = ds[0]
    n = len(sample.instances)
    idx = set(sample.train_idx) | set(sample.val_idx) | set(sample.unlabelled_idx)
    assert idx == set(range(n))
    # No instance appears in two partitions.
    assert len(sample.train_idx) + len(sample.val_idx) + len(sample.unlabelled_idx) == n


def test_eval_items_have_prompt_and_gt():
    ds = InstanceDataset.from_config(_cfg())
    items = list(iter_eval_items(ds, max_exemplars=2))
    assert items
    for it in items:
        assert it.image.ndim == 3 and it.image.shape[2] == 3
        assert len(it.exemplar_masks) >= 1
        assert it.gt_masks.shape[0] >= 1
        # Every exemplar's class matches the item class.
        assert all(m.shape == it.image.shape[:2] for m in it.exemplar_masks)


def test_target_size_letterbox():
    ds = InstanceDataset.from_config(_cfg(target_size=(128, 128)))
    sample = ds[0]
    assert sample.image.shape[1:] == (128, 128)
    assert sample.semantic_masks.shape[1:] == (128, 128)
