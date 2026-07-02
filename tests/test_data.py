import torch

from data import DataConfig, InstanceDataset
from experiments.datasets import (
    build_support_index,
    iter_inter_items,
    iter_intra_items,
)


def _cfg(**kw):
    base = dict(
        name="synthetic", categories=[], min_instances=2, train_images=4,
        interval_images=2, known_ratio=0.5, seed=0,
        options=dict(n_images=6, image_size=96, n_instances=4, n_categories=2),
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


def test_known_unknown_partitions_all_instances():
    ds = InstanceDataset.from_config(_cfg())
    sample = ds[0]
    n = len(sample.instances)
    idx = set(sample.known_idx) | set(sample.unknown_idx)
    assert idx == set(range(n))
    assert len(sample.known_idx) + len(sample.unknown_idx) == n          # disjoint, exhaustive
    assert len(sample.known_idx) >= 1                                    # at least one prompt


def test_build_returns_disjoint_intra_inter():
    intra, inter = InstanceDataset.build(_cfg())
    assert inter is not None
    assert set(intra.image_ids).isdisjoint(set(inter.image_ids))
    assert len(intra.image_ids) == 4 and len(inter.image_ids) == 2


def test_build_interval_minus_one_uses_all_remaining_images():
    # interval_images=-1 → every qualifying image not in the train pool becomes the inter/eval set.
    intra, inter = InstanceDataset.build(_cfg(train_images=4, interval_images=-1))
    assert inter is not None
    assert len(intra.image_ids) == 4
    assert len(inter.image_ids) == 2                        # 6 qualifying - 4 train = all the rest
    assert set(intra.image_ids).isdisjoint(set(inter.image_ids))
    assert len(set(intra.image_ids) | set(inter.image_ids)) == 6   # together cover the whole pool


def test_intra_items_prompt_with_known_discover_unknown():
    ds = InstanceDataset.from_config(_cfg())
    items = list(iter_intra_items(ds, max_exemplars=2))
    assert items
    for it in items:
        assert it.exemplar_image is None                 # same-image discovery
        assert it.image.ndim == 3 and it.image.shape[2] == 3
        assert len(it.exemplar_masks) >= 1 and it.gt_masks.shape[0] >= 1


def test_inter_items_are_cross_image():
    intra, inter = InstanceDataset.build(_cfg())
    support = build_support_index(intra)
    items = list(iter_inter_items(inter, support, max_exemplars=2))
    assert items
    for it in items:
        assert it.exemplar_image is not None             # prompts come from a different image
        assert it.gt_masks.shape[0] >= 1


def test_target_size_letterbox():
    ds = InstanceDataset.from_config(_cfg(target_size=(128, 128)))
    sample = ds[0]
    assert sample.image.shape[1:] == (128, 128)
    assert sample.semantic_masks.shape[1:] == (128, 128)
