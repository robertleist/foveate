"""Bridge the mask-based :mod:`data` package to foveate's discovery + eval.

An experiment iterates :class:`EvalItem`s: each is one image plus the **exemplar prompts**
(the labelled instances of a target concept) and the **ground-truth masks** to be rediscovered
(all instances of that concept in the image). Tensors from the dataset are converted to the
numpy ``HxWx3`` uint8 image and ``HxW`` boolean masks that :func:`foveate.discover_instances`
expects.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from data import DataConfig, InstanceDataset, InstanceSample


@dataclass
class EvalItem:
    image_id: str
    image: np.ndarray                  # (H, W, 3) uint8
    exemplar_masks: list[np.ndarray]   # prompts: (H, W) bool
    gt_masks: np.ndarray               # (M, H, W) bool — instances to discover
    class_id: int


def _image_to_numpy(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().float()
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    arr = arr[:3].permute(1, 2, 0).clamp(0, 1).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    return mask.detach().cpu().bool().numpy()


def sample_to_eval_item(
    sample: InstanceSample,
    image_id: str,
    *,
    max_exemplars: int = 3,
    exemplar_source: str = "auto",   # auto | train | val | known
) -> EvalItem | None:
    """Turn one :class:`InstanceSample` into an :class:`EvalItem`, or None if no prompt exists.

    The target concept is the majority class among the prompt instances. Prompts are the
    labelled instances of that class (capped at ``max_exemplars``); the GT is *all* instances
    of that class in the image.
    """
    if exemplar_source == "train":
        prompts = sample.train_instances
    elif exemplar_source == "val":
        prompts = sample.val_instances
    elif exemplar_source == "known":
        prompts = sample.known_instances
    else:  # auto: prefer the trained/labelled prompts, fall back to val.
        prompts = sample.train_instances or sample.val_instances or sample.known_instances
    if not prompts:
        return None

    target_class = Counter(inst.class_id for inst in prompts).most_common(1)[0][0]
    exemplars = [p for p in prompts if p.class_id == target_class][:max_exemplars]
    if not exemplars:
        return None

    gt = [inst.mask for inst in sample.instances if inst.class_id == target_class]
    if not gt:
        return None

    return EvalItem(
        image_id=image_id,
        image=_image_to_numpy(sample.image),
        exemplar_masks=[_mask_to_numpy(p.mask) for p in exemplars],
        gt_masks=np.stack([_mask_to_numpy(m) for m in gt]),
        class_id=int(target_class),
    )


def iter_eval_items(
    dataset: InstanceDataset,
    *,
    max_exemplars: int = 3,
    exemplar_source: str = "auto",
    limit: int | None = None,
):
    """Yield :class:`EvalItem`s for the (image-)samples of ``dataset`` that have a usable prompt."""
    n = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        item = sample_to_eval_item(
            sample, dataset.image_ids[idx],
            max_exemplars=max_exemplars, exemplar_source=exemplar_source,
        )
        if item is None:
            continue
        yield item
        n += 1
        if limit is not None and n >= limit:
            return


def build_eval_dataset(cfg: DataConfig) -> InstanceDataset:
    """Build the training dataset from a :class:`DataConfig` (the labelled instances are prompts)."""
    return InstanceDataset.from_config(cfg)
