"""Bridge the mask-based :mod:`data` package to foveate's discovery + eval.

Two evaluation protocols, both yielding :class:`EvalItem`s:

* **intra** (same image) — prompt with an image's ``known`` instances and segment *all*
  instances of that class in the same image (prompts included, since foveate re-finds them).
  ``exemplar_image is None`` (exemplars and targets share the image).
* **inter** (cross image) — prompt with ``known`` instances from a *support* intra-pool image and
  discover instances of that class in a disjoint *novel* image. ``exemplar_image`` is the support
  image; this is the cross-image setting (enable ``Config.debias`` to correct DINOv3's positional
  bias).

Tensors from the dataset are converted to the ``HxWx3`` uint8 image and ``HxW`` boolean masks that
:func:`foveate.foveate_cascade` expects.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch

from data import DataConfig, InstanceDataset, InstanceSample


@dataclass
class EvalItem:
    image_id: str
    image: np.ndarray                  # (H, W, 3) uint8 — the target to discover in
    exemplar_masks: list[np.ndarray]   # prompts: (H, W) bool (in `exemplar_image` coords)
    gt_masks: np.ndarray               # (M, H, W) bool — instances to discover in `image`
    class_id: int
    exemplar_image: np.ndarray | None = None   # cross-image support; None => intra-image
    class_name: str | None = None              # human-readable name for class_id (text baselines)


def _image_to_numpy(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().float()
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    arr = arr[:3].permute(1, 2, 0).clamp(0, 1).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    return mask.detach().cpu().bool().numpy()


def build_datasets(cfg: DataConfig) -> tuple[InstanceDataset, InstanceDataset | None]:
    """Build the ``(intra, inter)`` dataset pair from a :class:`DataConfig`."""
    return InstanceDataset.build(cfg)


# ---------------------------------------------------------------------------
# intra-image: known prompts -> unknown GT, same image
# ---------------------------------------------------------------------------
def _intra_item(sample: InstanceSample, image_id: str, max_exemplars: int,
                class_names: dict[int, str] | None = None) -> EvalItem | None:
    prompts = sample.known_instances
    if not prompts:
        return None
    target_class = Counter(inst.class_id for inst in prompts).most_common(1)[0][0]
    exemplars = [p for p in prompts if p.class_id == target_class][:max_exemplars]
    # GT is *all* instances of the class, prompts included: foveate segments the whole class
    # region (it can't know which instances were handed to it as exemplars), so re-finding a
    # prompt must count as a true positive, not a false positive. Scoring only the held-out
    # `unknown` instances penalised every correctly-segmented prompt as an FP and capped AP at
    # ~n_unknown/n_total even with perfect masks.
    gt = [inst.mask for inst in sample.instances_of_class(target_class)]
    if not exemplars or not gt:
        return None
    return EvalItem(
        image_id=image_id,
        image=_image_to_numpy(sample.image),
        exemplar_masks=[_mask_to_numpy(p.mask) for p in exemplars],
        gt_masks=np.stack([_mask_to_numpy(m) for m in gt]),
        class_id=int(target_class),
        exemplar_image=None,
        class_name=(class_names or {}).get(int(target_class)),
    )


def iter_intra_items(dataset: InstanceDataset, *, max_exemplars: int = 3, limit: int | None = None):
    """Yield same-image :class:`EvalItem`s for samples that have a prompt and an unknown target."""
    n = 0
    class_names = dataset.class_names
    for idx in range(len(dataset)):
        item = _intra_item(dataset[idx], dataset.image_ids[idx], max_exemplars, class_names)
        if item is None:
            continue
        yield item
        n += 1
        if limit is not None and n >= limit:
            return


# ---------------------------------------------------------------------------
# inter-image: support (intra-pool) prompts -> all class instances in a novel image
# ---------------------------------------------------------------------------
@dataclass
class _Support:
    """Per-class exemplar support drawn from the intra (training) pool."""

    by_class: dict[int, list[tuple[np.ndarray, list[np.ndarray]]]] = field(default_factory=dict)

    def best(self, class_id: int, max_exemplars: int):
        """The support image with the most known masks of ``class_id`` → (image, exemplar_masks)."""
        candidates = self.by_class.get(class_id)
        if not candidates:
            return None
        image, masks = max(candidates, key=lambda t: len(t[1]))
        return image, masks[:max_exemplars]


def build_support_index(dataset: InstanceDataset) -> _Support:
    """Index the intra-pool's ``known`` instances by class for cross-image prompting."""
    support = _Support()
    for idx in range(len(dataset)):
        sample = dataset[idx]
        img = _image_to_numpy(sample.image)
        by_class: dict[int, list[np.ndarray]] = {}
        for inst in sample.known_instances:
            by_class.setdefault(inst.class_id, []).append(_mask_to_numpy(inst.mask))
        for cls, masks in by_class.items():
            support.by_class.setdefault(cls, []).append((img, masks))
    return support


def iter_inter_items(
    dataset: InstanceDataset,
    support: _Support,
    *,
    max_exemplars: int = 3,
    limit: int | None = None,
):
    """Yield cross-image :class:`EvalItem`s: support prompts vs all class instances in novel images."""
    n = 0
    class_names = dataset.class_names
    for idx in range(len(dataset)):
        sample = dataset[idx]
        counts = Counter(inst.class_id for inst in sample.instances)
        # Pick the most frequent class in this novel image that has support exemplars.
        target_class = next((c for c, _ in counts.most_common() if c in support.by_class), None)
        if target_class is None:
            continue
        chosen = support.best(target_class, max_exemplars)
        if chosen is None:
            continue
        support_img, exemplars = chosen
        gt = [_mask_to_numpy(inst.mask) for inst in sample.instances_of_class(target_class)]
        if not exemplars or not gt:
            continue
        yield EvalItem(
            image_id=dataset.image_ids[idx],
            image=_image_to_numpy(sample.image),
            exemplar_masks=exemplars,
            gt_masks=np.stack(gt),
            class_id=int(target_class),
            exemplar_image=support_img,
            class_name=class_names.get(int(target_class)),
        )
        n += 1
        if limit is not None and n >= limit:
            return
