from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.config import DataConfig
from data.sample import GTInstance, InstanceSample
from data.source import DatasetSource


class InstanceDataset(Dataset):
    """A mask-based instance-segmentation dataset, driven by a :class:`DatasetSource`.

    The recommended entry point is :meth:`from_config`, which takes a
    :class:`~data.config.DataConfig` (typically the ``data:`` block of an experiment config),
    selects images and computes the positive-unlabelled (PU) instance split in memory::

        cfg = DataConfig(name="coco", root="/data/coco", categories=["person"],
                         train_images=20, known_ratio=0.2, seed=42)
        dataset = InstanceDataset.from_config(cfg)

    Each sample carries the image, all ground-truth instance masks, a per-class semantic mask
    stack, and the PU split (``train`` instances are usable as exemplar prompts; ``val`` /
    ``unlabelled`` are held out for evaluation). Masks are decoded on access — there is no flow
    rendering and no on-disk artifact cache.
    """

    def __init__(
        self,
        source: DatasetSource,
        category_ids: List[int],
        image_ids: List[str],
        pu_split: Optional[Dict[str, Dict[str, List[int]]]] = None,
        grayscale: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            source: The backing :class:`DatasetSource`.
            category_ids: Ordered category IDs. ``category_ids[i] -> class i+1``.
            image_ids: Image IDs covered by this dataset.
            pu_split: Per-image partition. If None/empty, all instances are unlabelled.
            grayscale: Load images as ``[1, H, W]`` instead of ``[3, H, W]``.
            target_size: ``(H, W)`` to letterbox all samples to (None = no padding).
        """
        self.source = source
        self.category_ids = category_ids
        self.image_ids = list(image_ids)
        self.pu_split = pu_split or {}
        self.grayscale = grayscale
        self.target_size = target_size

        self.category_to_class: Dict[int, int] = {
            cat_id: idx + 1 for idx, cat_id in enumerate(category_ids)
        }
        self.num_classes = len(category_ids)

    @property
    def class_names(self) -> Dict[int, str]:
        """Remapped class id (``1..num_classes``) -> human-readable category name.

        Names come from the source (:meth:`DatasetSource.category_name`), so baselines that
        prompt with text (e.g. "person") can resolve the remapped ids back to dataset names.
        """
        return {
            class_id: self.source.category_name(cat_id)
            for cat_id, class_id in self.category_to_class.items()
        }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, cfg: DataConfig) -> Tuple["InstanceDataset", Optional["InstanceDataset"]]:
        """Build the ``(intra, inter)`` dataset pair from a :class:`DataConfig`.

        A single seeded shuffle of the qualifying image pool is sliced into:
          - the first ``train_images`` → the **intra** pool (same-image discovery: prompt with
            an image's ``known`` instances, discover its ``unknown`` instances);
          - the next ``interval_images`` → a *disjoint* **inter** pool of novel images (used for
            cross-image discovery: prompt with intra-pool exemplars, discover in these). ``interval_images == -1`` uses **all** remaining images (every qualifying image not in the train pool).

        Both pools use the same per-image ``known``/``unknown`` split (``known_ratio``). The two
        image pools are disjoint by construction. Returns ``(intra, inter)`` where ``inter`` is
        ``None`` when ``interval_images == 0``.
        """
        from data.registry import build_source
        from data.splitter import compute_pu_split, select_images

        source = build_source(cfg)
        source.ensure_available()

        category_ids = source.resolve_categories(cfg.categories)

        # Filter to images with enough annotated instances in the chosen categories.
        metas = [m.filtered(category_ids) for m in source.list_images()]
        metas = [m for m in metas if m.instance_count >= cfg.min_instances]
        if not metas:
            raise ValueError(
                f"No images with >= {cfg.min_instances} instances in categories "
                f"{[source.category_name(c) for c in category_ids]}."
            )

        ordered = select_images(metas, cfg.selection, None, cfg.seed)

        n_train = cfg.train_images if cfg.train_images is not None else cfg.max_images
        # interval_images == -1 is a sentinel: evaluate cross-image on ALL images not in the train
        # pool (every remaining qualifying image becomes the inter/novel eval set).
        all_remaining = cfg.interval_images == -1
        if all_remaining and n_train is None:
            raise ValueError(
                "interval_images=-1 (evaluate on all images not in the train set) requires "
                "train_images to be set explicitly."
            )
        if n_train is None:
            n_train = len(ordered) - cfg.interval_images
        n_inter = (len(ordered) - n_train) if all_remaining else (cfg.interval_images or 0)

        if n_train <= 0:
            raise ValueError(f"train_images resolved to {n_train}; must be positive.")
        if n_train + n_inter > len(ordered):
            raise ValueError(
                f"Requested train_images={n_train} + interval_images={n_inter} "
                f"= {n_train + n_inter}, but only {len(ordered)} qualifying images "
                f"are available (categories={[source.category_name(c) for c in category_ids]}, "
                f"min_instances={cfg.min_instances})."
            )

        intra_pool = ordered[:n_train]
        inter_pool = ordered[n_train:n_train + n_inter]

        common = dict(
            source=source,
            category_ids=category_ids,
            grayscale=cfg.image_channels == 1,
            target_size=cfg.target_size,
        )

        def make(pool):
            return cls(
                image_ids=[m.image_id for m in pool],
                pu_split=compute_pu_split(
                    pool,
                    known_ratio=cfg.known_ratio,
                    stratify_by_class=cfg.stratify_by_class,
                    seed=cfg.seed,
                ),
                **common,
            )

        intra_dataset = make(intra_pool)
        inter_dataset = make(inter_pool) if n_inter > 0 else None
        return intra_dataset, inter_dataset

    @classmethod
    def from_config(cls, cfg: DataConfig) -> "InstanceDataset":
        """Build just the intra-image dataset from a :class:`DataConfig`."""
        return cls.build(cfg)[0]

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> "InstanceDataset":
        """Load a self-contained dataset YAML (``dataset:`` + ``image_ids`` + optional ``pu_split``)."""
        import yaml

        from data.registry import get_source_class

        with open(yaml_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)

        ds = doc["dataset"]
        pre = doc.get("preprocessing", {})

        source = get_source_class(ds.get("type", "coco"))(
            root=ds["root"],
            annotation_file=ds.get("annotation_file"),
            images_dir=ds.get("images_dir"),
        )

        if target_size is None:
            raw_size = pre.get("target_size")
            target_size = tuple(raw_size) if raw_size is not None else None

        return cls(
            source=source,
            category_ids=ds["category_ids"],
            image_ids=doc["image_ids"],
            pu_split=doc.get("pu_split"),
            grayscale=pre.get("image_channels", 3) == 1,
            target_size=target_size,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> InstanceSample:
        image_id = self.image_ids[idx]

        image = self.source.load_image(image_id, grayscale=self.grayscale)
        instances = self._load_instances(image_id)
        semantic_masks = self._build_semantic_masks(image.shape[1:], instances)
        known_idx, unknown_idx = self._partition_instances(image_id, instances)

        if self.target_size is not None:
            image, semantic_masks, instances = self._resize_and_pad(
                image, semantic_masks, instances
            )

        return InstanceSample(
            image=image,
            semantic_masks=semantic_masks,
            instances=instances,
            known_idx=known_idx,
            unknown_idx=unknown_idx,
        )

    # ------------------------------------------------------------------
    # Instances & masks
    # ------------------------------------------------------------------
    def _load_instances(self, image_id: str) -> List[GTInstance]:
        """Decode instances in the chosen categories, remapped to contiguous class IDs."""
        out: List[GTInstance] = []
        for im in self.source.load_masks(image_id):
            class_id = self.category_to_class.get(im.category_id)
            if class_id is None:
                continue
            mask = im.mask if torch.is_tensor(im.mask) else torch.from_numpy(im.mask)
            out.append(GTInstance(instance_id=im.ann_id, class_id=class_id, mask=mask.bool()))
        return out

    def _build_semantic_masks(
        self, image_hw: Tuple[int, int], instances: List[GTInstance]
    ) -> torch.Tensor:
        """[num_classes, H, W] bool — per-class union of the instance masks."""
        h, w = int(image_hw[0]), int(image_hw[1])
        out = torch.zeros((self.num_classes, h, w), dtype=torch.bool)
        for inst in instances:
            out[inst.class_id - 1] |= inst.mask
        return out

    def _partition_instances(
        self,
        image_id: str,
        instances: List[GTInstance],
    ) -> Tuple[List[int], List[int]]:
        """Resolve list indices for known / unknown from the split.

        Any instance not named in the split (or all of them, when no split exists) defaults to
        ``unknown`` — so it counts as a discovery target rather than a free prompt.
        """
        split = self.pu_split.get(str(image_id))
        if split is None:
            return [], list(range(len(instances)))

        id_to_idx = {inst.instance_id: i for i, inst in enumerate(instances)}

        def resolve(ann_ids: List[int]) -> List[int]:
            return [id_to_idx[aid] for aid in ann_ids if aid in id_to_idx]

        known_idx = resolve(split.get("known", []))
        unknown_idx = resolve(split.get("unknown", []))

        assigned = set(known_idx) | set(unknown_idx)
        for i in range(len(instances)):
            if i not in assigned:
                unknown_idx.append(i)

        return known_idx, unknown_idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resize_and_pad(
        self,
        image: torch.Tensor,
        semantic_masks: torch.Tensor,
        instances: List[GTInstance],
    ) -> tuple:
        """Letterbox: scale uniformly to fit target_size, then pad bottom/right with zeros."""
        target_h, target_w = self.target_size
        _, h, w = image.shape

        scale = min(target_h / h, target_w / w)

        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)

            image = F.interpolate(
                image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
            ).squeeze(0)

            semantic_masks = F.interpolate(
                semantic_masks.unsqueeze(0).float(), size=(new_h, new_w), mode="nearest"
            ).squeeze(0).bool()

            instances = [
                GTInstance(
                    instance_id=inst.instance_id,
                    class_id=inst.class_id,
                    mask=F.interpolate(
                        inst.mask.float().unsqueeze(0).unsqueeze(0),
                        size=(new_h, new_w), mode="nearest"
                    ).squeeze().bool(),
                )
                for inst in instances
            ]
        else:
            new_h, new_w = h, w

        pad = (0, target_w - new_w, 0, target_h - new_h)  # (left, right, top, bottom)
        image = F.pad(image, pad, value=0.0)
        semantic_masks = F.pad(semantic_masks.float(), pad, value=0.0).bool()
        instances = [
            GTInstance(
                instance_id=inst.instance_id,
                class_id=inst.class_id,
                mask=F.pad(inst.mask.float(), pad, value=0.0).bool(),
            )
            for inst in instances
        ]

        return image, semantic_masks, instances
