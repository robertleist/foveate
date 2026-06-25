"""PanNukeSource — nuclei instance segmentation (proof that "by name" works).

PanNuke ships as NumPy arrays per fold:

    images.npy  [N, 256, 256, 3]  uint8
    masks.npy   [N, 256, 256, 6]  instance maps; channels 0-4 are the 5 nuclei
                                   classes (instance-indexed), channel 5 is
                                   background.

Point ``data.root`` at a directory containing ``images.npy`` and ``masks.npy``
(or override the filenames via ``options``). Multiple folds can be combined by
concatenating into a single pair of arrays, or by setting
``options.fold_dirs: [fold1, fold2, fold3]`` with each holding its own arrays.

This source is the worked example for adding a dataset: it implements the same
:class:`DatasetSource` interface as COCO and gets selection, PU splitting,
rendering, caching and training for free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data.registry import register_dataset
from data.source import (
    AnnotationMeta,
    DatasetSource,
    ImageMeta,
    InstanceMask,
)

# Category IDs are 1-indexed channel positions; names per the PanNuke paper.
_CLASS_NAMES = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}
_NAME_TO_ID = {v.lower(): k for k, v in _CLASS_NAMES.items()}
_INSTANCE_ID_STRIDE = 1_000_000  # ann_id = channel_id * stride + raw_instance_id


@register_dataset("pannuke")
class PanNukeSource(DatasetSource):
    """PanNuke nuclei dataset loaded from per-fold NumPy arrays."""

    def __init__(
        self,
        root: str,
        images_file: str = "images.npy",
        masks_file: str = "masks.npy",
        fold_dirs: Optional[List[str]] = None,
    ):
        if root is None:
            raise ValueError("PanNukeSource requires a `root` pointing at the PanNuke arrays.")
        self.root = Path(root)
        self.images_file = images_file
        self.masks_file = masks_file
        self.fold_dirs = fold_dirs  # optional list of subdirs to concatenate
        self._images = None   # lazily memory-mapped [N, H, W, 3]
        self._masks = None    # lazily memory-mapped [N, H, W, 6]
        self._meta_cache: Optional[List[ImageMeta]] = None

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "PanNukeSource":
        return cls(root=cfg.root, **(cfg.options or {}))

    def ensure_available(self) -> None:
        dirs = [self.root / d for d in self.fold_dirs] if self.fold_dirs else [self.root]
        for d in dirs:
            for fname in (self.images_file, self.masks_file):
                if not (d / fname).exists():
                    raise FileNotFoundError(
                        f"PanNuke array not found: {d / fname}\n"
                        "Download PanNuke folds (https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/) "
                        "and point data.root at a directory containing images.npy and masks.npy "
                        "(or set options.fold_dirs to a list of fold subdirectories)."
                    )

    # ------------------------------------------------------------------
    def _load_arrays(self):
        if self._images is not None:
            return
        if self.fold_dirs:
            imgs = [np.load(self.root / d / self.images_file, mmap_mode="r") for d in self.fold_dirs]
            masks = [np.load(self.root / d / self.masks_file, mmap_mode="r") for d in self.fold_dirs]
            # Concatenation forces these into memory; PanNuke folds are a few GB each.
            self._images = np.concatenate(imgs, axis=0)
            self._masks = np.concatenate(masks, axis=0)
        else:
            self._images = np.load(self.root / self.images_file, mmap_mode="r")
            self._masks = np.load(self.root / self.masks_file, mmap_mode="r")

    def _instances_for(self, idx: int) -> List[Tuple[int, int, np.ndarray]]:
        """Return ``(category_id, ann_id, bool_mask)`` for one image's instances."""
        self._load_arrays()
        mask = np.asarray(self._masks[idx])  # [H, W, 6]
        out = []
        for ch in range(5):                   # channels 0-4 are the 5 classes
            inst_map = mask[..., ch]
            for raw_id in np.unique(inst_map):
                if raw_id == 0:
                    continue
                category_id = ch + 1
                ann_id = category_id * _INSTANCE_ID_STRIDE + int(raw_id)
                out.append((category_id, ann_id, inst_map == raw_id))
        return out

    # ------------------------------------------------------------------
    def list_images(self) -> List[ImageMeta]:
        if self._meta_cache is not None:
            return self._meta_cache
        self._load_arrays()
        n, h, w = self._masks.shape[0], self._masks.shape[1], self._masks.shape[2]
        metas: List[ImageMeta] = []
        for idx in range(n):
            anns = [
                AnnotationMeta(ann_id=ann_id, category_id=cat, area=float(m.sum()))
                for cat, ann_id, m in self._instances_for(idx)
            ]
            metas.append(ImageMeta(image_id=str(idx), height=h, width=w, annotations=anns))
        self._meta_cache = metas
        return metas

    def image_size(self, image_id: str) -> Tuple[int, int]:
        self._load_arrays()
        return self._masks.shape[1], self._masks.shape[2]

    def load_image(self, image_id: str, grayscale: bool = False) -> torch.Tensor:
        self._load_arrays()
        img = np.asarray(self._images[int(image_id)], dtype=np.float32) / 255.0  # [H, W, 3]
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
        if grayscale:
            tensor = tensor.mean(dim=0, keepdim=True)  # [1, H, W]
        return tensor

    def load_masks(self, image_id: str) -> List[InstanceMask]:
        return [
            InstanceMask(ann_id=ann_id, category_id=cat, area=float(m.sum()), mask=m)
            for cat, ann_id, m in self._instances_for(int(image_id))
        ]

    # ------------------------------------------------------------------
    def category_ids(self) -> List[int]:
        return list(_CLASS_NAMES.keys())

    def category_id_from_name(self, name: str) -> int:
        key = name.lower()
        if key not in _NAME_TO_ID:
            raise ValueError(
                f"Unknown PanNuke category {name!r}. Available: {list(_CLASS_NAMES.values())}"
            )
        return _NAME_TO_ID[key]

    def category_name(self, category_id: int) -> str:
        return _CLASS_NAMES.get(category_id, str(category_id))
