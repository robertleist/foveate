"""SyntheticSource — deterministic coloured-square images, no downloads.

Each instance is a solid-colour square; instances sharing a category share a base colour, so the
colour-based :class:`~foveate.backbones.mock.MockBackbone` groups same-category instances exactly
as DINOv3 groups same-concept patches. This makes the full experiment harness (discovery →
metrics → MLflow) runnable offline, in tests and demos.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from data.registry import register_dataset
from data.source import AnnotationMeta, DatasetSource, ImageMeta, InstanceMask

# Distinct, well-separated base colours per category.
_PALETTE = [
    (220, 40, 40), (40, 90, 220), (40, 200, 80), (230, 200, 40),
    (180, 40, 200), (40, 210, 210),
]


@register_dataset("synthetic")
class SyntheticSource(DatasetSource):
    """In-memory coloured-square instance dataset."""

    def __init__(
        self,
        root: Optional[str] = None,
        n_images: int = 8,
        image_size: int = 128,
        n_instances: int = 4,
        n_categories: int = 2,
        seed: int = 0,
    ):
        self.root = root
        self.n_images = int(n_images)
        self.img_size = int(image_size)
        self.n_instances = int(n_instances)
        self.n_categories = min(int(n_categories), len(_PALETTE))
        self.seed = int(seed)
        self._cache: dict[int, tuple[np.ndarray, List[InstanceMask]]] = {}

    @classmethod
    def from_config(cls, cfg) -> "SyntheticSource":
        return cls(root=cfg.root, seed=cfg.seed, **(cfg.options or {}))

    # ------------------------------------------------------------------
    def _render(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]
        rng = np.random.default_rng(self.seed * 10_000 + idx)
        s = self.img_size
        img = np.zeros((s, s, 3), dtype=np.uint8)
        instances: List[InstanceMask] = []
        for j in range(self.n_instances):
            cat = int(rng.integers(0, self.n_categories)) + 1
            colour = np.array(_PALETTE[cat - 1], dtype=np.uint8)
            side = int(rng.integers(s // 10, s // 5))
            y0 = int(rng.integers(0, s - side))
            x0 = int(rng.integers(0, s - side))
            jitter = rng.integers(-15, 16, size=3)
            img[y0:y0 + side, x0:x0 + side] = np.clip(colour.astype(int) + jitter, 0, 255)
            mask = np.zeros((s, s), dtype=bool)
            mask[y0:y0 + side, x0:x0 + side] = True
            instances.append(InstanceMask(
                ann_id=idx * 1000 + j, category_id=cat, area=float(mask.sum()), mask=mask,
            ))
        self._cache[idx] = (img, instances)
        return self._cache[idx]

    # ------------------------------------------------------------------
    def list_images(self) -> List[ImageMeta]:
        metas = []
        for idx in range(self.n_images):
            _, instances = self._render(idx)
            metas.append(ImageMeta(
                image_id=str(idx),
                height=self.img_size,
                width=self.img_size,
                annotations=[
                    AnnotationMeta(ann_id=im.ann_id, category_id=im.category_id, area=im.area)
                    for im in instances
                ],
            ))
        return metas

    def image_size(self, image_id: str) -> Tuple[int, int]:
        return self.img_size, self.img_size

    def load_image(self, image_id: str, grayscale: bool = False) -> torch.Tensor:
        img, _ = self._render(int(image_id))
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        if grayscale:
            t = t.mean(dim=0, keepdim=True)
        return t

    def load_masks(self, image_id: str) -> List[InstanceMask]:
        _, instances = self._render(int(image_id))
        return instances

    # ------------------------------------------------------------------
    def category_ids(self) -> List[int]:
        return list(range(1, self.n_categories + 1))

    def category_name(self, category_id: int) -> str:
        return f"colour{category_id}"
