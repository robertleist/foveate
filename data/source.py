"""DatasetSource — the single place a dataset format is described.

To add a new instance-segmentation dataset you implement **one**
:class:`DatasetSource` subclass and register it (see ``data/registry.py``).
Everything else — image/instance selection, PU splitting, flow/mask rendering,
caching, padding, and training — is shared and works automatically.

A source exposes three views of the same underlying data:

* :meth:`list_images`  — cheap metadata (id, size, per-instance category + area)
  used for selection and PU splitting. **No mask decoding.**
* :meth:`load_masks`   — decoded instance masks for an image, used both for
  rendering (all instances) and training (filtered to the chosen categories).
* :meth:`load_image`   — the pixel data.

``list_images`` MUST return images in a stable order (sorted by ``image_id``) so
that seeded selection is reproducible across machines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch


@dataclass
class AnnotationMeta:
    """Lightweight per-instance metadata (no mask) used for splitting."""

    ann_id: int
    category_id: int
    area: float = 0.0


@dataclass
class ImageMeta:
    """Per-image metadata used for selection and PU splitting."""

    image_id: str
    height: int
    width: int
    annotations: List[AnnotationMeta] = field(default_factory=list)

    @property
    def instance_count(self) -> int:
        return len(self.annotations)

    def annotations_by_class(self) -> Dict[int, List[AnnotationMeta]]:
        by_class: Dict[int, List[AnnotationMeta]] = {}
        for ann in self.annotations:
            by_class.setdefault(ann.category_id, []).append(ann)
        return by_class

    def filtered(self, category_ids: Sequence[int]) -> "ImageMeta":
        """Return a copy keeping only annotations in ``category_ids``."""
        keep = set(category_ids)
        return ImageMeta(
            image_id=self.image_id,
            height=self.height,
            width=self.width,
            annotations=[a for a in self.annotations if a.category_id in keep],
        )


@dataclass
class InstanceMask:
    """A decoded instance: metadata plus a ``[H, W]`` boolean mask."""

    ann_id: int
    category_id: int
    area: float
    mask: np.ndarray  # [H, W] bool


class DatasetSource(ABC):
    """Abstract description of one instance-segmentation dataset.

    Subclasses set the class attribute ``name`` (the registry key) and implement
    the abstract methods. Optionally override :meth:`ensure_available` to download
    or verify raw data, and :meth:`category_id_from_name` / :meth:`category_name`
    to support referring to categories by name in configs.
    """

    name: ClassVar[str] = ""

    # ------------------------------------------------------------------
    # Construction from a DataConfig (overridden per source as needed)
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "DatasetSource":
        """Build a source from a :class:`data.config.DataConfig`.

        Default implementation forwards ``root`` and any ``options``; sources
        with extra required fields should override this.
        """
        return cls(root=cfg.root, **(cfg.options or {}))

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    def ensure_available(self) -> None:
        """Verify (and optionally download) the raw data. Default: no-op."""
        return None

    # ------------------------------------------------------------------
    # Core data access — must be implemented
    # ------------------------------------------------------------------
    @abstractmethod
    def list_images(self) -> List[ImageMeta]:
        """All images with their per-instance metadata, sorted by ``image_id``.

        Includes annotations for *all* categories — category filtering happens
        downstream so the rendered cache is reusable across category subsets.
        """
        ...

    @abstractmethod
    def image_size(self, image_id: str) -> Tuple[int, int]:
        """Return ``(height, width)`` for an image without decoding masks."""
        ...

    @abstractmethod
    def load_image(self, image_id: str, grayscale: bool = False) -> torch.Tensor:
        """Load the image as a float32 tensor ``[C, H, W]`` in ``[0, 1]``."""
        ...

    @abstractmethod
    def load_masks(self, image_id: str) -> List[InstanceMask]:
        """Decode and return all instance masks for an image (all categories)."""
        ...

    # ------------------------------------------------------------------
    # Categories
    # ------------------------------------------------------------------
    @abstractmethod
    def category_ids(self) -> List[int]:
        """All category IDs defined by this dataset, in a stable order."""
        ...

    def category_id_from_name(self, name: str) -> int:
        """Map a human-readable category name to its ID. Override to support names."""
        raise ValueError(
            f"{type(self).__name__} does not support category names; "
            f"specify integer category IDs instead (got {name!r})."
        )

    def category_name(self, category_id: int) -> str:
        """Human-readable name for a category ID. Default: the ID as a string."""
        return str(category_id)

    def resolve_categories(
        self, categories: Sequence[Union[int, str]]
    ) -> List[int]:
        """Resolve a mix of names/ids (or empty = all) to an ordered ID list."""
        if not categories:
            return list(self.category_ids())
        resolved: List[int] = []
        for c in categories:
            if isinstance(c, str):
                resolved.append(self.category_id_from_name(c))
            else:
                resolved.append(int(c))
        return resolved
