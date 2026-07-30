"""COCOSource — a DatasetSource backed by a COCO-format annotation file."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from data.registry import register_dataset
from data.source import (
    AnnotationMeta,
    DatasetSource,
    ImageMeta,
    InstanceMask,
)

# COCO 2017 conventions, used when a config omits these.
_DEFAULT_ANNOTATION_FILE = "annotations/instances_train2017.json"
_DEFAULT_IMAGES_DIR = "train2017"


@register_dataset("coco")
class COCOSource(DatasetSource):
    """COCO-format instance segmentation dataset."""

    def __init__(
        self,
        root: str,
        annotation_file: Optional[str] = None,
        images_dir: Optional[str] = None,
    ):
        if root is None:
            raise ValueError("COCOSource requires a `root` pointing at the dataset directory.")
        self.root = Path(root)
        self.annotation_file = self.root / (annotation_file or _DEFAULT_ANNOTATION_FILE)
        self.images_dir = self.root / (images_dir or _DEFAULT_IMAGES_DIR)
        self._coco = None  # lazy: only build the COCO index when first needed

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "COCOSource":
        return cls(
            root=cfg.root,
            annotation_file=cfg.annotation_file,
            images_dir=cfg.images_dir,
            **(cfg.options or {}),
        )

    def ensure_available(self) -> None:
        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"COCO annotation file not found: {self.annotation_file}\n"
                "Set data.root / data.annotation_file to a downloaded COCO dataset."
            )
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"COCO images directory not found: {self.images_dir}\n"
                "Set data.root / data.images_dir to the directory containing the images."
            )

    # ------------------------------------------------------------------
    @property
    def coco(self):
        if self._coco is None:
            from pycocotools.coco import COCO

            self._coco = COCO(str(self.annotation_file))
            # LVIS annotations carry no ``iscrowd`` field, and pycocotools' ``getAnnIds(...,
            # iscrowd=False)`` indexes it unconditionally -> KeyError on an otherwise valid file.
            # Defaulting it to 0 (LVIS has no crowd regions to begin with) makes every pycocotools
            # path work on LVIS without a separate reader.
            for ann in self._coco.anns.values():
                ann.setdefault("iscrowd", 0)
        return self._coco

    def list_images(self) -> List[ImageMeta]:
        coco = self.coco
        metas: List[ImageMeta] = []
        for img_id in sorted(coco.getImgIds()):
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            img_meta = coco.loadImgs(img_id)[0]
            metas.append(ImageMeta(
                image_id=str(img_id),
                height=img_meta["height"],
                width=img_meta["width"],
                annotations=[
                    AnnotationMeta(
                        ann_id=a["id"],
                        category_id=a["category_id"],
                        area=float(a.get("area", 0.0)),
                    )
                    for a in anns
                ],
            ))
        return metas

    def image_size(self, image_id: str) -> Tuple[int, int]:
        info = self.coco.loadImgs(int(image_id))[0]
        return info["height"], info["width"]

    @staticmethod
    def _file_name(info: dict) -> str:
        """Image filename from a COCO *or* LVIS image record.

        LVIS v1 drops ``file_name`` and carries only ``coco_url`` (the images are COCO's), so a
        plain COCO reader cannot open an LVIS annotation file at all. Both are the same directory of
        jpegs, so taking the URL's basename is enough to read LVIS through this source unchanged.
        """
        name = info.get("file_name")
        if name:
            return name
        url = info.get("coco_url") or info.get("flickr_url")
        if not url:
            raise KeyError(
                f"image {info.get('id')} has neither 'file_name' nor 'coco_url'; cannot locate it."
            )
        return url.rsplit("/", 1)[-1]

    def load_image(self, image_id: str, grayscale: bool = False) -> torch.Tensor:
        info = self.coco.loadImgs(int(image_id))[0]
        img = Image.open(self.images_dir / self._file_name(info))
        if grayscale:
            img = img.convert("L")
            tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
            return tensor.unsqueeze(0)  # [1, H, W]
        img = img.convert("RGB")
        tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        return tensor.permute(2, 0, 1)  # [3, H, W]

    def load_masks(self, image_id: str) -> List[InstanceMask]:
        info = self.coco.loadImgs(int(image_id))[0]
        h, w = info["height"], info["width"]
        ann_ids = self.coco.getAnnIds(imgIds=int(image_id), iscrowd=False)
        out: List[InstanceMask] = []
        for ann in self.coco.loadAnns(ann_ids):
            out.append(InstanceMask(
                ann_id=ann["id"],
                category_id=ann["category_id"],
                area=float(ann.get("area", 0.0)),
                mask=self._decode_mask(ann, h, w),
            ))
        return out

    # ------------------------------------------------------------------
    def category_ids(self) -> List[int]:
        return sorted(self.coco.getCatIds())

    def category_id_from_name(self, name: str) -> int:
        ids = self.coco.getCatIds(catNms=[name])
        if not ids:
            available = [c["name"] for c in self.coco.loadCats(self.coco.getCatIds())]
            raise ValueError(f"Unknown COCO category {name!r}. Available: {available}")
        return ids[0]

    def category_name(self, category_id: int) -> str:
        cats = self.coco.loadCats([category_id])
        return cats[0]["name"] if cats else str(category_id)

    # ------------------------------------------------------------------
    @staticmethod
    def _decode_mask(ann: dict, h: int, w: int) -> np.ndarray:
        import pycocotools.mask as mask_util

        seg = ann.get("segmentation")
        if not seg:
            # Detection-only datasets (RF20-VL, the CD-FSOD sets, most Roboflow exports) carry an
            # empty ``segmentation``. Fall back to the bounding box as a rectangular mask so a
            # box-annotated dataset can be read through the same source. Mask metrics are then
            # meaningless by construction -- report BOX AP on these datasets, not mask AP.
            x, y, bw, bh = ann["bbox"]
            mask = np.zeros((h, w), dtype=bool)
            y0, y1 = max(0, int(round(y))), min(h, int(round(y + bh)))
            x0, x1 = max(0, int(round(x))), min(w, int(round(x + bw)))
            mask[y0:y1, x0:x1] = True
            return mask
        if isinstance(seg, list):
            rle = mask_util.frPyObjects(seg, h, w)
            rle = mask_util.merge(rle)
        elif isinstance(seg, dict):
            rle = seg
        else:
            raise ValueError(
                f"Unexpected segmentation format for annotation {ann['id']}: {type(seg)}"
            )
        return mask_util.decode(rle).astype(bool)
