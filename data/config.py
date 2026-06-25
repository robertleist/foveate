"""DataConfig — the self-contained dataset description embedded in an experiment run.

Everything needed to build a dataset (which dataset, which categories, how many images, the
PU ratios, the seed) lives here, and :meth:`InstanceDataset.from_config` turns it into a
ready-to-use mask-based dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class DataConfig:
    # --- which dataset ---
    name: str                                   # registry key, e.g. "coco", "pannuke"
    root: Optional[str] = None                  # raw-data root (None for auto-download sources)
    annotation_file: Optional[str] = None       # source-specific (COCO JSON path, relative to root)
    images_dir: Optional[str] = None            # source-specific (image subdir, relative to root)
    options: Dict[str, Any] = field(default_factory=dict)  # extra source-specific kwargs

    # --- categories & filtering ---
    categories: List[Union[int, str]] = field(default_factory=list)  # ids or names; empty = all
    min_instances: int = 1                      # drop images with fewer instances in the chosen categories

    # --- image selection (disjoint train / inter-validation pools) ---
    # One seeded shuffle of the qualifying pool is sliced into a training set and
    # a *disjoint* inter-validation set, so novel images never leak into training.
    train_images: Optional[int] = None          # images used for training (None = all remaining)
    interval_images: int = 0                     # novel images held out for inter-validation
    max_images: Optional[int] = None             # deprecated alias for train_images
    selection: str = "random"                    # random | first | densest

    # --- PU instance split ---
    known_ratio: float = 0.2                     # fraction of instances labelled (train AND inter-val images)
    intra_val_ratio: float = 0.3                 # fraction of known instances in TRAIN images held out (no grad) -> intra-val
    stratify_by_class: bool = True
    val_only: bool = False                       # internal: used for the inter-validation pool

    seed: int = 42                               # drives both selection and the PU split

    # --- preprocessing ---
    target_size: Optional[Tuple[int, int]] = None
    image_channels: int = 3                     # 1 -> grayscale

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        d = dict(d)
        ts = d.get("target_size")
        if ts is not None:
            d["target_size"] = tuple(ts)
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in d.items() if k not in known}
        kept = {k: v for k, v in d.items() if k in known}
        if extra:
            # Stash unknown keys as source-specific options rather than failing.
            kept.setdefault("options", {})
            kept["options"] = {**kept["options"], **extra}
        return cls(**kept)
