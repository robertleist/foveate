from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import List

import torch


@dataclass
class InstanceSample:
    """One image with its ground-truth instance masks and a PU instance split.

    Mask-based (no flow fields): instances carry binary ``[H, W]`` masks and ``semantic_masks``
    is a per-class stack. The ``train`` partition supplies exemplar prompts; ``val`` /
    ``unlabelled`` are scored against the discovered instances.
    """

    image: torch.Tensor  # [channels, H, W] float32 in [0, 1]
    semantic_masks: torch.Tensor  # [num_classes, H, W] bool — per-class foreground union

    instances: List[GTInstance] = field(default_factory=list)  # all instances, indexed by position

    train_idx: List[int] = field(default_factory=list)  # indices into instances → labelled/exemplars
    val_idx: List[int] = field(default_factory=list)  # indices into instances → mAP_seen
    unlabelled_idx: List[int] = field(default_factory=list)  # indices into instances → mAP comparison

    # --- Partition accessors ---
    @property
    def train_instances(self) -> List[GTInstance]:
        return [self.instances[i] for i in self.train_idx]

    @property
    def val_instances(self) -> List[GTInstance]:
        return [self.instances[i] for i in self.val_idx]

    @property
    def unlabelled_instances(self) -> List[GTInstance]:
        return [self.instances[i] for i in self.unlabelled_idx]

    @property
    def known_instances(self) -> List[GTInstance]:
        return self.train_instances + self.val_instances

    # --- Objectness masks ---
    @cached_property
    def train_objectness(self) -> torch.Tensor:
        objectness = torch.zeros(self.image.shape[1:], dtype=torch.bool)
        for inst in self.train_instances:
            objectness[inst.mask] = True
        return objectness

    @cached_property
    def val_objectness(self) -> torch.Tensor:
        objectness = torch.zeros(self.image.shape[1:], dtype=torch.bool)
        for inst in self.val_instances:
            objectness[inst.mask] = True
        return objectness


@dataclass
class GTInstance:
    instance_id: int      # COCO annotation ID (or synthetic), for PU split lookup
    class_id: int         # 1-indexed remapped ground truth class
    mask: torch.Tensor    # [H, W] bool
