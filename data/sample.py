from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import List

import torch


@dataclass
class InstanceSample:
    """One image with its ground-truth instance masks and a known/unknown split.

    Mask-based (no flow fields): instances carry binary ``[H, W]`` masks and ``semantic_masks``
    is a per-class stack. The ``known`` instances are the exemplar prompts; the ``unknown``
    instances are the GT the method must discover (the intra-image evaluation targets).
    """

    image: torch.Tensor  # [channels, H, W] float32 in [0, 1]
    semantic_masks: torch.Tensor  # [num_classes, H, W] bool — per-class foreground union

    instances: List[GTInstance] = field(default_factory=list)  # all instances, indexed by position

    known_idx: List[int] = field(default_factory=list)    # indices into instances → exemplar prompts
    unknown_idx: List[int] = field(default_factory=list)  # indices into instances → GT to discover

    # --- Partition accessors ---
    @property
    def known_instances(self) -> List[GTInstance]:
        return [self.instances[i] for i in self.known_idx]

    @property
    def unknown_instances(self) -> List[GTInstance]:
        return [self.instances[i] for i in self.unknown_idx]

    def instances_of_class(self, class_id: int) -> List[GTInstance]:
        return [inst for inst in self.instances if inst.class_id == class_id]

    # --- Objectness masks ---
    @cached_property
    def known_objectness(self) -> torch.Tensor:
        objectness = torch.zeros(self.image.shape[1:], dtype=torch.bool)
        for inst in self.known_instances:
            objectness[inst.mask] = True
        return objectness

    @cached_property
    def unknown_objectness(self) -> torch.Tensor:
        objectness = torch.zeros(self.image.shape[1:], dtype=torch.bool)
        for inst in self.unknown_instances:
            objectness[inst.mask] = True
        return objectness


@dataclass
class GTInstance:
    instance_id: int      # COCO annotation ID (or synthetic), for split lookup
    class_id: int         # 1-indexed remapped ground truth class
    mask: torch.Tensor    # [H, W] bool
