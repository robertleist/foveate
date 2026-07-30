"""Shared types and the :class:`Backbone` protocol.

The core depends only on this small protocol, never on a concrete backbone, so users
can plug their own encoder (a different DINOv3 size, DINOv2, or a mock for tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Backbone(Protocol):
    """A frozen dense encoder: image(s) -> patch-feature grid (+ optional CLS).

    Implementations resize every input to a square ``image_size`` and patchify with
    ``patch_size``, so the patch grid is a fixed ``(image_size // patch_size)`` per side.
    """

    image_size: int
    patch_size: int

    def preprocess(self, images: np.ndarray | list[np.ndarray]) -> torch.Tensor:
        """Turn one image or a list of images into a ``(B, 3, S, S)`` pixel tensor."""
        ...

    def __call__(
        self, pixel_values: torch.Tensor, return_cls: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode ``pixel_values`` -> ``(B, C, Hp, Wp)`` (and ``(B, C)`` CLS if requested)."""
        ...


@dataclass
class Instance:
    """A discovered instance in original-image coordinates."""

    mask: np.ndarray                       # (H, W) uint8, ORIGINAL image coordinates
    box: tuple[int, int, int, int]         # (y0, y1, x0, x1) crop that isolated it
    depth: int                             # BFS level at which it converged
    score: float                           # re-identification score g(c) of the crop (paper Eq. 2)


# Backward-compatible alias: the cascade historically returned ``DiscoveredInstance``.
DiscoveredInstance = Instance


@dataclass
class Stats:
    """Cost + structure accounting for one :func:`cascade` call."""

    n_embeds: int = 0
    max_depth: int = 0
    leaves: int = 0                                        # instances emitted (before NMS)
    discarded: int = 0
    suppressed: int = 0
    merged: int = 0                # fragments unioned by merge_fragments                                    # emitted leaves dropped by final NMS
    level_sizes: list[int] = field(default_factory=list)   # frontier size per BFS level


# Backward-compatible alias.
CascadeStats = Stats
