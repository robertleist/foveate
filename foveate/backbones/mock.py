"""A weightless mock backbone for tests and demos.

It produces deterministic **colour-based** patch features: each patch's feature is a fixed
linear embedding of its mean RGB. Same-coloured regions therefore get near-identical features
(high cosine), different colours get low cosine — exactly the structure the gate, clustering
and re-identification stages rely on — so the full pipeline can be exercised on synthetic
colour-blob images without downloading DINOv3 weights.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class MockBackbone:
    """Colour-feature backbone. ``image_size`` / ``patch_size`` set the patch grid."""

    def __init__(self, image_size: int = 224, patch_size: int = 14, dim: int = 32, seed: int = 0):
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid = self.image_size // self.patch_size
        self.dim = int(dim)
        rng = np.random.default_rng(seed)
        # Fixed 3 -> dim projection; first 3 columns kept ~orthonormal so colour is preserved.
        proj = rng.standard_normal((3, self.dim)).astype(np.float32)
        self._proj = torch.from_numpy(proj)

    def preprocess(self, images: np.ndarray | list[np.ndarray]) -> torch.Tensor:
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        tensors = []
        for img in images:
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            t = torch.from_numpy(np.ascontiguousarray(arr)).float()
            if t.max() > 1.5:
                t = t / 255.0
            t = t.permute(2, 0, 1)[:3]
            t = F.interpolate(
                t.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
            tensors.append(t)
        return torch.cat(tensors, dim=0)                        # (B, 3, S, S)

    def __call__(
        self, pixel_values: torch.Tensor, return_cls: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Average-pool to the patch grid -> per-patch mean colour, then project to `dim`.
        pooled = F.adaptive_avg_pool2d(pixel_values, (self.grid, self.grid))  # (B, 3, Hp, Wp)
        b, _, hp, wp = pooled.shape
        colour = pooled.permute(0, 2, 3, 1).reshape(b, hp * wp, 3)            # (B, P, 3)
        feat = colour @ self._proj.to(colour.dtype)                          # (B, P, D)
        grid = feat.transpose(1, 2).reshape(b, self.dim, hp, wp)            # (B, D, Hp, Wp)
        if return_cls:
            cls = feat.mean(dim=1)                                           # (B, D)
            return grid, cls
        return grid
