"""Positional debiasing for cross-image matching (INSID3 Sec. 3.1).

DINOv3 patch features carry a position-dependent component: features at the *same grid
location* spuriously match across images. When the exemplar bank comes from a different crop
than the query (the cross-image / per-node setting), this inflates the gate similarity.

The fix: estimate the low-rank positional subspace ``B`` by passing pure-noise image(s) through
the encoder (any structure in those features is positional, not semantic), then project it out
of features **used for matching** — leaving raw features for intra-image grouping where the
positional signal helps.

``B`` depends only on ``(model, image_size)``, so it is cached.
"""

from __future__ import annotations

import numpy as np
import torch

from foveate import features as featlib

_BASIS_CACHE: dict[tuple, torch.Tensor] = {}


def estimate_positional_basis(
    backbone, *, subspace_dim: int = 8, n_noise: int = 1, seed: int = 0,
    standardize: bool = True,
) -> torch.Tensor:
    """Top ``subspace_dim`` right-singular vectors of noise-image patch features → ``B`` (D, s)."""
    key = (id(backbone), getattr(backbone, "image_size", None),
           getattr(backbone, "patch_size", None), subspace_dim, n_noise, seed, standardize)
    if key in _BASIS_CACHE:
        return _BASIS_CACHE[key]

    rng = np.random.default_rng(seed)
    size = int(getattr(backbone, "image_size", 224))
    noise = [rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8) for _ in range(n_noise)]
    embedded = featlib.embed_batch(backbone, noise, chunk=max(1, n_noise), standardize=standardize)

    feats = torch.cat([f.reshape(-1, f.shape[-1]) for f, _ in embedded], dim=0)  # (P, D)
    feats = feats - feats.mean(dim=0, keepdim=True)
    # Right singular vectors V; columns are the principal feature directions of pure noise.
    _, _, vh = torch.linalg.svd(feats, full_matrices=False)
    B = vh[:subspace_dim].T.contiguous()                                        # (D, s)
    _BASIS_CACHE[key] = B
    return B


def project_out(feats: torch.Tensor, B: torch.Tensor | None) -> torch.Tensor:
    """Return ``feats @ (I - B Bᵀ)`` then re-L2-normalize. ``B=None`` is a no-op."""
    if B is None:
        return feats
    B = B.to(feats.device, feats.dtype)
    debiased = feats - (feats @ B) @ B.T
    return featlib.l2_normalize(debiased, dim=-1)
