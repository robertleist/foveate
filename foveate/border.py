"""Per-instance border refinement by *optimal* thresholding of the similarity map.

A converged crop's mask is only as good as the threshold that made it, and a single static
threshold is sub-par. Here we pick, per instance, the threshold that is optimal under a chosen
criterion — evaluated in closed form (one sorted cumulative-sum pass), not by iterating.

``mode="proto"`` — maximize ``cos( prototype(mask), exemplar_prototype )`` (the user's idea).
    Every reachable mask is a top-k prefix of patches sorted by similarity, so we evaluate all
    N thresholds at once via cumulative sums and take ``argmax``. Because the mask is then
    ``sims >= tau`` for the *similarity value* ``tau = s_(k*)`` (not the count k*), all equally
    similar patches are included automatically — so a homogeneous instance keeps its full extent
    rather than collapsing to one patch.

``mode="otsu"`` — classic Otsu split of the similarity histogram (maximize between-class
    variance). Robust for *heterogeneous* instances where the purity objective would shrink the
    mask to its core; cannot collapse.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_dilation, label

_CONN8 = np.ones((3, 3), dtype=bool)


def optimal_prototype_threshold(sims: torch.Tensor, feats: torch.Tensor):
    """Threshold maximizing ``cos(sum of selected feats, prototype)``.

    ``sims`` (N,) are ``⟨Fᵢ, p⟩``; ``feats`` (N, D) are the L2-normalized patch features.
    Returns ``(keep_bool (N,), tau, (sorted_sims, J_curve))``. The mask is ``sims >= tau`` for
    the *value* ``tau = s_(k*)``, so all patches as similar as the optimal cut are kept.
    """
    order = torch.argsort(sims, descending=True)
    s = sims[order]
    V = torch.cumsum(feats[order], dim=0)          # (N, D)  running Σ Fᵢ
    num = torch.cumsum(s, dim=0)                    # (N,)    running Σ sᵢ = ⟨ΣFᵢ, p⟩
    J = num / (V.norm(dim=1) + 1e-8)               # (N,)    cos(prototype(top-k), p)

    tau = float(s[int(J.argmax())])
    return sims >= tau, tau, (s.detach().cpu().numpy(), J.detach().cpu().numpy())


def otsu_threshold(sims: torch.Tensor, bins: int = 256) -> float:
    """Otsu threshold over the 1-D similarity values (maximize between-class variance)."""
    x = sims.detach().cpu().numpy()
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return lo
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / np.maximum(denom, 1e-12), 0.0)
    return float(centers[int(np.argmax(sigma_b))])


def refine(feat: torch.Tensor, comp_grid: np.ndarray, proto: torch.Tensor, *,
           mode: str = "proto", dilate: int = 1):
    """Re-threshold a component's border against the exemplar prototype.

    Selection is restricted to ``comp_grid`` dilated by ``dilate`` patches (so a stray high-sim
    patch elsewhere cannot join), and the result is reduced to the connected component(s)
    overlapping the original. Returns ``(refined_grid, tau, curve)`` (``curve`` is the
    ``(sorted_sims, J)`` pair for ``mode="proto"``, else ``None``).
    """
    if mode == "static":
        return comp_grid, None, None

    hp, wp, d = feat.shape
    cand = binary_dilation(comp_grid, iterations=dilate) if dilate > 0 else comp_grid.copy()
    flat = feat.reshape(hp * wp, d)
    sims = flat @ proto                            # (P,) cosine to prototype
    idx = np.flatnonzero(cand.reshape(-1))
    if idx.size == 0:
        return comp_grid, None, None
    idx_t = torch.from_numpy(idx).to(flat.device)
    cs = sims[idx_t]

    curve = None
    if mode == "proto":
        keep, tau, curve = optimal_prototype_threshold(cs, flat[idx_t])
        keep = keep.cpu().numpy()
    elif mode == "otsu":
        tau = otsu_threshold(cs)
        keep = (cs >= tau).cpu().numpy()
    else:
        raise ValueError(f"unknown border mode {mode!r}")

    sel = np.zeros(hp * wp, dtype=bool)
    sel[idx[keep]] = True
    refined = sel.reshape(hp, wp)

    # Keep only the connected component(s) overlapping the original component.
    lab, n = label(refined, structure=_CONN8)
    if n > 1:
        overlap = set(np.unique(lab[comp_grid & (lab > 0)])) - {0}
        if overlap:
            refined = np.isin(lab, list(overlap))
    return refined, tau, curve
