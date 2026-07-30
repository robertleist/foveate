"""Robust thresholding strategies for the two thresholds in the pipeline.

There are two places a similarity map is binarized:

* the **gate** — which patches belong to the exemplar class (a 2-D similarity grid → mask);
* the **border** — per-leaf refinement of an instance's extent (handled in :mod:`foveate.border`).

A single static cutoff is brittle: it broke when the prototype bank changed (cluster mode →
empty foreground). Making the gate threshold *adaptive* (Otsu / GMM-2 / percentile of the
per-crop similarity map) is the single highest-value robustness change (plan Sec. 5).
"""

from __future__ import annotations

import numpy as np


def otsu(values: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold over 1-D ``values`` (maximize between-class variance)."""
    x = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return lo
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    p = hist / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / np.maximum(denom, 1e-12), 0.0)
    return float(centers[int(np.argmax(sigma_b))])


def gmm2(values: np.ndarray, n_iter: int = 50) -> float:
    """Threshold at the crossover of a 2-component 1-D Gaussian mixture (EM)."""
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size < 2 or x.max() - x.min() < 1e-9:
        return float(x.min())
    lo, hi = float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))
    mu = np.array([lo, hi])
    var = np.array([x.var(), x.var()]) + 1e-6
    pi = np.array([0.5, 0.5])
    for _ in range(n_iter):
        g = pi / np.sqrt(2 * np.pi * var) * np.exp(-((x[:, None] - mu) ** 2) / (2 * var))
        g_sum = g.sum(axis=1, keepdims=True)
        r = g / np.clip(g_sum, 1e-12, None)                     # responsibilities (N, 2)
        nk = r.sum(axis=0) + 1e-9
        mu = (r * x[:, None]).sum(axis=0) / nk
        var = (r * (x[:, None] - mu) ** 2).sum(axis=0) / nk + 1e-6
        pi = nk / x.size
    hi_comp = int(np.argmax(mu))
    lo_comp = 1 - hi_comp
    # Search the boundary between the two means where the high component takes over.
    grid = np.linspace(mu[lo_comp], mu[hi_comp], 256)
    g = pi / np.sqrt(2 * np.pi * var) * np.exp(-((grid[:, None] - mu) ** 2) / (2 * var))
    cross = np.where(g[:, hi_comp] >= g[:, lo_comp])[0]
    return float(grid[cross[0]]) if cross.size else float(mu.mean())


def separability(values: np.ndarray, tau: float) -> float:
    """Otsu's separability ``eta = sigma_b^2 / sigma_total^2`` at cutoff ``tau`` — in ``[0, 1]``.

    Otsu *always* returns a cut, even for a single-mode histogram, where it slices the mode in half.
    That is the roadmap's over-zoom mechanism (§A1.2): deep in the cascade the object fills the crop,
    the similarity map goes unimodal, and the cut lands **inside** the object — so the padded bbox
    comes in tighter than the instance and the child crop truncates it.

    ``eta`` is the standard companion statistic to that cut: the fraction of the map's variance the
    split explains. A genuine fg/bg map separates cleanly (high ``eta``); a unimodal one does not
    (low ``eta``), which is the signal to stop trusting the cut. Cheaper and far more stable on the
    few-hundred values of a patch grid than a formal dip test.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    total = float(x.var())
    if x.size < 2 or total < 1e-12:
        return 0.0
    hi = x >= tau
    n_hi = int(hi.sum())
    if n_hi == 0 or n_hi == x.size:
        return 0.0
    w1 = n_hi / x.size
    between = w1 * (1.0 - w1) * (float(x[hi].mean()) - float(x[~hi].mean())) ** 2
    return float(min(1.0, between / total))


def hysteresis(values_grid: np.ndarray, hi: float, lo: float,
               connectivity: int = 8) -> np.ndarray:
    """Two-threshold foreground: seed at ``hi``, grow through ``>= lo`` connected to a seed.

    The Canny-style answer to a single brittle cut. Patches that are *confidently* the concept seed
    the mask; ambiguous ones join only when they are spatially attached to a seed, so a lenient
    ``lo`` recovers an object's dim rim (the part a single Otsu cut shaves off → over-zoom) without
    admitting free-floating background that happens to score moderately.

    Returns the ``hi`` mask unchanged when ``lo >= hi``.
    """
    from scipy.ndimage import generate_binary_structure, label

    grid = np.asarray(values_grid, dtype=np.float64)
    seeds = grid >= hi
    if lo >= hi or not seeds.any():
        return seeds
    weak = grid >= lo
    struct = generate_binary_structure(2, 2 if int(connectivity) == 8 else 1)
    labels, n = label(weak, structure=struct)
    if n == 0:
        return seeds
    keep = np.unique(labels[seeds])
    return np.isin(labels, keep[keep > 0])


def threshold(values: np.ndarray, method: str, *, static: float = 0.55,
              percentile: float = 80.0) -> float:
    """1-D similarity values → scalar cutoff ``tau`` by the chosen ``method``."""
    if method == "static":
        return float(static)
    if method == "otsu":
        return otsu(values)
    if method == "gmm2":
        return gmm2(values)
    if method == "percentile":
        return float(np.percentile(np.asarray(values).ravel(), percentile))
    raise ValueError(f"unknown threshold method {method!r}")


def foreground(sims_grid: np.ndarray, method: str, *, static: float = 0.55,
               percentile: float = 80.0) -> np.ndarray:
    """2-D gate-similarity grid → boolean foreground mask via :func:`threshold`."""
    tau = threshold(sims_grid, method, static=static, percentile=percentile)
    return np.asarray(sims_grid) >= tau
