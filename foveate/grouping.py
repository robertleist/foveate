"""Grouping — turn a *factorizable* extractor's foreground into instances.

This is the inside of :class:`foveate.extract.CompositeExtractor`, not a cascade slot of its own.
The cascade asks one question per crop (*which instances of the concept are here?*); an extractor
that answers it by first finding a class region and then cutting that region into objects is
**factorizable**, and this module is the second half of such a pair. A monolithic extractor (SAM,
NTT, the oracle) never touches it.

Keeping the two halves separately nameable is what preserves the Where-vs-grouping ablation:
``where=insid3 × group=cc`` and ``where=otsu × group=watershed`` are rows of one table, and the
error can still be attributed to a half — for the extractors where that decomposition is real.

``cc``
    Connected components. Never invents an internal boundary, so two touching objects stay one
    instance. The no-cut control arm.
``kmeans``, ``agglomerative``, ``watershed``
    Connected components **plus** an internal cut of a component that already fills the crop.

Why the cut is conditional on filling the crop
----------------------------------------------
A component that does *not* fill its crop still has framing to gain: the cascade will zoom onto it,
re-embed it at a larger effective resolution and ask again — and that answer is strictly better
informed than a cut made now, at the coarsest scale this component will ever be seen at. Cutting
early also destroys the zoom step itself: the cascade would only ever descend into halves, never
into the tightened whole object, so a single instance could no longer be framed. Once a component
fills the crop there is no framing left to gain, and an internal cut is the only remaining source of
new instances — which is exactly the moment the pre-two-slot cascade called "converged" and split.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label

from foveate import clustering, features as featlib, individuation, merge

_CONN8 = generate_binary_structure(2, 2)   # 8-connectivity: don't over-split single instances
_CONN4 = generate_binary_structure(2, 1)   # 4-connectivity: split diagonally-touching blobs


def _extract_structure(connectivity: int):
    """Connectivity structuring element for the connected-components step."""
    return _CONN4 if int(connectivity) == 4 else _CONN8


@runtime_checkable
class Grouping(Protocol):
    """Strategy interface for the grouping half of a factorizable extractor."""

    def group(self, feat, foreground: np.ndarray, box=None) -> list[np.ndarray]:
        """A ``(Hp, Wp)`` bool foreground → one bool grid per instance candidate (may be empty).

        ``box`` is the crop's ``(y0, y1, x0, x1)`` in original-image coordinates. Only the cutting
        groupings need it, to derive the crop a component would produce; ``None`` measures in patches
        (the grid *is* the box), which is what a standalone call wants.
        """
        ...


class ConnectedComponents:
    """``cc`` (≡ the legacy ``split_mode: none``) — components only, no internal cut.

    Also the base class every cutting grouping inherits its components from: the connected-component
    proposal is shared, only the internal boundary differs.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._struct = _extract_structure(cfg.extract_connectivity)

    def group(self, feat, foreground: np.ndarray, box=None) -> list[np.ndarray]:
        return self._components(foreground)

    def _components(self, foreground: np.ndarray) -> list[np.ndarray]:
        fg = np.asarray(foreground, dtype=bool)
        if not fg.any():
            return []
        labels, n = label(fg, structure=self._struct)
        return [labels == cid for cid in range(1, n + 1)]


class _CuttingGrouping(ConnectedComponents):
    """Components, then :meth:`divide` on any component that already fills the crop."""

    def group(self, feat, foreground: np.ndarray, box=None) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for comp in self._components(foreground):
            out.extend(self.divide(feat, comp) if self._fills_the_crop(comp, box) else [comp])
        return out

    def divide(self, feat, component: np.ndarray) -> list[np.ndarray]:
        """Cut one component into ≥ 2 sub-grids; ``[component]`` = it is a single instance."""
        raise NotImplementedError

    def _fills_the_crop(self, component: np.ndarray, box) -> bool:
        """Would the cascade gain anything by zooming onto this component alone?

        Exactly the cascade's own convergence test, run one step ahead: derive the crop this
        component would produce (:func:`foveate.features.child_box` — the same function the cascade
        uses, so the two can never drift) and compare its area to the crop we are on. At or above
        ``shrink_stop`` the box barely tightens, so re-embedding it buys no resolution and only an
        internal cut can produce new instances.
        """
        if not component.any():
            return False
        if box is None:                               # standalone call: the grid is the box
            hp, wp = component.shape
            box = (0, hp, 0, wp)
        cfg = self.cfg
        child = featlib.child_box(component, box, cfg.pad_frac, cfg.crop_dilate)
        area = lambda b: max(0, b[1] - b[0]) * max(0, b[3] - b[2])   # noqa: E731
        return area(child) / max(area(box), 1) >= cfg.shrink_stop


class KMeansGrouping(_CuttingGrouping):
    """``kmeans`` (default) — k=2 KMeans on the filled component's patch features.

    Deliberately *dumb*: it always proposes a 2-way cut rather than deciding an instance count, so
    the sub-crops are accepted or rejected by the Stop slot's similarity-peak guard (isolating a real
    object raises ``g``; halving a single object lowers it). That is what makes an appearance-blind
    cut safe — and it is also why ``kmeans`` is not a decomposition in its own right, unlike the
    monolithic extractors this slot was reshaped for.

    Features are L2-normalized, so euclidean KMeans ≈ spherical (cosine) clustering.
    """

    def divide(self, feat, component: np.ndarray) -> list[np.ndarray]:
        return split_kmeans(feat, component)


class AgglomerativeGrouping(_CuttingGrouping):
    """``agglomerative`` — connectivity-constrained agglomerative cut of a filled component.

    Clusters the component's patches with the same cosine-distance / spatial-graph agglomeration used
    for over-segmentation, cutting at ``cfg.cluster_tau``: the feature clustering *is* the cut, so a
    lower ``cluster_tau`` yields more, finer instances. A homogeneous component stays one cluster and
    is therefore reported as one instance.
    """

    def divide(self, feat, component: np.ndarray) -> list[np.ndarray]:
        labels = clustering.agglomerative_oversegment(feat, component, self.cfg.cluster_tau)
        return [labels == i for i in np.unique(labels) if i != 0]


class WatershedGrouping(_CuttingGrouping):
    """``watershed`` — marker-controlled, mask-constrained watershed for a seamless component.

    The geometric cut: it separates on the elevation ridge (waist / feature seam) *inside* a
    connected region, which is what components cannot do. May return a single basin (one instance).
    Retained for morphologies where a waist cut beats a feature cut — touching convex nuclei above all.
    """

    def divide(self, feat, component: np.ndarray) -> list[np.ndarray]:
        cfg = self.cfg
        labels = clustering.agglomerative_oversegment(feat, component, cfg.cluster_tau)
        indiv = individuation.individuate(
            feat, component, labels, mode=cfg.marker_mode, alpha=cfg.elevation_alpha,
            beta=cfg.elevation_beta, marker_min_distance=cfg.marker_min_distance,
            smooth_sigma=cfg.boundary_smooth_sigma,
        )
        merged = merge.merge_instances(
            feat, indiv.instances, indiv.feature_boundary,
            similarity_threshold=cfg.merge_similarity, boundary_threshold=cfg.merge_boundary,
        )
        return [merged == i for i in np.unique(merged) if i != 0]


def split_kmeans(feat, comp_grid) -> list[np.ndarray]:
    """k=2 KMeans on a component's patch features → two sub-grids (or one if < 2 patches)."""
    ys, xs = np.where(comp_grid)
    if ys.size < 2:
        return [np.asarray(comp_grid, dtype=bool)]            # single patch → one instance
    from sklearn.cluster import KMeans

    X = feat[torch.from_numpy(np.asarray(comp_grid)).to(feat.device)].detach().cpu().numpy()  # (M, D)
    lab = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(X)
    subs = []
    for c in (0, 1):
        g = np.zeros(comp_grid.shape, dtype=bool)
        g[ys[lab == c], xs[lab == c]] = True
        if g.any():
            subs.append(g)
    return subs if len(subs) == 2 else [np.asarray(comp_grid, dtype=bool)]


#: ``cfg.instance_extractor`` (alias ``grouping``; legacy ``split_mode``) → implementation.
#: ``"none"`` is the pre-slot spelling of ``"cc"`` and stays a supported synonym.
_GROUPINGS = {
    "cc": ConnectedComponents,
    "none": ConnectedComponents,
    "kmeans": KMeansGrouping,
    "agglomerative": AgglomerativeGrouping,
    "watershed": WatershedGrouping,
}


def build_grouping(cfg) -> Grouping:
    """Build the grouping named by ``cfg.instance_extractor``."""
    name = str(cfg.instance_extractor)
    try:
        return _GROUPINGS[name](cfg)
    except KeyError:
        raise ValueError(
            f"Unknown grouping {name!r}; expected one of {sorted(_GROUPINGS)}."
        ) from None
