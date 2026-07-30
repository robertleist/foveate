"""EXTRACT — turn a crop's concept foreground into instance candidates (slot 2 of 3).

The cascade is three swappable slots (see :mod:`foveate.cascade`):

* **Where** (:mod:`foveate.foreground`) — which patches of this crop are the concept?
* **Extract** (*this module*) — which *instances* does that foreground hold?
* **Stop** (:mod:`foveate.stop`) — descend, emit or reject?

Extract answers its question with two operations, and the split between them is the whole reason
this is one slot rather than two:

``components(fg)``
    Separate the foreground into instance candidates. Cheap, geometric, and the source of the
    cascade's *proposals*: ≥ 2 components ⇒ several instances to zoom into separately; 1 component
    that does not fill the crop ⇒ keep zooming; 1 component that fills it ⇒ converged.

``split(feat, component)``
    Force **one converged component** into ≥ 2 sub-candidates. Connected components cannot help
    here by construction (the clump *is* one component), so a different signal must draw the
    boundary — features (k-means, agglomerative) or geometry (watershed).

Both operations answer "which instances are in this region?", differing only in whether an internal
boundary has to be invented; that is why one named extractor supplies both. The default
:class:`KMeansExtractor` is deliberately *dumb* — it always proposes a 2-way cut and lets the
**Stop** slot's re-identification lookahead accept or reject it (the splitter proposes, ``g``
disposes), which is what makes an appearance-blind splitter safe.

Selected by ``cfg.instance_extractor`` (legacy key: ``split_mode``). ``"cc"`` (≡ ``"none"``) runs
components only and accepts a converged crop whole — the no-split ablation.

.. note::
   A future image-based extractor (SAM automatic masks / point prompts on the converged crop,
   roadmap T2) needs the crop *pixels*, which these signatures do not carry. The cascade has them
   at both call sites (``image[y0:y1, x0:x1]``), so that is a one-argument change when the first
   such extractor lands — not worth a dead parameter until then.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label

from foveate import clustering, individuation, merge

_CONN8 = generate_binary_structure(2, 2)   # 8-connectivity: don't over-split single instances
_CONN4 = generate_binary_structure(2, 1)   # 4-connectivity: split diagonally-touching blobs


def _extract_structure(connectivity: int):
    """Connectivity structuring element for the connected-components step."""
    return _CONN4 if int(connectivity) == 4 else _CONN8


def component_label_map(comps: list[np.ndarray], shape) -> np.ndarray:
    """Fold component grids back into one ``(Hp, Wp)`` int label image (``0`` = background).

    Only the tracing/visualization layer wants the label *image* — the cascade itself works on the
    component list — so the cascade builds this lazily, when an observer is attached.
    """
    out = np.zeros(shape, dtype=int)
    for i, c in enumerate(comps, start=1):
        out[c] = i
    return out


@runtime_checkable
class InstanceExtractor(Protocol):
    """Strategy interface for the Extract stage.

    ``can_split`` advertises whether :meth:`split` can ever return ≥ 2 sub-candidates. The cascade
    reads it to decide whether a converged crop is accepted whole (no splitter) or split-then-
    confirmed, and whether the marginal-peak retry is even attempted — so a no-split extractor
    short-circuits instead of paying a forward for a split that cannot happen.
    """

    can_split: bool

    def components(self, foreground: np.ndarray) -> list[np.ndarray]:
        """Instance candidates in a ``(Hp, Wp)`` bool foreground → list of bool grids (may be empty)."""
        ...

    def split(self, feat, component: np.ndarray) -> list[np.ndarray]:
        """Force one converged ``component`` into ≥ 2 sub-grids; ``[component]`` = unsplittable."""
        ...


class ConnectedComponentsExtractor:
    """``cc`` (≡ ``none``) — connected components only; a converged crop is accepted whole.

    The no-split ablation, and the base class every splitter inherits its ``components`` from:
    connected-components proposals are shared by all of them, only the internal cut differs.
    """

    can_split = False

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._struct = _extract_structure(cfg.extract_connectivity)

    def components(self, foreground: np.ndarray) -> list[np.ndarray]:
        fg = np.asarray(foreground, dtype=bool)
        if not fg.any():
            return []
        labels, n = label(fg, structure=self._struct)
        return [labels == cid for cid in range(1, n + 1)]

    def split(self, feat, component: np.ndarray) -> list[np.ndarray]:
        """No internal boundary is ever drawn → always "unsplittable"."""
        return [np.asarray(component, dtype=bool)]


class KMeansExtractor(ConnectedComponentsExtractor):
    """``kmeans`` (default) — k=2 KMeans on the component's foreground patch features.

    Unlike watershed this *always* yields a 2-way partition when the component has ≥ 2 patches, so
    "always try to split" is guaranteed and the Stop slot's re-id survivor rule does the accepting
    and rejecting. Features are L2-normalized, so euclidean KMeans ≈ spherical (cosine) clustering.
    """

    can_split = True

    def split(self, feat, component: np.ndarray) -> list[np.ndarray]:
        return _split_kmeans(feat, component)


class AgglomerativeExtractor(ConnectedComponentsExtractor):
    """``agglomerative`` — connectivity-constrained agglomerative split of the clump.

    Clusters the clump's foreground patches with the same cosine-distance / spatial-graph
    agglomeration used for over-segmentation, cutting at ``cfg.cluster_tau``: the feature clustering
    *is* the split, so a lower ``cluster_tau`` yields more, finer sub-crops. A homogeneous clump
    stays one cluster (→ unsplittable, the caller emits the parent).
    """

    can_split = True

    def split(self, feat, component: np.ndarray) -> list[np.ndarray]:
        labels = clustering.agglomerative_oversegment(feat, component, self.cfg.cluster_tau)
        return [labels == i for i in np.unique(labels) if i != 0]


class WatershedExtractor(ConnectedComponentsExtractor):
    """``watershed`` — marker-controlled, mask-constrained watershed for a seamless clump.

    The geometric splitter: it cuts on the elevation ridge (waist / feature seam) *inside* a
    connected region, which is exactly what components cannot do. Can return a single basin
    ("unsplittable"). Retained for morphologies where a waist cut beats a feature cut — touching
    convex nuclei above all.
    """

    can_split = True

    def split(self, feat, component: np.ndarray) -> list[np.ndarray]:
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


def _split_kmeans(feat, comp_grid) -> list[np.ndarray]:
    """k=2 KMeans on the component's foreground patch features → two sub-masks (or one if < 2 patches)."""
    ys, xs = np.where(comp_grid)
    if ys.size < 2:
        return [np.asarray(comp_grid, dtype=bool)]            # single patch → unsplittable
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


#: ``cfg.instance_extractor`` → implementation. ``"none"`` is the pre-slot spelling of ``"cc"``
#: (``split_mode: none`` in every existing config) and stays a supported synonym.
_EXTRACTORS = {
    "cc": ConnectedComponentsExtractor,
    "none": ConnectedComponentsExtractor,
    "kmeans": KMeansExtractor,
    "agglomerative": AgglomerativeExtractor,
    "watershed": WatershedExtractor,
}


def build_instance_extractor(cfg) -> InstanceExtractor:
    """Build the Extract strategy named by ``cfg.instance_extractor``."""
    name = str(cfg.instance_extractor)
    try:
        return _EXTRACTORS[name](cfg)
    except KeyError:
        raise ValueError(
            f"Unknown instance_extractor {name!r}; expected one of {sorted(_EXTRACTORS)}."
        ) from None
