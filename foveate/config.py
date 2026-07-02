"""Unified :class:`Config` — every tunable knob in one place.

This collapses the old ``InSID3Params`` (gate / cluster / split stage params) and the
loose ``discover_instances`` keyword arguments into a single dataclass that is threaded
through every stage, so the notebook, the experiments and any service wrapper share one
source of truth (no more parameter drift).

Grouped, roughly, into: stage params (gate / cluster / individuation / merge), recursion
bounds, prototype bank, thresholding, debiasing and acceptance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Config:
    # --- features ---
    standardize: bool = True              # z-score feature dims before L2-norm

    # --- foreground extraction strategy ---
    foreground_extractor: str = "insid3"  # insid3 | bank — how "where is the class" is decided

    # --- INSID3 foreground extractor (official algorithm) ---
    insid3_tau: float = 0.6               # fine-grained clustering similarity threshold
    insid3_aggregate_threshold: float = 0.2  # alpha: min combined score to merge a cluster
    insid3_linkage: str = "average"       # agglomerative linkage for INSID3 clustering (no graph)
    insid3_crop_reference: bool = True    # crop ref image+mask to each exemplar bbox before
                                          # embedding, so the reference scale matches zoomed crops
    insid3_ref_pad_frac: float = 0.15     # DEPRECATED / unused: the reference exemplar crops are now
                                          # padded by ``pad_frac`` (below) so bank and target crops
                                          # share one framing — a CLS comparison must be apples-to-
                                          # apples. Kept for config compat.
    insid3_top_k_exemplars: int = 1       # per crop, run INSID3 on the K exemplars whose CLS is
                                          # most cosine-similar to the crop (1 = standard INSID3)
    insid3_dynamic_params: bool = False   # derive tau / aggregate_threshold from the crop↔exemplar
                                          # CLS similarity s (overrides the static values above):
    insid3_tau_scale: float = 0.7         #   tau = tau_scale * s  (raw s≈1 over-segments; scale it)
    insid3_aggregate_scale: float = 0.3   #   aggregate_threshold = aggregate_scale * (1 - s)
                                          #   (similar crop → low threshold → aggregate freely)

    # --- semantic gate / clustering / individuation / merge (single-pass pipeline) ---
    gate_threshold: float = 0.55          # absolute cosine to keep a patch as the class
    cluster_tau: float = 0.18             # agglomerative distance threshold (granularity)
    marker_mode: str = "hybrid"           # geometric | feature | hybrid
    elevation_alpha: float = 1.0          # weight of feature-boundary in watershed elevation
    elevation_beta: float = 0.5           # weight of geometric (1 - distance) term
    marker_min_distance: int = 2          # min spacing (patches) between geometric seeds
    boundary_smooth_sigma: float = 0.0    # Gaussian sigma (patches) smoothing the distance /
                                          # feature-boundary maps before watershed -> fewer
                                          # spurious markers/minima (0 = off)
    merge_similarity: float = 0.6         # min prototype cosine to merge neighbours
    merge_boundary: float = 0.5           # max border feature-boundary to allow a merge
    min_instance_area: int = 4            # drop instances smaller than this (patches, pipeline)
    score_threshold: float = 0.0          # drop instances with mean gate score below this

    # --- recursion bounds (cascade) ---
    # The ONLY stopping signals are CLS re-identification (a child must beat the crop it came from)
    # and crop SIZE (``min_crop``). There is no depth cap and no split margin.
    max_depth: int = 8                    # DEPRECATED / unused: depth is no longer a stopping
                                          # signal (CLS + min_crop are); kept for config compat
    min_crop: int = 64                    # resolution floor (px) — stop zooming/splitting below this
    pad_frac: float = 0.08               # padding fraction around child crops
    shrink_stop: float = 0.9             # converge when child/crop area ratio >= this
    cls_threshold: float = 0.5           # absolute class FLOOR: converged crop below this is
                                         # not the class -> reject (was the accept threshold)
    zoom_split_retry_eps: float = 0.01   # when a zoom peaks by only a hair (parent CLS beats the
                                         # child by less than this), the crop may be a CLUMP that
                                         # tightening onto one component can't improve — so try ONE
                                         # k=2 split of the parent before emitting it. If the split
                                         # doesn't improve either, the parent is emitted. 0 disables.
    clump_area_factor: float = 1.5       # DEPRECATED / unused: convergence now ALWAYS attempts a
                                         # split (no tiny-blob fast path); kept for config compat
    split_margin: float = 0.0            # DEPRECATED / unused: a split is confirmed when its BEST
                                         # sub-crop strictly beats the parent CLS (isolating a real
                                         # object raises CLS), then every sub-crop above the class
                                         # floor is kept; kept for config compat
    split_aggregate: str = "mean"        # DEPRECATED / unused: sub-crops are gated by the split
                                         # confirm + class floor, not pooled; kept for config compat
    cascade_min_instance_area: int = 16  # drop leaf masks smaller than this (pixels)
    # --- deduplication (final NMS on emitted leaves) ---
    nms_iou: float = 0.5                 # suppress a lower-scored leaf overlapping a kept one above
                                         # this mask IoU (independent branches re-finding one object)
    nms_containment: float = 0.7         # ...or contained in a kept one beyond this fraction of its
                                         # area — catches the NESTED duplicate a plain IoU misses
                                         # (a tight zoomed mask inside a looser one). Set BOTH
                                         # nms_iou and nms_containment to 1.0 to disable NMS.
    embed_batch_size: int = 8            # crops per backbone forward
    max_total_embeds: int = 512          # global embed budget (safety cap)
    split_mode: str = "kmeans"           # how a converged clump is split: kmeans (k=2 on features,
                                         # always splits) | watershed (marker-controlled) | none

    # --- prototype bank ---
    prototype_reduction: str = "all"     # all | mean | cluster | kmeans
    prototype_budget: int = 64           # global cap on #prototypes (kmeans/all)
    prototype_per_exemplar_min: int = 1  # min prototypes guaranteed per exemplar (kmeans)
    n_prototypes: int = 4                # k for legacy per-exemplar "cluster" reduction

    # --- thresholding (bank extractor) ---
    gate_threshold_mode: str = "static"  # static | otsu | gmm2 | percentile (per-crop gate)
    gate_percentile: float = 80.0        # used when gate_threshold_mode == "percentile"

    # --- positional debiasing (cross-image matching) ---
    debias: bool = False                 # project out DINOv3 positional subspace for matching
    debias_subspace_dim: int = 8
    debias_n_noise: int = 1
    debias_seed: int = 0

    # --- leaf classification / acceptance ---
    # cls_threshold (above, recursion bounds) is the sole acceptance test: mean cosine of the
    # target CLS to all exemplar CLS must clear it for a converged crop to be kept.
    discard_rejected: bool = False       # drop (vs keep) converged crops that fail acceptance

    @classmethod
    def from_dict(cls, params: dict[str, Any] | None) -> "Config":
        """Build a Config, overriding defaults with non-None values in ``params``.

        Unknown keys are ignored (so a request can carry extra metadata harmlessly).
        """
        base = cls()
        if not params:
            return base
        for key, value in params.items():
            if hasattr(base, key) and value is not None:
                setattr(base, key, value)
        return base

    # Request params override defaults — the service path (plan Sec. 6).
    from_request = from_dict

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Backward-compatible alias for the pre-rename name.
InSID3Params = Config
