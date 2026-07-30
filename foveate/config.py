"""Unified :class:`Config` — every tunable knob in one place.

This collapses the old ``InSID3Params`` (gate / cluster / split stage params) and the
loose ``cascade`` keyword arguments into a single dataclass that is threaded
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

    # --- WHERE: foreground extraction strategy ---
    foreground_extractor: str = "insid3"  # insid3 | otsu | bank | oracle | oracle_cc — how "where is
                                          # the concept" is decided. "oracle" returns the GROUND-TRUTH
                                          # foreground (upper-bound Where ablation); "oracle_cc" also
                                          # carves seams between distinct GT instances so connected
                                          # components pre-separate touching ones. Both need the target
                                          # GT, supplied via cascade(gt_foreground=...) by the runner.

    # --- WHERE: Otsu extractor (cheap baseline; paper Sec. 3 "Where") ---
    otsu_top_k: int = 1                   # per crop, build the similarity map from the K exemplars
                                          # whose CLS is most cosine-similar to the crop
    otsu_reduce: str = "mean"             # reduce the per-patch similarity over the K exemplars:
                                          # mean | max

    # --- EXTRACT: which instances does the foreground hold? (slot 2 of 3, foveate.extract) ---
    instance_extractor: str = "kmeans"    # cc | kmeans | agglomerative | watershed — how a crop's
                                          # foreground becomes instance candidates. All share the
                                          # connected-components proposal; they differ in how a
                                          # CONVERGED component is split (an internal boundary CC
                                          # cannot draw): kmeans (k=2 on features, always splits, so
                                          # the Stop slot decides) | agglomerative (cluster the
                                          # clump's patches at cluster_tau) | watershed (marker-
                                          # controlled, geometric waist) | cc (never split — a
                                          # converged crop is accepted whole). Legacy key/value:
                                          # ``split_mode``, whose "none" == "cc".
    extract_connectivity: int = 8         # 4 | 8 pixel/patch connectivity for the CC on the
                                          # foreground grid (8 = don't over-split single instances)

    # --- INSID3 foreground extractor (official algorithm; paper Sec. 3, Tab. 1) ---
    insid3_tau_fg: float = 0.6            # tau_fg: INSID3 foreground-granularity threshold (paper
                                          # Tab. 1) — fine-grained clustering similarity threshold
    insid3_aggt: float = 0.2             # AggT: INSID3 aggregation threshold (paper Tab. 1) — min
                                          # combined score to merge a candidate cluster with the seed
    insid3_linkage: str = "average"       # agglomerative linkage for INSID3 clustering (no graph)
    insid3_crop_reference: bool = True    # crop ref image+mask to each exemplar bbox before
                                          # embedding, so the reference scale matches zoomed crops
    insid3_ref_pad_frac: float = 0.15     # DEPRECATED / unused: the reference exemplar crops are now
                                          # padded by ``pad_frac`` (below) so bank and target crops
                                          # share one framing — a CLS comparison must be apples-to-
                                          # apples. Kept for config compat.
    insid3_top_k_exemplars: int = 1       # per crop, run INSID3 on the K exemplars whose CLS is
                                          # most cosine-similar to the crop (1 = standard INSID3)
    insid3_dynamic_tau_fg: bool = False   # derive tau_fg from the crop↔exemplar CLS similarity s
                                          # (overrides insid3_tau_fg above)
    insid3_dynamic_aggt: bool = False     # derive AggT from s (overrides insid3_aggt above)
    insid3_tau_fg_scale: float = 0.7      #   tau_fg = tau_fg_scale * (1 - s): a dissimilar crop must
                                          #   be split into many fine clusters to find the object; a
                                          #   frame-filling match only needs a coarse fg/bg split
    insid3_aggt_scale: float = 0.3        #   AggT = aggt_scale * s: a dissimilar crop's many
                                          #   clusters re-merge freely

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
    # The ONLY stopping signals are re-identification (a child's re-id score g must beat the crop it
    # came from) and crop SIZE (the size floor rho, ``min_crop``). No depth cap, no split margin.
    max_depth: int = 8                    # DEPRECATED / unused: depth is no longer a stopping
                                          # signal (re-id + min_crop are); kept for config compat
    min_crop: int = 64                    # rho: size floor (px) — a crop at or below this is emitted
                                          # without splitting further (paper Sec. 3, Algorithm 1)
    pad_frac: float = 0.08               # padding fraction around child crops
    shrink_stop: float = 0.9             # converge when child/crop area ratio >= this
    stop_rule: str = "reid"               # STOP slot (slot 3 of 3, foveate.stop): reid | oracle —
                                          # what decides descend / emit / reject. "reid" is the paper
                                          # rule (peak guard + confirm-then-floor on g, floored by
                                          # crop_sim_floor below). "oracle" keeps that exact rule but
                                          # swaps g for a GT box<->instance isolation score, bounding
                                          # the rule's headroom independently of the signal; it needs
                                          # the target GT, supplied via cascade(gt_foreground=...).
    crop_sim_floor: float = 0.5           # tau_C: crop similarity floor (paper Sec. 3). A converged
                                         # crop whose re-id score g is below this is not the concept
                                         # -> reject; split children are kept only if g >= tau_C.
    reid_top_k: int = 0                   # re-identification score g(c) = mean crop similarity to the
                                         # top-K most-similar exemplar CLS in the bank (0 or >= S ⇒
                                         # all exemplars = the mean-over-bank default of Eq. 2; 1 ⇒
                                         # max; 2 ⇒ top-2 mean). Mirrors INSID3's top-k exemplar
                                         # selection. Only bites for multi-exemplar banks (S>1).
    reid_mode: str = "cls"                # how g(c) is computed by the standalone scorer (foveate.
                                         # reid), independent of the Where extractor. All modes score
                                         # a crop as a set of vectors (per exemplar: mean over target
                                         # parts of best cosine to an exemplar part): "cls" = the CLS
                                         # token (whole-crop, framing-sensitive, Eq. 1/2; needs no
                                         # mask) | "mean" = mean of the extracted FOREGROUND patches |
                                         # "kmeans" = reid_kmeans_k centroids of them (k=1 ≡ mean) |
                                         # "full" = every foreground patch (faithful, heaviest). The
                                         # masked modes score the actual Where mask, so they live on a
                                         # different scale than CLS — re-tune crop_sim_floor when
                                         # switching. reid_top_k aggregates all modes (top-k mean).
    reid_kmeans_k: int = 4                # k for reid_mode="kmeans": centroids per crop compared to
                                         # the exemplar's k centroids (1 collapses to reid_mode="mean").
    confidence_reid_mode: str | None = None  # how the FINAL leaf confidence (the score NMS + AP rank
                                         # by) is computed, DECOUPLED from the recursion stop signal g
                                         # (which stays reid_mode above). None = legacy: a leaf inherits
                                         # its crop's recursion score, so all instances carved out of
                                         # one crop share one confidence (a sliver ties the true
                                         # instance). Set to a MASKED mode ("mean" | "kmeans" | "full")
                                         # to instead score EACH emitted instance on its OWN foreground
                                         # patches — "full" = mean over the instance's foreground
                                         # patches of best cosine to the exemplar patch set (the "mean
                                         # cosine sim of the masked foreground features"). Reuses the
                                         # masked-family debias/top-k knobs.
    zoom_split_retry_eps: float = 0.01   # when a zoom peaks by only a hair (parent g beats the
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
    mask_upsample: str = "nearest"       # how a leaf's patch-grid mask is upsampled to pixels:
                                         # nearest (blocky patch staircase) | bilinear (smooth the
                                         # boundary — resize as float, re-binarize at 0.5)
    # --- deduplication (final NMS on emitted leaves) ---
    nms_iou: float = 0.5                 # suppress a lower-scored leaf overlapping a kept one above
                                         # this mask IoU (independent branches re-finding one object)
    nms_containment: float = 0.7         # ...or contained in a kept one beyond this fraction of its
                                         # area — catches the NESTED duplicate a plain IoU misses
                                         # (a tight zoomed mask inside a looser one). Set BOTH
                                         # nms_iou and nms_containment to 1.0 to disable NMS.
    embed_batch_size: int = 8            # crops per backbone forward
    max_total_embeds: int = 512          # global embed budget (safety cap)
                                         # (the converged-clump splitter moved to the Extract slot:
                                         # ``instance_extractor`` above; ``split_mode`` still works
                                         # as a config key via _ALIASES, with "none" == "cc")

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
    # crop_sim_floor (above, recursion bounds) is the sole acceptance test: mean cosine of the
    # target CLS to all exemplar CLS must clear it for a converged crop to be kept.
    discard_rejected: bool = False       # drop (vs keep) converged crops that fail acceptance
    emit_components: bool = False         # when a parent is emitted (reid-stop fallback), split its
                                         # OR-merged foreground into connected components and emit
                                         # each separately, instead of one merged mask. Recovers
                                         # non-touching instances a rejected split would otherwise fuse.

    #: Pre-rename config keys → their paper-aligned attribute names. Old YAMLs / notebooks that
    #: still use the legacy keys keep working; :meth:`from_dict` maps them transparently.
    _ALIASES = {
        "cls_threshold": "crop_sim_floor",          # tau_C, crop similarity floor
        "cls_top_k": "reid_top_k",                  # re-identification score g top-k
        "split_mode": "instance_extractor",         # the Extract slot (value "none" == "cc")
        "insid3_tau": "insid3_tau_fg",              # tau_fg
        "insid3_aggregate_threshold": "insid3_aggt",  # AggT
        "insid3_dynamic_tau": "insid3_dynamic_tau_fg",
        "insid3_tau_scale": "insid3_tau_fg_scale",
        "insid3_dynamic_aggregate_threshold": "insid3_dynamic_aggt",
        "insid3_aggregate_scale": "insid3_aggt_scale",
    }

    @classmethod
    def from_dict(cls, params: dict[str, Any] | None) -> "Config":
        """Build a Config, overriding defaults with non-None values in ``params``.

        Legacy key names (see :attr:`_ALIASES`) are accepted and mapped to their current
        attribute. Unknown keys are ignored (so a request can carry extra metadata harmlessly).
        """
        base = cls()
        if not params:
            return base
        for key, value in params.items():
            key = cls._ALIASES.get(key, key)
            if hasattr(base, key) and value is not None:
                setattr(base, key, value)
        return base

    # Request params override defaults — the service path (plan Sec. 6).
    from_request = from_dict

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Backward-compatible alias for the pre-rename name.
InSID3Params = Config
