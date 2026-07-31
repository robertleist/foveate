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

    # --- EXTRACT: which instances of the concept are on this crop? (slot 1 of 3, foveate.extract) ---
    extractor: str | None = None          # composite | oracle. None (default) DERIVES it from the
                                          # pre-two-slot key pair below, so every existing YAML keeps
                                          # working: "oracle" when the DECOMPOSITION was oracular
                                          # (instance_extractor: oracle, or foreground_extractor:
                                          # oracle_cc — itself a Where+group method), "composite"
                                          # otherwise. A plain foreground_extractor: oracle therefore
                                          # stays a composite with a perfect Where, so the
                                          # Where-headroom ablation keeps its meaning.

    # --- EXTRACT / composite: the Where half — which patches are the concept? ---
    foreground_extractor: str = "insid3"  # insid3 | otsu | bank | oracle | oracle_cc (alias key:
                                          # `where`). Only consulted by the composite extractor.
                                          # "oracle" returns the GROUND-TRUTH foreground (upper-bound
                                          # Where ablation); "oracle_cc" also carves seams between
                                          # distinct GT instances so connected components pre-separate
                                          # touching ones. Both need the target GT, supplied via
                                          # cascade(gt_foreground=...) by the runner.

    # --- WHERE: Otsu extractor (cheap baseline; paper Sec. 3 "Where") ---
    otsu_top_k: int = 1                   # per crop, build the similarity map from the K exemplars
                                          # whose CLS is most cosine-similar to the crop
    otsu_reduce: str = "mean"             # reduce the per-patch similarity over the K exemplars:
                                          # mean | max
    otsu_min_separability: float = 0.0    # scale guard (roadmap A1.2). Otsu always returns a cut,
                                          # even on a UNIMODAL map — which is what a crop the object
                                          # already fills produces — and then the cut lands *inside*
                                          # the object, the padded bbox comes in too tight and the
                                          # child crop truncates it (over-zoom). When the cut's
                                          # separability eta (thresholding.separability) is below
                                          # this, the map is judged unimodal and
                                          # ``otsu_unimodal_fallback`` decides instead. 0 = off
                                          # (always trust Otsu, the pre-A1.2 behaviour).
    otsu_unimodal_fallback: str = "accept_all"  # what to do on a map judged unimodal:
                                          # accept_all (the concept fills the crop → converge here,
                                          # let Stop decide) | percentile (cut at gate_percentile) |
                                          # otsu (keep the cut anyway — a no-op control for the
                                          # ablation, so the guard's *detection* can be measured
                                          # separately from its *action*)
    otsu_hysteresis_lo: float = 0.0       # two-threshold foreground (roadmap A1.2): seed at the
                                          # Otsu cut, then grow into connected patches within this
                                          # margin BELOW it, on the min-max-normalized map (so the
                                          # margin is a fraction of the crop's similarity range).
                                          # Recovers the dim rim a single cut shaves off — the
                                          # cheapest over-zoom fix. 0 = off (single cut).

    # --- EXTRACT / composite: the grouping half — which instances does that foreground hold? ---
    instance_extractor: str = "kmeans"    # cc | kmeans | agglomerative | watershed (alias key:
                                          # `grouping`; legacy key `split_mode`, whose "none" ==
                                          # "cc"). All share the connected-components proposal; they
                                          # differ in whether they cut a component that already FILLS
                                          # the crop — the only moment an internal boundary can add
                                          # instances, because below it the cascade's own zoom will
                                          # ask again at a larger effective resolution (see
                                          # foveate.grouping): kmeans (k=2 on features, always cuts,
                                          # so the Stop slot's peak guard decides) | agglomerative
                                          # (cluster the patches at cluster_tau) | watershed (marker-
                                          # controlled, geometric waist) | cc (never cut).
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
    crop_dilate: float = 0.0             # dilate a component by this many PATCHES before taking the
                                         # child crop's bbox (roadmap A1.2). Padding is relative
                                         # (pad_frac * side), so it shrinks with the crop and cannot
                                         # cover a fixed one-patch quantization error; at 48x48 one
                                         # patch is ~2% of the side, and the foreground's outermost
                                         # patch is exactly where a threshold is least certain.
                                         # Affects ONLY the box, never the emitted mask — widening
                                         # the mask would trade boundary precision for it. 0 = off.
                                         # FRACTIONAL on purpose: a patch is foreground when its
                                         # CENTRE falls inside the object, so the grid bbox can be
                                         # short by at most HALF a patch on each side. 0.5 is the
                                         # quantization error itself; whole patches over-correct and
                                         # inflate the crop (and the cost) beyond what is needed.
    shrink_stop: float = 0.9             # converge when child/crop area ratio >= this
    oracle_isolation: str = "bbox"       # ORACLE STOP ONLY. How "am I framing exactly one instance?"
                                         # is scored: "bbox" = IoU(crop box, GT instance BBOX) — 1
                                         # exactly when the crop equals that bbox, and comparable
                                         # across instance shapes. "mask" = the original IoU(crop as
                                         # a filled rectangle, GT instance MASK), whose maximum is
                                         # only the instance's fill ratio (mask area / bbox area) and
                                         # can be raised by shrinking INTO the mask — so its peak
                                         # lies tighter than the true bbox and the rule over-zooms.
                                         # Kept as an ablation arm, not a default.
    oracle_isolation_select: str = "max"  # which GT instance the crop is scored against: "max" (the
                                         # best-scoring one — the upper envelope, so the peak is the
                                         # best crop available) | "center" (the one whose bbox centre
                                         # is nearest the crop centre, among those the crop overlaps;
                                         # keeps the target fixed along a zoom chain)
    oracle_coverage: str = "any"         # ORACLE SLOTS ONLY. How the GT is put on the patch grid:
                                         # "any" = a patch is positive if it contains ANY pixel of an
                                         # instance | "center" = only if the patch centre is inside.
                                         # "any" is the right upper bound because Where is a
                                         # PROPOSAL ("the concept may be here, go and look"), and the
                                         # cascade then foveates onto it. Centre sampling deletes
                                         # every object smaller than a patch before the recursion can
                                         # see it — no patch, no component, no crop — which is a
                                         # downsampling artefact, not a Where error an oracle should
                                         # inherit. "any" also makes the patch-interval box CONTAIN
                                         # the object instead of under-covering it, so crop_dilate
                                         # stops being needed. "center" kept for the ablation.
    stop_rule: str = "reid"               # STOP slot (slot 2 of 3, foveate.stop): reid | oracle —
                                          # what decides descend / emit / reject. "reid" is the paper
                                          # rule (fixed point + peak guard on g, floored by
                                          # crop_sim_floor below). "oracle" keeps that exact rule but
                                          # swaps g for a GT box<->instance isolation score, bounding
                                          # the rule's headroom independently of the signal; it needs
                                          # the target GT, supplied via cascade(gt_foreground=...).
    stop_fixed_point_iou: float = 0.9     # the FIXED POINT tolerance: a child crop stops when its
                                          # extraction is the single instance it was cropped for, at
                                          # or above this IoU in original-image coordinates. Not a
                                          # difference-of-opinion threshold — parent and child crops
                                          # have different pixel extents, so the same object is a
                                          # coarser mask on the parent's patch grid than on the
                                          # child's, and this is the tolerance for that
                                          # re-quantization. 1.0 effectively disables it (only the
                                          # geometric shrink_stop and the peak guard then stop the
                                          # descent); lower values stop earlier and cost mask detail.
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
    zoom_split_retry_eps: float = 0.01   # DEPRECATED / unused: it hedged a hair-thin zoom peak by
                                         # forcing ONE extra k=2 split, which only existed because
                                         # the splitter could not decide an instance count. The
                                         # extractor now returns the instances, so there is nothing
                                         # to retry; kept for config compat.
    clump_area_factor: float = 1.5       # DEPRECATED / unused: convergence now ALWAYS attempts a
                                         # split (no tiny-blob fast path); kept for config compat
    split_margin: float = 0.0            # DEPRECATED / unused: a split is confirmed when its BEST
                                         # sub-crop strictly beats the parent CLS (isolating a real
                                         # object raises CLS), then every sub-crop above the class
                                         # floor is kept; kept for config compat
    split_aggregate: str = "mean"        # DEPRECATED / unused: sub-crops are gated by the split
                                         # confirm + class floor, not pooled; kept for config compat
    cascade_min_instance_area: int = 16  # drop leaf masks smaller than this (pixels)
    mask_refine: str = "none"            # none | grabcut — snap the emitted patch-grid mask to the
                                         # image's own boundaries (foveate.mask_refine). A mask built
                                         # on a 16 px grid cannot score well at strict IoU: an
                                         # instance 2 patches across is localized to ~half its width,
                                         # so mask AP@0.75+ measures the patch size, not the method.
                                         # Matters most on the dense slice, where instances are 1-2
                                         # patches. Costs CPU per emitted instance.
    mask_refine_band: int = 2            # patches of uncertainty around the mask edge that GrabCut
                                         # re-decides; inside stays foreground, outside background
    mask_refine_iters: int = 3           # GrabCut iterations
    mask_refine_max_change: float = 0.5  # reject a refinement that changes the area by more than
                                         # this fraction (the colour model latched onto background)
    report_boxes: str = "mask"           # which box is scored for DETECTION AP: "mask" = the tight
                                         # box of the emitted mask (what COCO-FSOD / RF20-VL /
                                         # CD-FSOD compare against; GT is always a tight box) |
                                         # "crop" = the final crop the cascade converged on. The
                                         # crop is padded by pad_frac/crop_dilate, so scoring it as
                                         # a detection penalises the method for its own padding —
                                         # keep it as a framing DIAGNOSTIC, not as the reported box.
    mask_upsample: str = "nearest"       # MASK slot (foveate.upsample): how a leaf's patch-grid mask
                                         # becomes pixels. nearest (the exact patch staircase) |
                                         # bilinear (resize as float, re-binarize at 0.5 — a
                                         # smoothing prior that uses NO image evidence, so it is the
                                         # floor for this slot, not the answer) | oracle (the upper
                                         # bound: the GT shape of a detection that ALREADY matches).
                                         # A guided/joint-bilateral arm and feature upsampling
                                         # (roadmap A1.3) are the real entries this slot is for.
    oracle_upsample_iou: float = 0.5     # ORACLE MASK ONLY. Minimum IoU the base mask must already
                                         # reach before its shape is replaced. Shape-ONLY on purpose:
                                         # an oracle allowed to snap any mask to its nearest object
                                         # would do the Merge slot's job (~44 % of dense detections
                                         # are clipped fragments) and book a search gain as boundary
                                         # quality. Below this the base mask is returned untouched.
    oracle_upsample_base: str = "bilinear"  # ORACLE MASK ONLY: which upsample decides what "already
                                         # matches" means (nearest | bilinear)
    # --- MERGE: how the emitted leaves combine (slot 3 of 3, foveate.merge_rule) ---
    merge_rule: str = "nms"              # nms | soft | none. "nms" (default) is the behaviour of
                                         # record: greedy score-ranked suppression by mask IoU AND
                                         # containment, after the optional fragment union below.
                                         # "soft" decays an overlapping detection's score instead of
                                         # deleting it (soft-NMS, Bodla et al.) — it keeps the
                                         # second-best detection of a crowded region alive, which is
                                         # where hard suppression costs recall. NOTE: this is the
                                         # standard, extractor-agnostic soft rule, NOT the
                                         # semantic-aware soft merge of "No Time to Train!" — that
                                         # needs an extractor's own semantic scores and belongs next
                                         # to an NTT extractor as its own entry. "none" emits the
                                         # recursion's leaves untouched (the diagnostic arm).
    merge_soft_sigma: float = 0.5        # soft rule only: Gaussian decay width, s <- s * exp(-o^2/s)
                                         # where o is the larger of IoU and containment against an
                                         # already-kept detection. Smaller = harsher.
    merge_soft_score_floor: float = 1e-3  # soft rule only: drop a detection once its decayed score
                                         # falls below this (the soft analogue of deletion)
    # --- deduplication (inputs to the nms / soft / oracle merge rules) ---
    merge_drop_clipped: bool = False     # drop a detection whose mask touches its own CROP border
                                         # (an image border does not count — the object really ends
                                         # there). Such a mask is PROVABLY a piece: the object
                                         # continues outside the crop it was found in, while the
                                         # same object is framed properly by its own branch. This is
                                         # provenance, not overlap, so suppression cannot see it —
                                         # the piece barely overlaps the real detection. With Extract
                                         # and Stop oracular these fragments are the dominant false
                                         # positive (~44 % of dense detections, ~32 % of general, at
                                         # a median 0.21 of their object's area). Runs BEFORE the
                                         # fragment union and the suppression rule.
    merge_fragments: bool = False        # union detections that are pieces of ONE object cut apart
                                         # by a crop boundary. NMS cannot do this: two halves found
                                         # in two crops are DISJOINT (IoU ~ 0, no containment), so no
                                         # suppression rule relates them, and deleting either would
                                         # lose pixels. Pairs are joined when they are adjacent AND
                                         # at least one touches its own crop border (= provably
                                         # partial). Runs BEFORE nms.
    merge_fragment_gap: int = 2          # pixels of dilation used for the adjacency test
    nms_iou: float = 0.5                 # suppress a lower-scored leaf overlapping a kept one above
                                         # this mask IoU (independent branches re-finding one object)
    nms_containment: float = 0.7         # ...or contained in a kept one beyond this fraction of its
                                         # area — catches the NESTED duplicate a plain IoU misses
                                         # (a tight zoomed mask inside a looser one). Set BOTH
                                         # nms_iou and nms_containment to 1.0 to disable NMS.
    embed_batch_size: int = 8            # crops per backbone forward
    max_total_embeds: int = 512          # global embed budget (safety cap)

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
        "split_mode": "instance_extractor",         # the composite's grouping (value "none" == "cc")
        "where": "foreground_extractor",            # the composite's Where half (short spelling)
        "grouping": "instance_extractor",           # the composite's grouping half (short spelling)
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
