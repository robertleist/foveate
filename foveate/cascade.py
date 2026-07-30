"""The Foveate cascade — recursively zoom until each instance is re-identified (paper Algorithm 1).

Breadth-first, batched, connected-components fixed point over a frontier of crops. Every crop is
embedded once and answers two questions (paper Sec. 3): *where is the concept?* and, once the crop
has converged, *is this a single instance?* Both are guided by the exemplar bank via the
re-identification score ``g`` (mean crop similarity to the bank, Eq. 2), where crop similarity is
the cosine of two CLS tokens (Eq. 1).

**Three swappable slots.** This module owns the *loop* — batching, geometry, budget, termination —
and delegates every *policy* to one of three strategy interfaces, each with its own registry and its
own oracle (so an ablation attributes the remaining error to a slot):

===============  =========================  ==========================================
Slot             Module / config key        Question
===============  =========================  ==========================================
**Where**        :mod:`foveate.foreground`  which patches of this crop are the concept?
                 ``foreground_extractor``
**Extract**      :mod:`foveate.extract`     which instances does that foreground hold?
                 ``instance_extractor``
**Stop**         :mod:`foveate.stop`        descend, emit or reject?
                 ``stop_rule``
===============  =========================  ==========================================

Two things stay here on purpose: the **size floor** ρ (``min_crop``) and the strict box shrink, which
together are the termination *invariant* — the recursion is finite whatever the slots do — and the
no-geometric-progress guards (a child box equal to its parent's), which protect the loop, not the
science. For each crop:

1. **Where** — re-embed (batched per level) and **extract the concept foreground** with the
   configured foreground extractor (INSID3 by default, the exemplar bank as an alternative) → the
   region of the crop that belongs to the exemplar concept.
2. **Extract** — connected components propose tighter crops:
   - ≥ 2 components  → enqueue each (tighter child crops);
   - 1 component whose bbox does **not** fill the crop → enqueue the tighter crop (keep zooming);
   - 1 component whose bbox **fills** the crop → *converged* (no tighter crop extractable).

   Zooming does **not** run open-loop: a child crop is only pursued if its re-id score ``g``
   re-identifies the exemplar *more strongly* than the crop it was zoomed out of. When no child of
   a parent beats the parent, the parent was the ``g`` peak and is emitted as the instance (if it
   clears the crop similarity floor ``crop_sim_floor``, τ_C); when some children beat it, only those
   continue and the lower-similarity siblings are discarded. This is the ``reid-stop`` rule — it
   keeps the cascade from over-zooming past the scale at which the object is best recognized.
3. **Split** — a converged crop (zoom exhausted) is **never accepted as a leaf on the spot**; it is
   *always* split k=2 on its foreground and the sub-crops are put back on the frontier against this
   crop as their parent (next level). Whether the split "took" is judged by ``g``:
   - **Confirm** the split is real — its *best* sub-crop must *strictly* beat the parent's ``g``.
     Isolating a real object from a mixed crop raises ``g`` (the other instance and the background
     that diluted the parent drop away), so a genuine clump always has a sub-crop above the parent;
     a single object only yields weaker partial halves (and flat/tied ``g`` never beats it), so its
     split is not confirmed → fall back and emit **this crop** (``reid-stop``).
   - Once confirmed, keep **every** sub-crop that independently clears ``crop_sim_floor`` (τ_C) —
     each is its own instance and is pursued. This is the key subtlety: a crop can hold the original
     exemplar *and* a genuinely novel instance, so the parent's ``g`` is biased high by the exemplar
     it contains; gating each sub-crop on "beat the parent" would wrongly discard the novel sibling
     (concept-like, but a different instance, so scoring below that inflated parent). Confirm-then-floor
     keeps it. (A single-child *zoom*, by contrast, still uses the strict beat-the-parent peak guard.)
   The other exits: the Stop rule rejects the crop (below ``config.crop_sim_floor``) → not the
   concept; or the Extract slot cannot split (``instance_extractor="cc"``) → accepted whole.

   So the only stopping signals are re-identification (``g``) and the size floor ρ (``min_crop``): a
   crop at or below it is emitted without splitting further. No depth cap, no split margin.

The exemplar defines the target *scale and granularity*. Only leaves are returned (after NMS).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from foveate import features as featlib, mask_refine
from foveate.config import Config
from foveate.extract import build_instance_extractor, component_label_map
from foveate.foreground import build_extractor
from foveate.reid import build_reid_scorer
from foveate.stop import build_stop_rule
from foveate.types import Instance, Stats


@dataclass
class _Parent:
    """The crop a batch of child crops was zoomed out of.

    Kept so the cascade can fall back to it when none of its children re-identify the exemplar
    more strongly: the re-id score ``g`` peaked at the parent, so *it* is the instance. ``comps``
    are the parent's foreground components (patch grids); they are OR-merged into one instance on
    emit (a converged ``g`` says "the object is in here", so a multi-component parent is one object
    the foreground extractor happened to fragment).
    """
    reid: float                                # parent re-identification score g(c)
    box: tuple[int, int, int, int]
    depth: int
    comps: list
    # The parent's own observer event, held back until its children's fate is known: a zoom/split
    # region is only labelled ``zoom``/``split`` if a child improved on it, else ``reid-stop``. So
    # the observer sees each region once, with its FINAL decision (the trajectory tooling rebuilds
    # the tree by unique box, so a region must appear exactly once).
    event: dict | None = None
    # The parent's patch feature grid ``(Hp, Wp, D)``. Held so a hair-thin zoom peak can be split
    # k=2 in place before emit (``zoom_split_retry``) AND so each emitted component can be scored on
    # its OWN foreground for the masked confidence (see ``confidence_reid_mode``). Always set now.
    feat: object | None = None
    # Whether this parent may still attempt a one-shot k=2 retry. True only for single-component
    # (zoom) parents; a retry produces a split parent with this False, so no unbounded retry chain.
    allow_retry: bool = False


@dataclass
class _Region:
    box: tuple[int, int, int, int]
    depth: int
    # Zoom/split children are re-embedded next level; clump-split children carry their lookahead
    # embedding so the k=2 split forward isn't paid twice over the same sub-crops.
    embedded: tuple | None = None
    # The crop this one was zoomed OR split out of. A child is only pursued if it beats
    # ``parent.reid``; if no child of a parent beats it, the parent is emitted as the instance.
    # Set for zoom, split AND clump-split children — the split is g-gated exactly like zoom.
    # ``None`` only for the root.
    parent: _Parent | None = None


def _grid_bbox(comp):
    rows, cols = np.where(comp)
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


def _child_box(comp, box, pad_frac, dilate: float = 0.0):
    """Pixel bbox (in original coords) of a grid component within ``box``, padded.

    ``dilate`` (``cfg.crop_dilate``, in **patches**) widens the grid bbox before it is converted to
    pixels — a fixed, scale-independent safety margin on the Where step's outermost patch, which is
    exactly where its threshold is least certain. ``pad_frac`` cannot play that role: being relative,
    it shrinks along with the crop just as the quantization error stays constant.
    """
    y0, y1, x0, x1 = box
    ch, cw = y1 - y0, x1 - x0
    hp, wp = comp.shape
    rmin, rmax, cmin, cmax = _grid_bbox(comp)
    if dilate:
        # Fractional: the grid bbox under-states the object by up to half a patch on each side
        # (a patch is foreground when its *centre* is inside), so 0.5 is the exact correction.
        rmin = max(0.0, rmin - dilate); rmax = min(float(hp), rmax + dilate)
        cmin = max(0.0, cmin - dilate); cmax = min(float(wp), cmax + dilate)
    py0 = y0 + int(np.floor(rmin / hp * ch)); py1 = y0 + int(np.ceil(rmax / hp * ch))
    px0 = x0 + int(np.floor(cmin / wp * cw)); px1 = x0 + int(np.ceil(cmax / wp * cw))
    pady, padx = int(round((py1 - py0) * pad_frac)), int(round((px1 - px0) * pad_frac))
    return (max(y0, py0 - pady), min(y1, py1 + pady),
            max(x0, px0 - padx), min(x1, px1 + padx))


def _component_scores(comps: list[np.ndarray], base_score: float, grid_shape) -> list[float]:
    """Per-component confidence when a parent crop is emitted as connected components.

    All components share the crop's re-id score ``g`` (``base_score``), but that ``g`` is biased
    high by the true exemplar in the crop. A component touching the crop **border** is only
    partially in frame — a cut-off sliver of a neighbouring instance — so it must not inherit the
    full ``g``: score-ranked NMS would keep the tiny sliver and suppress that neighbour's full
    detection (found in its own crop). Border-touching components are scaled by their area fraction
    of the crop's dominant component; whole (interior) components keep ``base_score`` unchanged, so
    genuine non-touching instances are unaffected.
    """
    hp, wp = grid_shape
    top = max((int(c.sum()) for c in comps), default=1) or 1
    scores: list[float] = []
    for c in comps:
        ys, xs = np.where(c)
        cut = bool(ys.min() == 0 or ys.max() == hp - 1 or xs.min() == 0 or xs.max() == wp - 1)
        scores.append(base_score * (int(c.sum()) / top) if cut else base_score)
    return scores


def _box_area(box):
    return (box[1] - box[0]) * (box[3] - box[2])


def _boxes_overlap(b1, b2) -> bool:
    """Do two ``(y0, y1, x0, x1)`` boxes intersect? A cheap pre-filter before any mask math."""
    return (min(b1[1], b2[1]) > max(b1[0], b2[0]) and
            min(b1[3], b2[3]) > max(b1[2], b2[2]))


def _mask_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """``(IoU, containment)`` of two boolean masks; containment = intersection / smaller area.

    Containment catches the *nested* duplicate a plain IoU misses: a tightly-zoomed mask sitting
    inside a looser one has low IoU (small∩ over big∪) yet containment ≈ 1.
    """
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0, 0.0
    union = int(np.logical_or(a, b).sum())
    smaller = min(int(a.sum()), int(b.sum())) or 1
    return inter / union, inter / smaller


def _touches_border(mask: np.ndarray, box) -> bool:
    """Does this instance's mask reach the edge of the crop it was found in?

    A component touching its crop border is *provably* only partially in frame — the object
    continues outside the crop — so the emitted mask is a fragment, not an instance.
    """
    y0, y1, x0, x1 = box
    sub = mask[y0:y1, x0:x1]
    if sub.size == 0:
        return False
    return bool(sub[0].any() or sub[-1].any() or sub[:, 0].any() or sub[:, -1].any())


def _merge_fragments(instances: list[Instance], gap: int) -> tuple[list[Instance], int]:
    """Union detections that are pieces of one object cut apart by crop boundaries.

    **Why NMS cannot do this.** Suppression compares a pair by IoU or containment and *deletes* the
    weaker one. Two halves of an instance found in two different crops are **disjoint**: their IoU is
    ~0 and neither contains the other, so no suppression rule relates them at all — and deleting one
    would be wrong anyway, because each holds pixels the other lacks. The repair is a **union**, and
    the pair is recognised by geometry (adjacent) plus provenance (at least one piece is clipped by
    its own crop border), not by overlap.

    Merged score is the maximum over the pieces: a piece that does *not* touch a border was seen
    whole, so its confidence is the trustworthy one.
    """
    if len(instances) < 2:
        return instances, 0

    masks = [inst.mask.astype(bool) for inst in instances]
    partial = [_touches_border(m, inst.box) for m, inst in zip(masks, instances)]
    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        boxes.append((ys.min(), ys.max() + 1, xs.min(), xs.max() + 1) if ys.size else (0, 0, 0, 0))

    parent = list(range(len(instances)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    struct = np.ones((2 * gap + 1, 2 * gap + 1), dtype=bool)
    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            if not (partial[i] or partial[j]):
                continue                      # both were seen whole → different objects
            bi, bj = boxes[i], boxes[j]       # cheap reject: tight boxes further apart than the gap
            if (bi[0] > bj[1] + gap or bj[0] > bi[1] + gap
                    or bi[2] > bj[3] + gap or bj[2] > bi[3] + gap):
                continue
            y0 = max(0, min(bi[0], bj[0]) - gap); y1 = max(bi[1], bj[1]) + gap
            x0 = max(0, min(bi[2], bj[2]) - gap); x1 = max(bi[3], bj[3]) + gap
            a = masks[i][y0:y1, x0:x1]
            b = masks[j][y0:y1, x0:x1]
            if (a & b).any() or (binary_dilation(a, structure=struct) & b).any():
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(instances)):
        groups.setdefault(find(i), []).append(i)
    if len(groups) == len(instances):
        return instances, 0

    out = []
    for members in groups.values():
        if len(members) == 1:
            out.append(instances[members[0]])
            continue
        merged = np.logical_or.reduce([masks[i] for i in members]).astype(np.uint8)
        best = max(members, key=lambda i: instances[i].score)
        ys, xs = np.where(merged)
        box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        out.append(Instance(merged, box, instances[best].depth, instances[best].score))
    return out, len(instances) - len(out)


def _nms(instances: list[Instance], iou_thresh: float, contain_thresh: float
         ) -> tuple[list[Instance], int]:
    """Greedy, score-ranked NMS dropping duplicate detections of the *same* object.

    A converged crop is always split k=2; the two sub-crops carry disjoint foreground *patches*
    but their padded pixel boxes can nest or overlap (e.g. a core cluster vs a surrounding one), so
    each re-discovers the whole object down its own branch and it is emitted twice — at possibly
    different depths/scales. Keep the highest-scoring instance and suppress any later one that
    overlaps it above ``iou_thresh`` OR is contained in it beyond ``contain_thresh`` (the nested
    case). Returns ``(kept in original emission order, n_suppressed)``.
    """
    order = sorted(range(len(instances)), key=lambda i: instances[i].score, reverse=True)
    kept: list[int] = []
    kept_masks: list[np.ndarray] = []
    for i in order:
        m = instances[i].mask.astype(bool)
        dup = False
        for j, km in zip(kept, kept_masks):
            if not _boxes_overlap(instances[i].box, instances[j].box):
                continue
            iou, contain = _mask_overlap(m, km)
            if iou >= iou_thresh or contain >= contain_thresh:
                dup = True
                break
        if not dup:
            kept.append(i)
            kept_masks.append(m)
    kept.sort()                                        # back to emission order for a stable result
    return [instances[i] for i in kept], len(instances) - len(kept)


def cascade(
    backbone,
    image: np.ndarray,
    exemplar_masks: list[np.ndarray],
    negative_masks: list[np.ndarray] | None = None,
    config: Config | dict | None = None,
    *,
    exemplar_image: np.ndarray | None = None,
    exemplar_images: list[np.ndarray] | None = None,
    extractor=None,
    reid_scorer=None,
    confidence_scorer=None,
    gt_foreground: np.ndarray | None = None,
    observer=None,
) -> tuple[list[Instance], Stats]:
    """Discover instances of the exemplar class by recursive, batched zoom-in.

    Parameters
    ----------
    config:
        A :class:`~foveate.config.Config` (or dict) holding every tunable knob.
    exemplar_image:
        If given, the exemplar lives in *this* image while ``image`` is a **different target**
        (the in-context / cross-image setting). The foreground extractor's reference is built
        from ``exemplar_image`` and every target crop is matched against it. ``None`` (default)
        = intra-image (reference and targets share the image). Cross-image matching carries a
        DINOv3 positional bias — enable ``config.debias`` to correct it.
    exemplar_images:
        Multi-image exemplars: a list **parallel to** ``exemplar_masks`` giving the image each
        mask lives on (each exemplar is cropped from its own image; the reference banks are
        stacked across images). Mutually exclusive with ``exemplar_image``; implies the
        cross-image setting (enable ``config.debias``).
    extractor:
        Optional pre-built foreground extractor with its reference already set (see
        :func:`foveate.foreground.build_extractor` + ``set_reference``). When the same exemplar
        bank is reused across many target images — the cross-image (inter) protocol, where the
        support is identical for every target of a class — building it once and passing it in
        skips re-embedding every exemplar crop per image. ``None`` (default) builds and sets the
        reference here, as before. The caller is responsible for passing an extractor whose
        reference matches ``exemplar_image`` / ``exemplar_images``.
    reid_scorer:
        Optional pre-built re-identification scorer (see :func:`foveate.reid.build_reid_scorer`),
        the standalone ``g`` over the exemplar bank. Like ``extractor`` it can be built once and
        reused across the target images of one class (inter protocol) so the exemplars are embedded
        once. ``None`` (default) builds it here — reusing the extractor's exemplar CLS for free in
        ``reid_mode="cls"``, embedding the exemplars once for ``"masked"``.
    gt_foreground:
        Optional ``(H, W)`` ground-truth class foreground of ``image`` — a bool union mask or an int
        instance-label map (``0`` = bg, ``i`` = the ``i``-th GT instance). Only the ``"oracle"`` /
        ``"oracle_cc"`` foreground extractors consume it (the upper-bound *Where* ablation: perfect
        foreground, everything else the real cascade — ``oracle_cc`` additionally uses the per-instance
        ids to pre-separate touching instances); other extractors ignore it. Injected on the extractor
        once here, before any crop is predicted.
    observer:
        Optional ``callable(info: dict)`` invoked once per processed region for tracing.
    """
    cfg = config if isinstance(config, Config) else Config.from_dict(config)

    H, W = image.shape[:2]
    if exemplar_images is not None and exemplar_image is not None:
        raise ValueError("Pass either exemplar_image or exemplar_images, not both.")
    multi = exemplar_images is not None
    same_image = exemplar_image is None and not multi
    leaves: list[Instance] = []
    stats = Stats()

    # Foreground extractor: set the reference (image(s) + exemplar masks) once, then predict the
    # class region on every target crop. This is the INSID3 reference-at-gating-time flow.
    # A caller reusing one bank across images (inter) may hand in a pre-built extractor, so the
    # exemplar crops are embedded once, not per target image.
    ref_image = exemplar_images if multi else (image if same_image else exemplar_image)
    if extractor is None:
        extractor = build_extractor(cfg)
        extractor.set_reference(backbone, ref_image, exemplar_masks, negative_masks, cfg)
        stats.n_embeds += 1
    # Extract (slot 2) and Stop (slot 3) — the other two swappable stages. Both are built per call
    # (they are cheap, config-only objects) and hold this image's transient oracle state, if any.
    instance_extractor = build_instance_extractor(cfg)
    stop = build_stop_rule(cfg)

    # Oracle ablations: hand the target's GT to whichever slot asks for it (transient per-image
    # state, so a cached inter-protocol extractor is simply refreshed each image). Only the oracle
    # Where extractor / Stop rule expose these hooks; every other implementation never sees the GT.
    if gt_foreground is not None:
        if hasattr(extractor, "set_target_foreground"):
            extractor.set_target_foreground(gt_foreground)
        if hasattr(stop, "set_target_instances"):
            stop.set_target_instances(gt_foreground)
        if hasattr(instance_extractor, "set_target_instances"):
            instance_extractor.set_target_instances(gt_foreground)

    # Re-identification score g: a standalone scorer over the exemplar bank, independent of the
    # Where extractor above. "cls" reuses the extractor's already-embedded exemplar CLS for free;
    # "masked" embeds the exemplars once to get their foreground-patch prototypes. A caller may hand
    # in a pre-built scorer (inter protocol) to embed the exemplars once across many target images.
    if reid_scorer is None:
        reid_scorer = build_reid_scorer(cfg, backbone, ref_image, exemplar_masks,
                                        exemplar_cls=extractor.exemplar_cls)

    # Final leaf confidence: a SEPARATE masked scorer, independent of the recursion stop signal g.
    # When set, each emitted instance's score is the masked reid of its OWN foreground patches, so
    # two instances carved out of one crop no longer share the crop's (biased) score. ``None`` keeps
    # the legacy behaviour (the leaf inherits the recursion score passed to ``emit``).
    if confidence_scorer is None and cfg.confidence_reid_mode:
        confidence_scorer = build_reid_scorer(cfg, backbone, ref_image, exemplar_masks,
                                              mode=cfg.confidence_reid_mode)

    def emit(region, comp_grid, score, feat=None):
        y0, y1, x0, x1 = region.box
        comp_grid = np.asarray(comp_grid, dtype=bool)
        # Score this instance on its own foreground patches (masked confidence) rather than letting
        # it inherit the crop's shared recursion score. ``feat`` is the crop's patch grid, whose
        # (Hp, Wp) matches ``comp_grid`` exactly, so the mask indexes straight into it.
        if confidence_scorer is not None and feat is not None and comp_grid.any():
            score = confidence_scorer.score(feat, None, comp_grid)
        mask_local = featlib.upsample_mask(comp_grid, (x1 - x0, y1 - y0),
                                           bilinear=cfg.mask_upsample == "bilinear")
        if int(mask_local.sum()) < cfg.cascade_min_instance_area:
            stats.discarded += 1
            return
        full = np.zeros((H, W), np.uint8)
        full[y0:y1, x0:x1] = mask_local
        if cfg.mask_refine != "none":
            # Snap the patch staircase to the image's own boundaries. Done here, on the crop the
            # instance was found in, where the object is largest in pixels — the best conditions
            # this instance will ever get for a colour-based refinement.
            refined = mask_refine.refine_mask(full[y0:y1, x0:x1].astype(bool),
                                              image[y0:y1, x0:x1], cfg)
            full[y0:y1, x0:x1] = refined.astype(np.uint8)
        leaves.append(Instance(full, region.box, region.depth, score))
        stats.leaves += 1

    def emit_parent(parent: _Parent) -> None:
        """Fall back to the crop the (now worse) children were zoomed out of.

        Emitted iff the predecessor clears the class floor; a predecessor that is itself below
        ``crop_sim_floor`` is not the class → drop. The observer event is finalized by the caller
        (decision ``reid-stop``). By default the components are OR-merged into ONE instance (a
        converged ``g`` says "the object is in here"); with ``cfg.emit_components`` the merged
        foreground is instead split into connected components and each emitted separately — so a
        rejected split whose instances don't touch is recovered as several instances.

        **Confidence of split-out components.** The crop's ``g`` is biased high by the true exemplar
        it contains, so a component that is merely a *cut-off sliver* of a neighbouring instance (the
        crop clipped it at its border) must not inherit that ``g`` — score-ranked NMS would then keep
        the tiny sliver and suppress that neighbour's full detection (found in its own crop). A
        component touching the crop border is provably only partially in frame, so its confidence is
        scaled by its area fraction of the crop's dominant component; a whole (border-free) component
        keeps ``g`` unchanged, so genuine non-touching instances are unaffected.
        """
        if not stop.accept(parent.reid, parent.box):
            stats.discarded += 1
            return
        pr = _Region(parent.box, parent.depth)
        merged = np.logical_or.reduce(parent.comps)
        if cfg.emit_components:
            comps = instance_extractor.components(merged, box=parent.box)
            for c, score in zip(comps, _component_scores(comps, parent.reid, merged.shape)):
                emit(pr, c, score, feat=parent.feat)
        else:
            emit(pr, merged, parent.reid, feat=parent.feat)

    frontier = [_Region((0, H, 0, W), 0)]
    level_idx = 0
    while frontier and stats.n_embeds < cfg.max_total_embeds:
        stats.level_sizes.append(len(frontier))
        # Embed only regions without a cached lookahead embedding (clump children carry theirs).
        fresh_idx = [i for i, r in enumerate(frontier) if r.embedded is None]
        fresh = featlib.embed_batch(
            backbone, [image[frontier[i].box[0]:frontier[i].box[1],
                             frontier[i].box[2]:frontier[i].box[3]] for i in fresh_idx],
            chunk=cfg.embed_batch_size, standardize=cfg.standardize,
        )
        stats.n_embeds += len(fresh_idx)
        fresh_map = dict(zip(fresh_idx, fresh))
        embedded = [r.embedded if r.embedded is not None else fresh_map[i]
                    for i, r in enumerate(frontier)]
        # g scores the crop's EXTRACTED foreground, so when the scorer needs it (masked modes) run
        # Where predict for the whole frontier up front and reuse each result for its survivor below
        # — no wasted predict, only the losers pay extra. CLS g needs no mask → predict survivors
        # only, as before (``gates`` stays None and predict runs lazily in the survivor loop).
        gates: list = [None] * len(frontier)
        if reid_scorer.needs_foreground:
            gates = [extractor.predict(feat, cls=cls, box=frontier[i].box,
                                       return_internals=observer is not None)
                     for i, (feat, cls) in enumerate(embedded)]
        reid_scores = [reid_scorer.score(feat, cls, gates[i].foreground if gates[i] else None)
                       for i, (feat, cls) in enumerate(embedded)]

        # reid-stop: pursue children by the Stop rule (:mod:`foveate.stop`) — a single zoom child must beat
        # the crop it came from (over-zoom peak guard), while a SPLIT's children are kept whenever
        # the split is confirmed real (best child strictly beats the parent) and they clear the class
        # floor, so a novel sibling instance is not discarded just for scoring below a biased parent.
        # Group children by parent; if the rule keeps none, the parent was the peak → emit it.
        survivors: list[int] = []
        groups: dict[int, tuple[_Parent, list[int]]] = {}
        deferred_retry: list[_Region] = []                # zoom-peak split retries → next frontier
        for i, r in enumerate(frontier):
            if r.parent is None:                          # root: always pursued (no crop to beat)
                survivors.append(i)
            else:
                groups.setdefault(id(r.parent), (r.parent, []))[1].append(i)
        for parent, members in groups.values():
            keep = set(stop.survivors(parent.reid, parent.box,
                                      [reid_scores[i] for i in members],
                                      [frontier[i].box for i in members]))
            better = [members[j] for j in keep]
            dropped = [members[j] for j in range(len(members)) if j not in keep]
            retried = False
            if better:                                    # at least one child survived → keep going
                survivors.extend(better)
                stats.discarded += len(dropped)
            else:                                         # no child survived → the parent was it
                # A zoom that peaked by only a HAIR (parent beat the child by < eps) may be sitting
                # on a clump that tightening onto one component can't improve — so before emitting,
                # try ONE k=2 split of the parent. The sub-crops are g-gated next level exactly
                # like a convergence split (``max child > parent`` → pursue, else emit the parent
                # there). Only single-component (zoom) parents qualify; a retry yields a multi-child
                # split parent, which never retries again → no unbounded chain.
                if (len(members) == 1 and parent.allow_retry and instance_extractor.can_split
                        and stop.accept(parent.reid, parent.box)
                        and stop.allow_retry(parent.reid, reid_scores[members[0]])):
                    comp = parent.comps[0]
                    subs = instance_extractor.split(parent.feat, comp, box=parent.box)
                    if len(subs) >= 2:
                        # Same no-progress guard as the convergence split: a sub-box equal to
                        # the parent's box would re-embed the identical crop forever.
                        pairs = [(s, _child_box(s, parent.box, cfg.pad_frac, cfg.crop_dilate)) for s in subs]
                        pairs = [(s, b) for s, b in pairs if b != parent.box]
                        subs = [s for s, _ in pairs]
                        sub_boxes = [b for _, b in pairs]
                    if len(subs) >= 2:
                        sub_emb = featlib.embed_batch(
                            backbone, [image[b[0]:b[1], b[2]:b[3]] for b in sub_boxes],
                            chunk=cfg.embed_batch_size, standardize=cfg.standardize,
                        )
                        stats.n_embeds += len(sub_emb)
                        if parent.event is not None:              # re-label the held-back event as a
                            parent.event["decision"] = "clump-split"   # split, fired next level
                            parent.event["children"] = list(sub_boxes)
                            parent.event["instance_grids"] = list(subs)
                        pnew = _Parent(reid=parent.reid, box=parent.box, depth=parent.depth,
                                       comps=[comp], event=parent.event, feat=parent.feat,
                                       allow_retry=False)   # keep feat for confidence; no re-retry
                        deferred_retry += [_Region(cb, parent.depth + 1, embedded=ce, parent=pnew)
                                           for cb, ce in zip(sub_boxes, sub_emb)]
                        retried = True
                if not retried:
                    if parent.event is not None:
                        parent.event["decision"] = "reid-stop"
                    emit_parent(parent)
                    stats.discarded += len(members)
            # Trace the discarded children too (their CLS fell vs the parent) so the trajectory /
            # tool can *show* why the cascade stopped — a terminal ``reid-worse`` node per drop.
            # Skipped when the parent was retried: the drop is subsumed by the split, whose deferred
            # event (now ``clump-split``) is fired next level instead.
            if observer is not None and not retried:
                # Two distinct drop reasons. If the group had survivors this was a CONFIRMED split:
                # a dropped child fell below BOTH the parent and the class floor (anything that beat
                # the parent survived) and is pruned while its stronger siblings keep zooming. With
                # no survivors, no child beat the parent → the parent was the peak (``reid-worse``,
                # why the cascade stopped here).
                drop_decision = "below-floor" if better else "reid-worse"
                for i in dropped:
                    dr = frontier[i]
                    observer(dict(level=level_idx, depth=dr.depth, box=dr.box,
                                  decision=drop_decision, n_components=0, reid_score=reid_scores[i],
                                  parent_reid=parent.reid, children=[], internals={},
                                  instance_grids=[]))
            if observer is not None and parent.event is not None and not retried:
                observer(parent.event)                    # fire the deferred event, now finalized
        survivors.sort()

        nxt: list[_Region] = []
        for i in survivors:
            r = frontier[i]
            feat, cls = embedded[i]
            reid_score = reid_scores[i]
            stats.max_depth = max(stats.max_depth, r.depth)

            gr = gates[i] if gates[i] is not None else extractor.predict(
                feat, cls=cls, box=r.box, return_internals=observer is not None)
            fg = gr.foreground
            comps = instance_extractor.components(fg, box=r.box)   # Extract: candidates on the crop
            n = len(comps)

            floor = (r.box[1] - r.box[0]) <= cfg.min_crop or (r.box[3] - r.box[2]) <= cfg.min_crop
            decision, children = "empty", []
            child_embeds = None                           # cached lookahead embeds for kept clumps
            child_comps: list = []                        # foreground comps behind zoom/split kids

            if n == 0:
                pass
            elif floor:                                   # SIZE floor (px) — the only non-CLS stop
                decision = "leaf-cap"
                for c in comps:
                    emit(r, c, reid_score, feat=feat)
            elif n >= 2:                                  # multiple instances → tighter crops
                # A component whose padded bbox clips back to THIS crop's box makes no
                # geometric progress: re-enqueuing it re-embeds the identical crop, finds the
                # identical components and splits again — an infinite loop only the embed
                # budget stops (and a self-referential trace event that hangs the trajectory
                # tools). Emit such a component as-is; recurse only into genuinely tighter ones.
                boxes_all = [_child_box(c, r.box, cfg.pad_frac, cfg.crop_dilate) for c in comps]
                kept = [(c, b) for c, b in zip(comps, boxes_all) if b != r.box]
                for c, b in zip(comps, boxes_all):
                    if b == r.box:
                        emit(r, c, reid_score, feat=feat)
                if kept:
                    decision = "split"
                    child_comps = [c for c, _ in kept]
                    children = [b for _, b in kept]
                else:                                     # nothing tightenable → all emitted
                    decision = "leaf-cap"
            else:                                         # single component
                comp = comps[0]
                child = _child_box(comp, r.box, cfg.pad_frac, cfg.crop_dilate)
                if _box_area(child) / max(_box_area(r.box), 1) < cfg.shrink_stop:
                    decision, children, child_comps = "zoom", [child], [comp]  # strictly tighter
                elif not stop.accept(reid_score, r.box):    # converged, below the floor → not the class
                    if cfg.discard_rejected:
                        decision = "discard"; stats.discarded += 1
                    else:
                        decision = "leaf"; emit(r, comp, reid_score, feat=feat)
                elif not instance_extractor.can_split:
                    decision = "leaf"; emit(r, comp, reid_score, feat=feat)   # splitting disabled → whole
                else:
                    # Cannot zoom in (converged) → NEVER accept as a leaf on the spot. ALWAYS split
                    # k=2 on the foreground and enqueue the sub-crops against THIS crop as their
                    # parent: the re-id survivor rule (next level, the group loop above) keeps a
                    # sub-crop only if it re-identifies the exemplar more strongly than this crop.
                    # If neither beats it (they got worse, or the blob is unsplittable), the cascade
                    # falls back and emits this crop — the crop that led to the split — as ``reid-stop``.
                    subs = instance_extractor.split(feat, comp, box=r.box)
                    if len(subs) >= 2:
                        # Drop a sub-crop whose padded box clips back to THIS crop's box: it
                        # makes no geometric progress (identical crop → identical CLS →
                        # identical split), so pursuing it loops until the embed budget.
                        pairs = [(s, _child_box(s, r.box, cfg.pad_frac, cfg.crop_dilate)) for s in subs]
                        pairs = [(s, b) for s, b in pairs if b != r.box]
                    else:
                        pairs = []
                    if pairs:
                        subs = [s for s, _ in pairs]
                        sub_boxes = [b for _, b in pairs]
                        child_embeds = featlib.embed_batch(   # lookahead embeds, reused next level
                            backbone, [image[b[0]:b[1], b[2]:b[3]] for b in sub_boxes],
                            chunk=cfg.embed_batch_size, standardize=cfg.standardize,
                        )
                        stats.n_embeds += len(child_embeds)
                        decision, children, child_comps = "clump-split", sub_boxes, [comp]
                    else:                                 # unsplittable / no tighter sub-crop
                        decision = "leaf"; emit(r, comp, reid_score, feat=feat)

            if decision == "leaf-cap":
                instance_grids = comps
            elif decision in ("split", "zoom"):
                instance_grids = child_comps
            elif decision == "clump-split":
                instance_grids = subs
            elif decision == "leaf":
                instance_grids = [comp]
            else:                                          # empty / discard
                instance_grids = []

            ev = dict(level=level_idx, depth=r.depth, box=r.box, decision=decision,
                      n_components=int(n), reid_score=reid_score, feat=feat, fg=fg,
                      # The component *label image* is a viz-only view of ``comps`` — build it only
                      # when someone is watching (the cascade itself works on the component list).
                      comp_labels=(component_label_map(comps, fg.shape) if observer is not None
                                   else None),
                      children=list(children),
                      internals=(gr.internals if observer is not None else {}),
                      instance_grids=instance_grids)
            if child_comps:                               # zoom / split / clump-split: g-gated here
                # Defer this region's event: it becomes ``reid-stop`` if no child beats it (decided
                # next level, in the group loop above), so it is fired there with the final label.
                # A zoom parent also carries its features so a hair-thin peak can be split in place.
                par = _Parent(reid=reid_score, box=r.box, depth=r.depth, comps=child_comps, event=ev,
                              feat=feat, allow_retry=(decision == "zoom"))
                if child_embeds is not None:              # clump-split: reuse the lookahead embeds
                    nxt += [_Region(cb, r.depth + 1, embedded=ce, parent=par)
                            for cb, ce in zip(children, child_embeds)]
                else:                                     # zoom / split: re-embedded next level
                    nxt += [_Region(cb, r.depth + 1, parent=par) for cb in children]
            elif observer is not None:                    # terminal (leaf / discard / cap / empty)
                observer(ev)

        frontier = nxt + deferred_retry                   # + zoom-peak split retries from this level
        level_idx += 1

    # The embed budget (or an empty gate) can cut the loop with children still queued whose parent
    # was never finalized. Fall back to each pending predecessor — the best crop seen on its chain
    # — rather than silently dropping it, and fire its held-back event as a ``reid-stop``.
    flushed: set[int] = set()
    for r in frontier:
        p = r.parent
        if p is None or p.event is None or id(p) in flushed:
            continue
        flushed.add(id(p))
        p.event["decision"] = "reid-stop"
        emit_parent(p)
        if observer is not None:
            observer(p.event)

    # Dedup: independent branches (especially the two sub-crops of a k=2 split whose boxes nest)
    # can converge on the same object and emit it twice. A final score-ranked NMS keeps the best
    # detection of each object. Disable by setting both thresholds to 1.0.
    # Fragment merge first, suppression second: a crop boundary can cut one object into disjoint
    # pieces that NMS is blind to (IoU ~ 0, no containment), and merging them before the score-ranked
    # pass means the union — not an arbitrary piece — is what competes for survival.
    if cfg.merge_fragments:
        leaves, n_merged = _merge_fragments(leaves, cfg.merge_fragment_gap)
        stats.merged += n_merged
    if cfg.nms_iou < 1.0 or cfg.nms_containment < 1.0:
        leaves, n_suppressed = _nms(leaves, cfg.nms_iou, cfg.nms_containment)
        stats.suppressed += n_suppressed

    return leaves, stats


# Backward-compatible alias: the cascade entry point used to be ``discover_instances``.
discover_instances = cascade
