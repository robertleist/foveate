"""The Foveate cascade — recursively zoom until each instance is re-identified (paper Algorithm 1).

Breadth-first, batched fixed point over a frontier of crops. Every crop is embedded once and asked
one question (paper Sec. 3): *which instances of the concept are on this crop?* The answer is
guided by the exemplar bank, and the descent is guided by the re-identification score ``g`` (mean
crop similarity to the bank, Eq. 2), where crop similarity is the cosine of two CLS tokens (Eq. 1).

**Three swappable slots.** This module owns the *loop* — batching, geometry, budget, termination —
and delegates every *policy* to one of three strategy interfaces, each with its own registry:

===============  ==========================  =========================================
Slot             Module / config key         Question
===============  ==========================  =========================================
**Extract**      :mod:`foveate.extract`      which instances of the concept are here?
                 ``extractor``
**Stop**         :mod:`foveate.stop`         descend, emit or reject?
                 ``stop_rule``
**Merge**        :mod:`foveate.merge_rule`   how do the emitted leaves combine?
                 ``merge_rule``
===============  ==========================  =========================================

Extract used to be two slots — a *Where* stage that found the class region and an *Extract* stage
that cut it into instances. Collapsing them is what lets a segmenter (SAM, NTT, the oracle) be
plugged in whole instead of being flattened into a foreground and re-split; the factorizable case is
preserved inside :class:`~foveate.extract.CompositeExtractor` (``where`` × ``group``), so the
Where-vs-grouping ablation and every existing config survive. See roadmap §1.5.

Two things stay here on purpose: the **size floor** ρ (``min_crop``) and the strict box shrink
(``shrink_stop``, plus the no-progress guard on a child box equal to its parent's). Together they are
the termination *invariant* — the recursion is finite whatever the slots do, including an
adversarial rule. For each crop:

1. **Extract** — re-embed (batched per level) and ask the extractor for the instances of the concept
   on this crop. Each instance proposes a tighter child crop (its padded patch-interval box):
   - ≥ 2 instances → enqueue each;
   - 1 instance whose box does **not** fill the crop → enqueue it (keep zooming);
   - 1 instance whose box **fills** the crop → nothing left to frame → emit.
2. **Stop** — two conditions decide where the descent ends, and nothing else does:
   - the **fixed point**: a child crop whose extraction is exactly the one instance it was cropped
     for is where zooming stopped changing the answer → emit it. Compared in original-image
     coordinates with an IoU tolerance, because parent and child have different patch grids.
   - the **peak guard**: a child is pursued only if it re-identifies the exemplar more strongly than
     the crop it came from. When no child of a parent beats it, the parent was the ``g`` peak and is
     emitted (``reid-stop``) if it clears the crop similarity floor τ_C; when some children beat it,
     the split is confirmed and every child above the parent *or* above τ_C continues. The floor
     clause matters because a crop holding the prompt exemplar has an inflated ``g``, and gating each
     child on "beat the parent" would discard a genuinely novel sibling instance.
3. **Merge** — the emitted leaves are deduplicated by the Merge slot (NMS by default): independent
   branches converge on the same object, so one object is emitted several times at several scales.

The exemplar defines the target *scale and granularity*. Only leaves are returned (after Merge).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foveate import features as featlib, mask_refine
from foveate.config import Config
from foveate.extract import build_extractor, component_label_map
from foveate.merge_rule import build_merge_rule
from foveate.reid import build_reid_scorer
from foveate.stop import build_stop_rule
from foveate.upsample import build_mask_upsampler
from foveate.types import Instance, Stats


@dataclass
class _Parent:
    """The crop a batch of child crops was zoomed out of.

    Kept so the cascade can fall back to it when none of its children re-identify the exemplar
    more strongly: the re-id score ``g`` peaked at the parent, so *it* is the instance.
    ``instances`` are the parent's instance grids behind those children; they are OR-merged into one
    instance on emit (a converged ``g`` says "the object is in here", so a multi-instance parent is
    one object the extractor happened to fragment).
    """
    reid: float                                # parent re-identification score g(c)
    box: tuple[int, int, int, int]
    depth: int
    instances: list
    # Native crop-local pixel masks parallel to ``instances``, when the extractor produced
    # them (a segmenter arm); ``None`` falls back to upsampling the grid (ExtractResult.masks).
    masks: list | None = None
    # The parent's own observer event, held back until its children's fate is known: a zoom/split
    # region is only labelled ``zoom``/``split`` if a child improved on it, else ``reid-stop``. So
    # the observer sees each region once, with its FINAL decision (the trajectory tooling rebuilds
    # the tree by unique box, so a region must appear exactly once).
    event: dict | None = None
    # The parent's patch feature grid ``(Hp, Wp, D)``, so each emitted instance can be scored on its
    # OWN foreground for the masked confidence (see ``confidence_reid_mode``). Always set.
    feat: object | None = None


@dataclass
class _Region:
    box: tuple[int, int, int, int]
    depth: int
    # The crop this one was zoomed OR split out of. A child is only pursued if it beats
    # ``parent.reid``; if no child of a parent beats it, the parent is emitted as the instance.
    # ``None`` only for the root.
    parent: _Parent | None = None
    # The parent instance grid this crop was derived from — the answer the child is asked to
    # reproduce or improve on. Returning exactly this instance is the fixed point (foveate.stop).
    seed: np.ndarray | None = None


#: The crop derived from one instance proposal. It lives in :mod:`foveate.features` (with the rest
#: of the grid ↔ pixel conversions) because the Extract slot needs the very same computation to ask
#: whether a component still has framing to gain — two formulas for one crop is how §A0 happened.
_grid_bbox = featlib.grid_bbox
_child_box = featlib.child_box


def _component_scores(comps: list[np.ndarray], base_score: float, grid_shape) -> list[float]:
    """Per-instance confidence when a parent crop is emitted as separate instances.

    All of them share the crop's re-id score ``g`` (``base_score``), but that ``g`` is biased
    high by the true exemplar in the crop. An instance touching the crop **border** is only
    partially in frame — a cut-off sliver of a neighbouring object — so it must not inherit the
    full ``g``: score-ranked NMS would keep the tiny sliver and suppress that neighbour's full
    detection (found in its own crop). Border-touching instances are scaled by their area fraction
    of the crop's dominant one; whole (interior) instances keep ``base_score`` unchanged, so
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
        (the in-context / cross-image setting). The extractor's reference is built from
        ``exemplar_image`` and every target crop is matched against it. ``None`` (default)
        = intra-image (reference and targets share the image). Cross-image matching carries a
        DINOv3 positional bias — enable ``config.debias`` to correct it.
    exemplar_images:
        Multi-image exemplars: a list **parallel to** ``exemplar_masks`` giving the image each
        mask lives on (each exemplar is cropped from its own image; the reference banks are
        stacked across images). Mutually exclusive with ``exemplar_image``; implies the
        cross-image setting (enable ``config.debias``).
    extractor:
        Optional pre-built Extract slot with its reference already set (see
        :func:`foveate.extract.build_extractor` + ``set_reference``). When the same exemplar
        bank is reused across many target images — the cross-image (inter) protocol, where the
        support is identical for every target of a class — building it once and passing it in
        skips re-embedding every exemplar crop per image. ``None`` (default) builds and sets the
        reference here. The caller is responsible for passing an extractor whose reference matches
        ``exemplar_image`` / ``exemplar_images``.
    reid_scorer:
        Optional pre-built re-identification scorer (see :func:`foveate.reid.build_reid_scorer`),
        the standalone ``g`` over the exemplar bank. Like ``extractor`` it can be built once and
        reused across the target images of one class (inter protocol) so the exemplars are embedded
        once. ``None`` (default) builds it here — reusing the extractor's exemplar CLS for free in
        ``reid_mode="cls"``, embedding the exemplars once for ``"masked"``.
    gt_foreground:
        Optional ``(H, W)`` ground-truth class foreground of ``image`` — a bool union mask or an int
        instance-label map (``0`` = bg, ``i`` = the ``i``-th GT instance). Consumed only by the
        oracle slots (``extractor="oracle"``, ``stop_rule="oracle"``, or a composite whose Where
        stage is oracular); every other implementation never sees it. Injected on the slots once
        here, before any crop is predicted.
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

    # Extract slot: set the reference (image(s) + exemplar masks) once, then propose the concept's
    # instances on every target crop. A caller reusing one bank across images (inter) may hand in a
    # pre-built extractor, so the exemplar crops are embedded once, not per target image.
    ref_image = exemplar_images if multi else (image if same_image else exemplar_image)
    if extractor is None:
        extractor = build_extractor(cfg)
        extractor.set_reference(backbone, ref_image, exemplar_masks, negative_masks, cfg)
        stats.n_embeds += 1
    # Stop and Merge are built per call (cheap, config-only objects); Stop additionally holds this
    # image's transient oracle state, if any.
    stop = build_stop_rule(cfg)
    merge_rule = build_merge_rule(cfg)
    upsampler = build_mask_upsampler(cfg)

    # Oracle ablations: hand the target's GT to whichever slot asks for it (transient per-image
    # state, so a cached inter-protocol extractor is simply refreshed each image). Only the oracle
    # Extract / Stop / Merge / Mask implementations expose this hook; no other one sees the GT.
    if gt_foreground is not None:
        for slot in (extractor, stop, merge_rule, upsampler):
            if hasattr(slot, "set_target_instances"):
                slot.set_target_instances(gt_foreground)

    # Re-identification score g: a standalone scorer over the exemplar bank, independent of the
    # Extract slot above. "cls" reuses the extractor's already-embedded exemplar CLS for free;
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

    def emit(region, comp_grid, score, feat=None, mask=None):
        y0, y1, x0, x1 = region.box
        comp_grid = np.asarray(comp_grid, dtype=bool)
        # Score this instance on its own foreground patches (masked confidence) rather than letting
        # it inherit the crop's shared recursion score. ``feat`` is the crop's patch grid, whose
        # (Hp, Wp) matches ``comp_grid`` exactly, so the mask indexes straight into it.
        if confidence_scorer is not None and feat is not None and comp_grid.any():
            score = confidence_scorer.score(feat, None, comp_grid)
        # A segmenter arm already produced this instance at pixel resolution; re-quantizing it
        # onto the patch grid and upsampling it back is exactly the loss the two-slot contract
        # exists to avoid, so the Mask slot is bypassed when a native mask is available.
        mask_local = (np.asarray(mask, dtype=np.uint8) if mask is not None
                      else upsampler.upsample(comp_grid, region.box))
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
        (decision ``reid-stop``). By default the parent's instances are OR-merged into ONE instance
        (a converged ``g`` says "the object is in here"); with ``cfg.emit_components`` each is
        emitted separately — so a rejected split whose instances don't touch is recovered as several.

        **Confidence of separately emitted instances.** The crop's ``g`` is biased high by the true
        exemplar it contains, so one that is merely a *cut-off sliver* of a neighbour (the crop
        clipped it at its border) must not inherit that ``g`` — score-ranked NMS would then keep the
        tiny sliver and suppress that neighbour's full detection. See :func:`_component_scores`.
        """
        if not stop.accept(parent.reid, parent.box):
            stats.discarded += 1
            return
        pr = _Region(parent.box, parent.depth)
        pmasks = parent.masks or [None] * len(parent.instances)
        if cfg.emit_components:
            shape = parent.instances[0].shape
            scores = _component_scores(parent.instances, parent.reid, shape)
            for c, score, pm in zip(parent.instances, scores, pmasks):
                emit(pr, c, score, feat=parent.feat, mask=pm)
        else:
            union = (np.logical_or.reduce(pmasks)
                     if all(m is not None for m in pmasks) else None)
            emit(pr, np.logical_or.reduce(parent.instances), parent.reid,
                 feat=parent.feat, mask=union)

    def _crop(box):
        y0, y1, x0, x1 = box
        return image[y0:y1, x0:x1]

    frontier = [_Region((0, H, 0, W), 0)]
    level_idx = 0
    while frontier and stats.n_embeds < cfg.max_total_embeds:
        stats.level_sizes.append(len(frontier))
        embedded = featlib.embed_batch(
            backbone, [_crop(r.box) for r in frontier],
            chunk=cfg.embed_batch_size, standardize=cfg.standardize,
        )
        stats.n_embeds += len(frontier)
        # g scores the crop's EXTRACTED foreground, so when the scorer needs it (masked modes) run
        # Extract for the whole frontier up front and reuse each result for its survivor below — no
        # wasted work, only the losers pay extra. CLS g needs no mask → extract survivors only
        # (``results`` stays None and the survivor loop fills it in).
        results: list = [None] * len(frontier)
        if reid_scorer.needs_foreground:
            results = [extractor.extract(feat, cls=cls, box=frontier[i].box,
                                         image=_crop(frontier[i].box),
                                         return_internals=observer is not None)
                       for i, (feat, cls) in enumerate(embedded)]
        reid_scores = [reid_scorer.score(feat, cls, results[i].foreground if results[i] else None)
                       for i, (feat, cls) in enumerate(embedded)]

        # The peak guard (:mod:`foveate.stop`): a single zoom child must beat the crop it came from,
        # while a SPLIT's children are kept whenever the split is confirmed real (best child strictly
        # beats the parent) and they clear the class floor, so a novel sibling instance is not
        # discarded just for scoring below a biased parent. Group children by parent; if the rule
        # keeps none, the parent was the peak → emit it.
        survivors: list[int] = []
        groups: dict[int, tuple[_Parent, list[int]]] = {}
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
            if better:                                    # at least one child survived → keep going
                survivors.extend(better)
                stats.discarded += len(dropped)
            else:                                         # no child survived → the parent was it
                if parent.event is not None:
                    parent.event["decision"] = "reid-stop"
                emit_parent(parent)
                stats.discarded += len(members)
            # Trace the discarded children too (their g fell vs the parent) so the trajectory tool
            # can *show* why the cascade stopped — a terminal drop node per child.
            if observer is not None:
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
                if parent.event is not None:
                    observer(parent.event)                # fire the deferred event, now finalized
        survivors.sort()

        nxt: list[_Region] = []
        for i in survivors:
            r = frontier[i]
            feat, cls = embedded[i]
            reid_score = reid_scores[i]
            stats.max_depth = max(stats.max_depth, r.depth)

            res = results[i] if results[i] is not None else extractor.extract(
                feat, cls=cls, box=r.box, image=_crop(r.box),
                return_internals=observer is not None)
            fg = res.foreground
            instances = res.instances                     # Extract: the candidates on this crop
            pmasks = res.masks or [None] * len(instances)
            n = len(instances)

            floor = (r.box[1] - r.box[0]) <= cfg.min_crop or (r.box[3] - r.box[2]) <= cfg.min_crop
            decision, children = "empty", []
            child_instances: list = []                    # the instance grids behind the children
            child_masks: list = []                        # their native pixel masks, if any

            if n == 0:
                # The extractor says there is nothing here — but its parent said there was, and this
                # crop exists only because of that proposal. Falling back to the seed keeps the
                # instance instead of losing it to a disagreement between two scales. A composite
                # extractor almost never returns empty, so this never mattered before; a segmenter
                # legitimately does, and without the fallback every such branch dies silently.
                if cfg.emit_empty_seed and r.seed is not None:
                    emit(_Region(r.parent.box, r.parent.depth), r.seed, reid_score,
                         feat=r.parent.feat)
            elif floor:                                   # SIZE floor (px) — the termination invariant
                decision = "leaf-cap"
                for c, pm in zip(instances, pmasks):
                    emit(r, c, reid_score, feat=feat, mask=pm)
            else:
                # An instance whose padded bbox clips back to THIS crop's box makes no geometric
                # progress: re-enqueuing it re-embeds the identical crop, extracts the identical
                # instances and recurses again — an infinite loop only the embed budget stops (and a
                # self-referential trace event that hangs the trajectory tools).
                boxes_all = [_child_box(c, r.box, cfg.pad_frac, cfg.crop_dilate) for c in instances]
                tight = [(c, b) for c, b in zip(instances, boxes_all) if b != r.box]
                # Two ways a branch ends here rather than deeper. The FIXED POINT: re-extracting at
                # the finer scale returned exactly the instance this crop was cropped for, so zooming
                # has stopped changing the answer. The GEOMETRIC one: a single instance the crop can
                # no longer tighten onto (``shrink_stop``) — the invariant that keeps the loop finite
                # regardless of what the Stop rule says.
                fixed = (r.seed is not None
                         and stop.converged(instances, r.box, r.seed, r.parent.box))
                stuck = n == 1 and (not tight or
                                    _box_area(tight[0][1]) / max(_box_area(r.box), 1) >= cfg.shrink_stop)
                if fixed or stuck:
                    if not stop.accept(reid_score, r.box) and cfg.discard_rejected:
                        decision = "discard"              # converged but below τ_C → not the concept
                        stats.discarded += 1
                    else:
                        decision = "leaf"
                        emit(r, instances[0], reid_score, feat=feat, mask=pmasks[0])
                elif not tight:                           # several instances, none tightenable
                    decision = "leaf-cap"
                    for c, pm in zip(instances, pmasks):
                        emit(r, c, reid_score, feat=feat, mask=pm)
                else:
                    keep = [j for j, b in enumerate(boxes_all) if b != r.box]
                    for j, (c, b, pm) in enumerate(zip(instances, boxes_all, pmasks)):
                        if b == r.box:                    # this one is stuck, its siblings are not
                            emit(r, c, reid_score, feat=feat, mask=pm)
                    decision = "zoom" if len(tight) == 1 else "split"
                    child_instances = [instances[j] for j in keep]
                    child_masks = [pmasks[j] for j in keep]
                    children = [boxes_all[j] for j in keep]

            if decision == "leaf-cap":
                instance_grids = instances
            elif decision in ("split", "zoom"):
                instance_grids = child_instances
            elif decision == "leaf":
                instance_grids = [instances[0]]
            else:                                          # empty / discard
                instance_grids = []

            ev = dict(level=level_idx, depth=r.depth, box=r.box, decision=decision,
                      n_components=int(n), reid_score=reid_score, feat=feat, fg=fg,
                      # The instance *label image* is a viz-only view of ``instances`` — build it
                      # only when someone is watching (the cascade works on the list).
                      comp_labels=(component_label_map(instances, fg.shape) if observer is not None
                                   else None),
                      children=list(children),
                      internals=(res.internals if observer is not None else {}),
                      instance_grids=instance_grids)
            if child_instances:                           # zoom / split: g-gated next level
                # Defer this region's event: it becomes ``reid-stop`` if no child beats it (decided
                # next level, in the group loop above), so it is fired there with the final label.
                par = _Parent(reid=reid_score, box=r.box, depth=r.depth,
                              instances=child_instances, event=ev, feat=feat,
                              masks=(child_masks if any(m is not None for m in child_masks)
                                     else None))
                nxt += [_Region(cb, r.depth + 1, parent=par, seed=ci)
                        for cb, ci in zip(children, child_instances)]
            elif observer is not None:                    # terminal (leaf / discard / cap / empty)
                observer(ev)

        frontier = nxt
        level_idx += 1

    # The embed budget (or an empty extraction) can cut the loop with children still queued whose
    # parent was never finalized. Fall back to each pending predecessor — the best crop seen on its
    # chain — rather than silently dropping it, and fire its held-back event as a ``reid-stop``.
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

    # Merge (slot 3): independent branches — especially the sub-crops of a split, whose boxes nest —
    # converge on the same object and emit it twice. The Merge rule decides what survives.
    leaves, n_merged, n_suppressed = merge_rule.merge(leaves)
    stats.merged += n_merged
    stats.suppressed += n_suppressed

    return leaves, stats


# Backward-compatible alias: the cascade entry point used to be ``discover_instances``.
discover_instances = cascade
