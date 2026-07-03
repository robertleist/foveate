"""Recursive instance discovery — zoom until the exemplar is re-identified.

Breadth-first, batched, connected-components fixed point. For each crop:

1. **Re-embed** (batched per level) and **extract the class foreground** with the configured
   foreground extractor (INSID3 by default, the prototype bank as an alternative) → the region
   of the crop that belongs to the exemplar concept. This answers *where* on the crop the class
   is.
2. **Connected components** propose tighter crops:
   - ≥ 2 components  → enqueue each (tighter child crops);
   - 1 component whose bbox does **not** fill the crop → enqueue the tighter crop (keep zooming);
   - 1 component whose bbox **fills** the crop → *converged* (no tighter crop extractable).

   Zooming does **not** run open-loop: a child crop is only pursued if its CLS re-identifies the
   exemplar *more strongly* than the crop it was zoomed out of. When no child of a parent beats
   the parent, the parent was the CLS peak and is emitted as the instance (if it clears
   ``cls_threshold``); when some children beat it, only those continue and the lower-similarity
   siblings are discarded. This is the ``cls-stop`` rule — it keeps the cascade from over-zooming
   past the scale at which the object is best recognized.
3. A converged crop (zoom exhausted) is **never accepted as a leaf on the spot** — it is
   *always* split k=2 on its foreground and the sub-crops are put back on the frontier against
   this crop as their parent (next level). Whether the split "took" is judged by CLS:
   - **Confirm** the split is real — its *best* sub-crop must *strictly* beat the parent CLS.
     Isolating a real object from a mixed crop raises CLS (the other instance and the background
     that diluted the parent drop away), so a genuine clump always has a sub-crop above the parent;
     a single object only yields weaker partial halves (and flat/tied CLS never beats it), so its
     split is not confirmed → fall back and emit **this crop** (``cls-stop``).
   - Once confirmed, keep **every** sub-crop that independently clears ``cls_threshold`` — each is
     its own instance and is pursued. This is the key subtlety: a crop can hold the original
     exemplar *and* a genuinely novel instance, so the parent CLS is biased high by the exemplar it
     contains; gating each sub-crop on "beat the parent" would wrongly discard the novel sibling
     (class-like, but a different instance, so scoring below that inflated parent). Confirm-then-floor
     keeps it. (A single-child *zoom*, by contrast, still uses the strict beat-the-parent peak guard.)
   The other exits: below ``config.cls_threshold`` → not the class → reject; ``split_mode="none"``
   → accepted whole (splitting disabled).

   So the only stopping signals are CLS re-identification and the size floor (``min_crop``): a crop
   at or below it is emitted without splitting further. No depth cap, no split margin.

The exemplar defines the target *scale and granularity*. Only leaves are returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label

from foveate import clustering, features as featlib, individuation, merge
from foveate.config import Config
from foveate.foreground import build_extractor
from foveate.types import Instance, Stats

_CONN8 = generate_binary_structure(2, 2)   # 8-connectivity: don't over-split single instances


@dataclass
class _Parent:
    """The crop a batch of child crops was zoomed out of.

    Kept so the cascade can fall back to it when none of its children re-identify the exemplar
    more strongly: the CLS peak was the parent, so *it* is the instance. ``comps`` are the
    parent's foreground components (patch grids); they are OR-merged into one instance on emit
    (a converged CLS says "the object is in here", so a multi-component parent is one object the
    foreground extractor happened to fragment).
    """
    cls: float                                 # parent CLS re-identification score
    box: tuple[int, int, int, int]
    depth: int
    comps: list
    # The parent's own observer event, held back until its children's fate is known: a zoom/split
    # region is only labelled ``zoom``/``split`` if a child improved on it, else ``cls-stop``. So
    # the observer sees each region once, with its FINAL decision (the trajectory tooling rebuilds
    # the tree by unique box, so a region must appear exactly once).
    event: dict | None = None
    # The parent's patch features, kept ONLY for a single-component (zoom) parent so that a zoom
    # which peaks by a hair can be split k=2 in place before being emitted (``zoom_split_retry``).
    # ``None`` once used (a retry produces a multi-child split parent, which never retries again).
    feat: object | None = None


@dataclass
class _Region:
    box: tuple[int, int, int, int]
    depth: int
    # Zoom/split children are re-embedded next level; clump-split children carry their lookahead
    # embedding so the k=2 split forward isn't paid twice over the same sub-crops.
    embedded: tuple | None = None
    # The crop this one was zoomed OR split out of. A child is only pursued if it beats
    # ``parent.cls``; if no child of a parent beats it, the parent is emitted as the instance.
    # Set for zoom, split AND clump-split children — the split is CLS-gated exactly like zoom.
    # ``None`` only for the root.
    parent: _Parent | None = None


def _grid_bbox(comp):
    rows, cols = np.where(comp)
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


def _child_box(comp, box, pad_frac):
    """Pixel bbox (in original coords) of a grid component within ``box``, padded."""
    y0, y1, x0, x1 = box
    ch, cw = y1 - y0, x1 - x0
    hp, wp = comp.shape
    rmin, rmax, cmin, cmax = _grid_bbox(comp)
    py0 = y0 + int(np.floor(rmin / hp * ch)); py1 = y0 + int(np.ceil(rmax / hp * ch))
    px0 = x0 + int(np.floor(cmin / wp * cw)); px1 = x0 + int(np.ceil(cmax / wp * cw))
    pady, padx = int(round((py1 - py0) * pad_frac)), int(round((px1 - px0) * pad_frac))
    return (max(y0, py0 - pady), min(y1, py1 + pady),
            max(x0, px0 - padx), min(x1, px1 + padx))


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


def _split_component(feat, comp_grid, cfg):
    """Split a converged clump into sub-component grids — dispatched by ``cfg.split_mode``.

    ``"kmeans"`` (default) always proposes a 2-way split so the CLS-survivor rule, not the
    splitter, decides whether to keep it; ``"watershed"`` uses the marker-controlled pipeline
    (which can return a single basin, i.e. "unsplittable").
    """
    if cfg.split_mode == "kmeans":
        return _split_kmeans(feat, comp_grid)
    return _split_watershed(feat, comp_grid, cfg)


def _split_kmeans(feat, comp_grid):
    """k=2 KMeans on the component's foreground patch features → two sub-masks.

    Unlike watershed, this *always* yields a 2-way partition when the component has >= 2 patches,
    so "always try to split" is guaranteed and the CLS-survivor rule does the accepting/rejecting.
    Features are L2-normalized, so euclidean KMeans ≈ spherical (cosine) clustering.
    """
    ys, xs = np.where(comp_grid)
    if ys.size < 2:
        return [comp_grid.astype(bool)]                      # single patch → unsplittable
    from sklearn.cluster import KMeans

    X = feat[torch.from_numpy(comp_grid).to(feat.device)].detach().cpu().numpy()  # (M, D)
    lab = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(X)
    subs = []
    for c in (0, 1):
        g = np.zeros(comp_grid.shape, dtype=bool)
        g[ys[lab == c], xs[lab == c]] = True
        if g.any():
            subs.append(g)
    return subs if len(subs) == 2 else [comp_grid.astype(bool)]


def _split_watershed(feat, comp_grid, cfg):
    """Marker-controlled watershed split for a seamless clump → sub-component grids (>= 1)."""
    labels = clustering.agglomerative_oversegment(feat, comp_grid, cfg.cluster_tau)
    indiv = individuation.individuate(
        feat, comp_grid, labels, mode=cfg.marker_mode, alpha=cfg.elevation_alpha,
        beta=cfg.elevation_beta, marker_min_distance=cfg.marker_min_distance,
        smooth_sigma=cfg.boundary_smooth_sigma,
    )
    merged = merge.merge_instances(
        feat, indiv.instances, indiv.feature_boundary,
        similarity_threshold=cfg.merge_similarity, boundary_threshold=cfg.merge_boundary,
    )
    return [merged == i for i in np.unique(merged) if i != 0]


def _cls_survivors(parent_cls: float, child_scores) -> list[int]:
    """Indices of children that re-identify the exemplar *more strongly* than the parent.

    Strict ``>``: a child only continues zooming if it genuinely improves on the crop it came
    from. An empty result means no child beat the parent — the parent was the CLS peak, so the
    cascade stops and emits it (the ``cls-stop`` rule). Any child that ties or falls below the
    parent is dropped. This is the **single-object zoom** guard; splits use :func:`_survivors`.
    """
    return [i for i, s in enumerate(child_scores) if s > parent_cls]


def _survivors(parent_cls: float, child_scores, *, cls_threshold: float) -> list[int]:
    """Which of a parent's children to pursue — zoom and split handled differently.

    **One child (zoom)** → the over-zoom peak guard: keep it only if it strictly beats the crop it
    came from (:func:`_cls_survivors`). Descending a single object, CLS should keep rising; when it
    stops rising we have passed the peak and emit the parent.

    **Several children (a split)** → the parent CLS is a *biased baseline*. If the crop already
    contains a strong exemplar match, its CLS is pulled up by that sub-region, so gating each child
    on "beat the parent" wrongly discards a genuinely novel sibling instance that is class-like but
    (being a *different* instance) scores lower than that inflated parent. Instead:

    1. **Confirm the split is real.** Its BEST child must *strictly* beat the parent. Isolating a
       real object from a mixed crop *raises* CLS — the other instance and the background that were
       diluting the parent's CLS token drop away — so a genuine clump always has a sub-crop above
       the parent. A single object, by contrast, only yields *partial* sub-crops that score *below*
       the whole (and flat/tied CLS never beats the parent), so its split is not confirmed → keep
       nothing and the caller emits the parent. This is what stops a uniform blob over-segmenting.
    2. **Keep every child above the class floor.** Once confirmed, each child that independently
       clears ``cls_threshold`` is its own instance and is pursued — the novel sibling included,
       even though it scores below the exemplar-biased parent.
    """
    if len(child_scores) <= 1:
        return _cls_survivors(parent_cls, child_scores)
    if max(child_scores) > parent_cls:                       # isolating a real object raised CLS
        return [i for i, s in enumerate(child_scores) if s >= cls_threshold]
    return []                                                # no sub-crop beat the parent → emit it


def discover_instances(
    backbone,
    image: np.ndarray,
    exemplar_masks: list[np.ndarray],
    negative_masks: list[np.ndarray] | None = None,
    config: Config | dict | None = None,
    *,
    exemplar_image: np.ndarray | None = None,
    exemplar_images: list[np.ndarray] | None = None,
    extractor=None,
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
    if extractor is None:
        extractor = build_extractor(cfg)
        ref_image = exemplar_images if multi else (image if same_image else exemplar_image)
        extractor.set_reference(backbone, ref_image, exemplar_masks, negative_masks, cfg)
        stats.n_embeds += 1
    cls_bank = extractor.cls_bank                       # (S, D) L2-normalized exemplar CLS

    def classify(cls: torch.Tensor) -> float:
        """Mean cosine of the crop's CLS to all exemplar CLS — the "what is in the bbox" test."""
        gallery = cls_bank.to(cls.device, cls.dtype)
        return float((cls @ gallery.T).mean())

    def emit(region, comp_grid, score):
        y0, y1, x0, x1 = region.box
        mask_local = cv2.resize(comp_grid.astype(np.uint8), (x1 - x0, y1 - y0),
                                interpolation=cv2.INTER_NEAREST)
        if int(mask_local.sum()) < cfg.cascade_min_instance_area:
            stats.discarded += 1
            return
        full = np.zeros((H, W), np.uint8)
        full[y0:y1, x0:x1] = mask_local
        leaves.append(Instance(full, region.box, region.depth, score))
        stats.leaves += 1

    def emit_parent(parent: _Parent) -> None:
        """Fall back to the crop the (now worse) children were zoomed out of.

        Emitted as ONE instance (components OR-merged) iff the predecessor clears the class
        floor; a predecessor that is itself below ``cls_threshold`` is not the class → drop.
        The observer event is finalized by the caller (decision ``cls-stop``).
        """
        if parent.cls < cfg.cls_threshold:
            stats.discarded += 1
            return
        pr = _Region(parent.box, parent.depth)
        emit(pr, np.logical_or.reduce(parent.comps), parent.cls)

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
        cls_scores = [classify(cls) for _, cls in embedded]

        # CLS-stop: pursue children by the :func:`_survivors` rule — a single zoom child must beat
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
            keep = set(_survivors(parent.cls, [cls_scores[i] for i in members],
                                  cls_threshold=cfg.cls_threshold))
            better = [members[j] for j in keep]
            dropped = [members[j] for j in range(len(members)) if j not in keep]
            retried = False
            if better:                                    # at least one child survived → keep going
                survivors.extend(better)
                stats.discarded += len(dropped)
            else:                                         # no child survived → the parent was it
                # A zoom that peaked by only a HAIR (parent beat the child by < eps) may be sitting
                # on a clump that tightening onto one component can't improve — so before emitting,
                # try ONE k=2 split of the parent. The sub-crops are CLS-gated next level exactly
                # like a convergence split (``max child > parent`` → pursue, else emit the parent
                # there). Only single-component (zoom) parents qualify; a retry yields a multi-child
                # split parent, which never retries again → no unbounded chain.
                if (len(members) == 1 and parent.feat is not None and cfg.split_mode != "none"
                        and cfg.zoom_split_retry_eps > 0 and parent.cls >= cfg.cls_threshold
                        and 0.0 <= parent.cls - cls_scores[members[0]] < cfg.zoom_split_retry_eps):
                    comp = parent.comps[0]
                    subs = _split_component(parent.feat, comp, cfg)
                    if len(subs) >= 2:
                        sub_boxes = [_child_box(s, parent.box, cfg.pad_frac) for s in subs]
                        sub_emb = featlib.embed_batch(
                            backbone, [image[b[0]:b[1], b[2]:b[3]] for b in sub_boxes],
                            chunk=cfg.embed_batch_size, standardize=cfg.standardize,
                        )
                        stats.n_embeds += len(sub_emb)
                        if parent.event is not None:              # re-label the held-back event as a
                            parent.event["decision"] = "clump-split"   # split, fired next level
                            parent.event["children"] = list(sub_boxes)
                            parent.event["instance_grids"] = list(subs)
                        pnew = _Parent(cls=parent.cls, box=parent.box, depth=parent.depth,
                                       comps=[comp], event=parent.event)   # feat=None: no re-retry
                        deferred_retry += [_Region(cb, parent.depth + 1, embedded=ce, parent=pnew)
                                           for cb, ce in zip(sub_boxes, sub_emb)]
                        retried = True
                if not retried:
                    if parent.event is not None:
                        parent.event["decision"] = "cls-stop"
                    emit_parent(parent)
                    stats.discarded += len(members)
            # Trace the discarded children too (their CLS fell vs the parent) so the trajectory /
            # tool can *show* why the cascade stopped — a terminal ``cls-worse`` node per drop.
            # Skipped when the parent was retried: the drop is subsumed by the split, whose deferred
            # event (now ``clump-split``) is fired next level instead.
            if observer is not None and not retried:
                for i in dropped:
                    dr = frontier[i]
                    observer(dict(level=level_idx, depth=dr.depth, box=dr.box,
                                  decision="cls-worse", n_components=0, cls_score=cls_scores[i],
                                  parent_cls=parent.cls, children=[], internals={},
                                  instance_grids=[]))
            if observer is not None and parent.event is not None and not retried:
                observer(parent.event)                    # fire the deferred event, now finalized
        survivors.sort()

        nxt: list[_Region] = []
        for i in survivors:
            r = frontier[i]
            feat, cls = embedded[i]
            cls_score = cls_scores[i]
            stats.max_depth = max(stats.max_depth, r.depth)

            gr = extractor.predict(feat, cls=cls, return_internals=observer is not None)
            fg = gr.foreground
            labels, n = (label(fg, structure=_CONN8) if fg.any()
                         else (np.zeros_like(fg, dtype=int), 0))

            floor = (r.box[1] - r.box[0]) <= cfg.min_crop or (r.box[3] - r.box[2]) <= cfg.min_crop
            decision, children = "empty", []
            child_embeds = None                           # cached lookahead embeds for kept clumps
            child_comps: list = []                        # foreground comps behind zoom/split kids

            if n == 0:
                pass
            elif floor:                                   # SIZE floor (px) — the only non-CLS stop
                decision = "leaf-cap"
                for cid in range(1, n + 1):
                    emit(r, labels == cid, cls_score)
            elif n >= 2:                                  # multiple instances → tighter crops
                decision = "split"
                child_comps = [labels == cid for cid in range(1, n + 1)]
                children = [_child_box(c, r.box, cfg.pad_frac) for c in child_comps]
            else:                                         # single component
                comp = labels == 1
                child = _child_box(comp, r.box, cfg.pad_frac)
                if _box_area(child) / max(_box_area(r.box), 1) < cfg.shrink_stop:
                    decision, children, child_comps = "zoom", [child], [comp]  # strictly tighter
                elif cls_score < cfg.cls_threshold:       # converged, below the floor → not the class
                    if cfg.discard_rejected:
                        decision = "discard"; stats.discarded += 1
                    else:
                        decision = "leaf"; emit(r, comp, cls_score)
                elif cfg.split_mode == "none":
                    decision = "leaf"; emit(r, comp, cls_score)   # splitting disabled → whole
                else:
                    # Cannot zoom in (converged) → NEVER accept as a leaf on the spot. ALWAYS split
                    # k=2 on the foreground and enqueue the sub-crops against THIS crop as their
                    # parent: the CLS-survivor rule (next level, the group loop above) keeps a
                    # sub-crop only if it re-identifies the exemplar more strongly than this crop.
                    # If neither beats it (they got worse, or the blob is unsplittable), the cascade
                    # falls back and emits this crop — the crop that led to the split — as ``cls-stop``.
                    subs = _split_component(feat, comp, cfg)
                    if len(subs) >= 2:
                        sub_boxes = [_child_box(s, r.box, cfg.pad_frac) for s in subs]
                        child_embeds = featlib.embed_batch(   # lookahead embeds, reused next level
                            backbone, [image[b[0]:b[1], b[2]:b[3]] for b in sub_boxes],
                            chunk=cfg.embed_batch_size, standardize=cfg.standardize,
                        )
                        stats.n_embeds += len(child_embeds)
                        decision, children, child_comps = "clump-split", sub_boxes, [comp]
                    else:                                 # unsplittable (one patch) → this crop is it
                        decision = "leaf"; emit(r, comp, cls_score)

            if decision == "leaf-cap":
                instance_grids = [labels == cid for cid in range(1, n + 1)]
            elif decision in ("split", "zoom"):
                instance_grids = child_comps
            elif decision == "clump-split":
                instance_grids = subs
            elif decision == "leaf":
                instance_grids = [comp]
            else:                                          # empty / discard
                instance_grids = []

            ev = dict(level=level_idx, depth=r.depth, box=r.box, decision=decision,
                      n_components=int(n), cls_score=cls_score, feat=feat, fg=fg,
                      comp_labels=labels, children=list(children),
                      internals=(gr.internals if observer is not None else {}),
                      instance_grids=instance_grids)
            if child_comps:                               # zoom / split / clump-split: CLS-gated here
                # Defer this region's event: it becomes ``cls-stop`` if no child beats it (decided
                # next level, in the group loop above), so it is fired there with the final label.
                # A zoom parent also carries its features so a hair-thin peak can be split in place.
                par = _Parent(cls=cls_score, box=r.box, depth=r.depth, comps=child_comps, event=ev,
                              feat=(feat if decision == "zoom" else None))
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
    # — rather than silently dropping it, and fire its held-back event as a ``cls-stop``.
    flushed: set[int] = set()
    for r in frontier:
        p = r.parent
        if p is None or p.event is None or id(p) in flushed:
            continue
        flushed.add(id(p))
        p.event["decision"] = "cls-stop"
        emit_parent(p)
        if observer is not None:
            observer(p.event)

    # Dedup: independent branches (especially the two sub-crops of a k=2 split whose boxes nest)
    # can converge on the same object and emit it twice. A final score-ranked NMS keeps the best
    # detection of each object. Disable by setting both thresholds to 1.0.
    if cfg.nms_iou < 1.0 or cfg.nms_containment < 1.0:
        leaves, n_suppressed = _nms(leaves, cfg.nms_iou, cfg.nms_containment)
        stats.suppressed += n_suppressed

    return leaves, stats
