"""GT-aware over/under-zoom diagnostics for the cascade (roadmap §A1.1).

:mod:`experiments.trajectories` classifies a zoom chain by the *shape of its ``g`` curve*
(``rising`` / ``peaked`` / ``overshoot``) — a **self-referential** read: it says where the cascade
*thought* the peak was, never whether that crop actually framed an object. This module answers the
same questions against the **ground truth**, so "``g`` peaked here" can be scored against "the
instance was whole here".

Three failure modes, one number each:

``overzoom_rate``
    A crop was **emitted** whose own dominant GT instance is **clipped** by its box — the padded
    foreground bbox came in tighter than the object, so the emitted mask can only be partial.
    This is the GT-side counterpart of the tracer's ``overshoot``.

``underzoom_rate``
    A crop was **emitted** that still holds **≥ 2 GT instances** (each ≥ ``cover_eps`` covered) —
    the clump was never broken, so those instances are fused into one detection. GT-side
    counterpart of ``rising``.

``slow_zoom_rate``
    A zoom step that *did* tighten but by almost nothing (child/parent area ratio above
    ``slow_ratio``). Purely a **cost** signal: those forwards bought no isolation.

Plus the plot §A1.1 actually asks for: **foreground precision / recall binned by crop size**,
measured in units of the backbone's input side (``image_size``). If precision climbs and recall
falls as crops shrink, the Where step is over-tight at leaf scale and over-zoom is *mechanical*,
not incidental — which is the whole hypothesis behind §A1.2.

What counts as "the class" here
-------------------------------
The union of ``gt_masks`` **and** ``exemplar_masks``. The Where step and the zoom are *class*-level
operations: a crop that descends onto a prompt instance is doing its job, and its foreground
covering that prompt is not a false positive. Only the eval metrics (:mod:`experiments.eval`) care
about the PU known/unknown split; these diagnostics deliberately do not.

Usage (the trace is the same observer stream the tracer consumes)::

    trace: list[dict] = []
    pred = method.predict(item, observer=trace.append)
    diag = zoom_diagnostics(trace, item.gt_masks, item.exemplar_masks, item.image.shape[:2])
    print(diag.report())
    mlflow.log_metrics(diag.to_metrics("intra"))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Decisions whose region is *emitted* as one or more instances — where over/under-zoom is scored.
#: (``reid-stop`` = the peak guard fell back to this crop; ``leaf-cap`` = the size floor stopped it.)
_TERMINAL = frozenset({"leaf", "leaf-cap", "reid-stop"})
#: Decisions that enqueue tighter children — where the *cost* of a step is scored.
_DESCEND = frozenset({"zoom", "split", "clump-split"})

#: Crop-size bins, in multiples of the backbone input side (``image_size``). Above 1x a crop is
#: **downsampled** into the encoder, so zooming still buys pixels; below 1x it is upsampled and
#: zoom buys only framing (roadmap §0, §A1.4).
_BINS = ((4.0, np.inf, "ge4x"), (2.0, 4.0, "2-4x"), (1.0, 2.0, "1-2x"),
         (0.5, 1.0, "0.5-1x"), (0.0, 0.5, "lt0.5x"))


def _bin_name(side: float, ref: float) -> str:
    r = side / max(ref, 1e-9)
    for lo, hi, name in _BINS:
        if lo <= r < hi:
            return name
    return _BINS[0][2]


def _box_area(box) -> int:
    y0, y1, x0, x1 = box
    return max(0, y1 - y0) * max(0, x1 - x0)


def _inter_area(a, b) -> int:
    """Intersection area of two ``(y0, y1, x0, x1)`` boxes."""
    y0 = max(a[0], b[0]); y1 = min(a[1], b[1])
    x0 = max(a[2], b[2]); x1 = min(a[3], b[3])
    return max(0, y1 - y0) * max(0, x1 - x0)


def _mask_boxes(masks) -> list[tuple[int, int, int, int]]:
    """Tight ``(y0, y1, x0, x1)`` box per mask; empty masks are dropped."""
    out = []
    for m in masks:
        m = np.asarray(m, dtype=bool)
        if not m.any():
            continue
        ys, xs = np.where(m)
        out.append((int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1))
    return out


def _union_at(masks, hw, work_side: int):
    """Union of ``masks`` downsampled so its longest side is ``work_side`` → ``(grid, scale)``.

    The foreground grids are 48x48-ish, so pixel-exact GT is wasted work; one downsample up front
    turns every per-event crop+resize into a sub-megapixel operation.
    """
    h, w = hw
    scale = min(1.0, work_side / max(h, w))
    gh, gw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    union = np.zeros((gh, gw), dtype=bool)
    for m in masks:
        m = np.asarray(m, dtype=bool)
        if not m.any():
            continue
        union |= _resize_bool(m, (gh, gw))
    return union, (gh / h, gw / w)


def _resize_bool(mask: np.ndarray, shape) -> np.ndarray:
    """Nearest-neighbour resize of a bool mask to ``shape`` (no OpenCV dependency)."""
    gh, gw = shape
    h, w = mask.shape
    if (h, w) == (gh, gw):
        return np.asarray(mask, dtype=bool)
    rows = np.clip((np.arange(gh) + 0.5) * h / gh, 0, h - 1).astype(np.intp)
    cols = np.clip((np.arange(gw) + 0.5) * w / gw, 0, w - 1).astype(np.intp)
    return np.asarray(mask, dtype=bool)[rows[:, None], cols[None, :]]


@dataclass
class _Acc:
    """Micro-averaged foreground precision/recall accumulator for one crop-size bin."""

    n: int = 0
    inter: float = 0.0
    fg: float = 0.0
    gt: float = 0.0

    def add(self, fg: np.ndarray, gt: np.ndarray) -> None:
        self.n += 1
        self.inter += float(np.count_nonzero(fg & gt))
        self.fg += float(np.count_nonzero(fg))
        self.gt += float(np.count_nonzero(gt))

    @property
    def precision(self) -> float:
        return self.inter / self.fg if self.fg else float("nan")

    @property
    def recall(self) -> float:
        return self.inter / self.gt if self.gt else float("nan")


@dataclass
class ZoomDiagnostics:
    """Per-image (or pooled) GT-aware read-out of the cascade's zoom behaviour."""

    n_events: int = 0
    n_terminal: int = 0
    n_descend: int = 0
    n_overzoom: int = 0
    n_underzoom: int = 0
    n_slow_zoom: int = 0
    #: Retained fraction of the dominant GT instance, per terminal crop (1.0 = whole).
    retained: list[float] = field(default_factory=list)
    #: Longest side (px) of every emitted crop — the §A1.4 "what does zoom buy" histogram.
    emit_sides: list[float] = field(default_factory=list)
    bins: dict[str, _Acc] = field(default_factory=dict)
    image_size: float = 768.0

    # -- rates ---------------------------------------------------------------
    @property
    def overzoom_rate(self) -> float:
        return self.n_overzoom / self.n_terminal if self.n_terminal else float("nan")

    @property
    def underzoom_rate(self) -> float:
        return self.n_underzoom / self.n_terminal if self.n_terminal else float("nan")

    @property
    def slow_zoom_rate(self) -> float:
        return self.n_slow_zoom / self.n_descend if self.n_descend else float("nan")

    @property
    def mean_retained(self) -> float:
        return float(np.mean(self.retained)) if self.retained else float("nan")

    def merge(self, other: "ZoomDiagnostics") -> "ZoomDiagnostics":
        """Pool another image's diagnostics into this one (rates stay micro-averaged)."""
        self.n_events += other.n_events
        self.n_terminal += other.n_terminal
        self.n_descend += other.n_descend
        self.n_overzoom += other.n_overzoom
        self.n_underzoom += other.n_underzoom
        self.n_slow_zoom += other.n_slow_zoom
        self.retained.extend(other.retained)
        self.emit_sides.extend(other.emit_sides)
        for name, acc in other.bins.items():
            mine = self.bins.setdefault(name, _Acc())
            mine.n += acc.n
            mine.inter += acc.inter
            mine.fg += acc.fg
            mine.gt += acc.gt
        return self

    # -- outputs -------------------------------------------------------------
    def to_metrics(self, prefix: str = "") -> dict[str, float]:
        """Flat ``{name: float}`` for MLflow. ``prefix`` is the eval target (``intra``/``inter``)."""
        p = f"{prefix}_" if prefix else ""
        out = {
            f"{p}zoom_overzoom_rate": self.overzoom_rate,
            f"{p}zoom_underzoom_rate": self.underzoom_rate,
            f"{p}zoom_slow_rate": self.slow_zoom_rate,
            f"{p}zoom_mean_retained": self.mean_retained,
            f"{p}zoom_n_terminal": float(self.n_terminal),
            f"{p}zoom_n_descend": float(self.n_descend),
        }
        if self.emit_sides:
            sides = np.asarray(self.emit_sides, dtype=np.float64)
            out[f"{p}zoom_emit_side_median"] = float(np.median(sides))
            out[f"{p}zoom_emit_frac_below_input"] = float((sides < self.image_size).mean())
        for _, _, name in _BINS:
            acc = self.bins.get(name)
            if acc is None or not acc.n:
                continue
            out[f"{p}zoom_fg_precision_{name}"] = acc.precision
            out[f"{p}zoom_fg_recall_{name}"] = acc.recall
            out[f"{p}zoom_n_crops_{name}"] = float(acc.n)
        return {k: float(v) for k, v in out.items()}

    def report(self) -> str:
        """Human-readable summary — the table §A1.1 asks for."""
        lines = [
            f"zoom diagnostics — {self.n_events} events "
            f"({self.n_terminal} emitted, {self.n_descend} descending)",
            f"  over-zoom  (emitted crop clips its own instance) : {self.overzoom_rate:6.1%}"
            f"   mean retained {self.mean_retained:.3f}",
            f"  under-zoom (emitted crop holds >= 2 instances)   : {self.underzoom_rate:6.1%}",
            f"  slow zoom  (tightened by almost nothing)         : {self.slow_zoom_rate:6.1%}",
            "  foreground precision / recall by crop size "
            f"(x = backbone input {self.image_size:.0f} px):",
        ]
        for _, _, name in _BINS:
            acc = self.bins.get(name)
            if acc is None or not acc.n:
                continue
            lines.append(f"    {name:>7}  n={acc.n:5d}   P={acc.precision:.3f}  R={acc.recall:.3f}")
        if self.emit_sides:
            sides = np.asarray(self.emit_sides, dtype=np.float64)
            lines.append(f"  emitted crop side: median {np.median(sides):.0f} px, "
                         f"{(sides < self.image_size).mean():.1%} below the backbone input")
        return "\n".join(lines)


def zoom_diagnostics(
    events: list[dict],
    gt_masks,
    exemplar_masks=None,
    image_hw: tuple[int, int] | None = None,
    *,
    image_size: float = 768.0,
    clip_eps: float = 0.05,
    cover_eps: float = 0.5,
    slow_ratio: float = 0.8,
    work_side: int = 1024,
) -> ZoomDiagnostics:
    """Score one image's observer ``events`` against its ground truth.

    Parameters
    ----------
    events:
        The cascade observer stream (one dict per visited region).
    gt_masks, exemplar_masks:
        ``(M, H, W)`` / list of ``(H, W)`` bool. Their **union** is "the class" (see module docstring).
    image_hw:
        Image shape; inferred from the masks when omitted.
    image_size:
        The backbone's input side — the unit the crop-size bins are expressed in.
    clip_eps:
        A terminal crop is **over-zoomed** when it retains < ``1 - clip_eps`` of its dominant instance.
    cover_eps:
        A GT instance counts as "held" by a crop when the crop covers ``>= cover_eps`` of its box.
    slow_ratio:
        A descend step is **slow** when child/parent box area exceeds this.
    """
    gt_list = [np.asarray(m, dtype=bool) for m in (gt_masks if gt_masks is not None else [])]
    ex_list = [np.asarray(m, dtype=bool) for m in (exemplar_masks or [])]
    all_masks = gt_list + ex_list
    diag = ZoomDiagnostics(image_size=float(image_size))
    if not all_masks or not events:
        return diag

    if image_hw is None:
        image_hw = all_masks[0].shape
    boxes = _mask_boxes(all_masks)                       # exact, pixel coords
    areas = [float(_box_area(b)) for b in boxes]
    union, (sy, sx) = _union_at(all_masks, image_hw, work_side)

    for ev in events:
        box = ev.get("box")
        if box is None:
            continue
        diag.n_events += 1
        decision = ev.get("decision", "")

        # -- foreground precision / recall on this crop, binned by crop size ---------------
        fg = ev.get("fg")
        if fg is not None and np.size(fg):
            fg = np.asarray(fg, dtype=bool)
            y0, y1, x0, x1 = box
            gy0, gy1 = int(round(y0 * sy)), max(int(round(y1 * sy)), int(round(y0 * sy)) + 1)
            gx0, gx1 = int(round(x0 * sx)), max(int(round(x1 * sx)), int(round(x0 * sx)) + 1)
            sub = union[gy0:gy1, gx0:gx1]
            if sub.size:
                gt_grid = _resize_bool(sub, fg.shape)
                side = float(max(y1 - y0, x1 - x0))
                diag.bins.setdefault(_bin_name(side, image_size), _Acc()).add(fg, gt_grid)

        # -- over / under-zoom on emitted crops --------------------------------------------
        if decision in _TERMINAL:
            diag.n_terminal += 1
            diag.emit_sides.append(float(max(box[1] - box[0], box[3] - box[2])))
            inters = [_inter_area(box, b) for b in boxes]
            best = int(np.argmax(inters)) if inters else -1
            if best >= 0 and inters[best] > 0:
                retained = inters[best] / max(areas[best], 1.0)
                diag.retained.append(retained)
                if retained < 1.0 - clip_eps:
                    diag.n_overzoom += 1
                held = sum(1 for i, a in zip(inters, areas) if a > 0 and i / a >= cover_eps)
                if held >= 2:
                    diag.n_underzoom += 1

        # -- cost of a descend step ---------------------------------------------------------
        elif decision in _DESCEND:
            children = ev.get("children") or []
            if not children:
                continue
            diag.n_descend += 1
            parent_area = max(_box_area(box), 1)
            if max(_box_area(c) for c in children) / parent_area > slow_ratio:
                diag.n_slow_zoom += 1

    return diag
