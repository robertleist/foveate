"""Reconstruct and plot the CLS-similarity trajectory of the recursive zoom.

Step 1 of the CLS-peak investigation — **no algorithm change**. ``discover_instances``
already computes the crop's CLS score at every region (``cls_score``) and hands it to the
``observer`` hook, together with the crop ``box`` and the child ``children`` boxes it enqueues.
Because a child's box becomes *exactly* the box of its child region, the parent->child zoom
tree can be rebuilt from the events alone, with nothing added to the cascade.

The point is to look at the load-bearing assumption before building any peak-seeking logic:

* Along a **single-component** descent (a chain of ``zoom`` steps, never crossing a fork),
  is ``cos(CLS, exemplar)`` actually unimodal in zoom?
* At the point the cascade stops, is CLS still **rising** (underzoom — ran out of room before
  isolating the object) or already **falling** (overshoot — a shallower ancestor crop scored
  higher and should have been emitted)?

A "segment" here is a maximal single-component chain: it starts at the root or at a child of a
fork (a region with >= 2 children), and follows one-child ``zoom`` steps until it terminates or
forks again. The peak / overshoot question is only well posed *within* a segment — across a
fork the curve is confounded by how many instances are in view (see the design discussion).

Quickstart
----------
>>> from experiments.trajectories import trace_discovery, plot_cls_trajectories
>>> instances, stats, events = trace_discovery(backbone, image, exemplar_masks, config=cfg)
>>> fig = plot_cls_trajectories(events, cls_threshold=cfg.cls_threshold)
>>> print(trajectory_report(events))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Decisions the cascade emits for a region that has >= 2 children (a fork). A segment ends
# at any of these; each child starts a fresh segment. Kept as data so the walk never has to
# re-derive "is this a fork" from n_components.
_FORK_DECISIONS = frozenset({"split", "leaf-cap", "clump-split"})


@dataclass
class TraceNode:
    """One region the cascade processed, as seen through the observer."""

    box: tuple[int, int, int, int]
    depth: int
    cls_score: float
    decision: str
    n_components: int
    children: list[tuple[int, int, int, int]]


@dataclass
class SegmentSummary:
    """A single-component zoom chain plus the peak/overshoot read-out for it."""

    nodes: list[TraceNode]
    peak_index: int
    unimodal: bool
    kind: str  # "single" | "rising" | "peaked" | "overshoot" | "truncated"
    stop_decision: str

    @property
    def depths(self) -> list[int]:
        return [n.depth for n in self.nodes]

    @property
    def scores(self) -> list[float]:
        return [n.cls_score for n in self.nodes]

    @property
    def peak_cls(self) -> float:
        return self.nodes[self.peak_index].cls_score

    @property
    def stop_cls(self) -> float:
        return self.nodes[-1].cls_score

    @property
    def overshoot(self) -> float:
        """How much CLS was given up by zooming past the peak (>= 0). 0 ==> peak is the stop."""
        return self.peak_cls - self.stop_cls


# ---------------------------------------------------------------------------
# Recording helpers — friction-free capture of the observer stream.
# ---------------------------------------------------------------------------
def recording_observer() -> tuple[list[dict], callable]:
    """``events, observer = recording_observer()`` then pass ``observer=`` to the cascade."""
    events: list[dict] = []
    return events, events.append


def trace_discovery(backbone, image, exemplar_masks, **kwargs):
    """Run :func:`foveate.discover_instances` with a recording observer attached.

    Returns ``(instances, stats, events)``. Any ``observer=`` passed in ``kwargs`` is chained,
    so an existing trace hook still fires.
    """
    from foveate import discover_instances

    events, record = recording_observer()
    user_observer = kwargs.pop("observer", None)

    def observer(info):
        record(info)
        if user_observer is not None:
            user_observer(info)

    instances, stats = discover_instances(
        backbone, image, exemplar_masks, observer=observer, **kwargs
    )
    return instances, stats, events


# ---------------------------------------------------------------------------
# Tree reconstruction — parent->child by exact box match.
# ---------------------------------------------------------------------------
def _node(ev: dict) -> TraceNode:
    return TraceNode(
        box=tuple(ev["box"]),
        depth=int(ev["depth"]),
        cls_score=float(ev.get("cls_score", float("nan"))),
        decision=ev.get("decision", "?"),
        n_components=int(ev.get("n_components", 0)),
        children=[tuple(c) for c in ev.get("children", [])],
    )


def build_segments(events: list[dict]) -> list[list[TraceNode]]:
    """Rebuild the zoom tree from observer events and slice it into single-component segments.

    A child box becomes the box of its child region, so an exact box->event map reconstructs
    parent->child. Roots are events whose box is never referenced as a child. Each segment is a
    maximal chain of one-child ``zoom`` steps; forks (>= 2 children) end the segment and seed a
    new one per child.
    """
    by_box: dict[tuple, dict] = {}
    for ev in events:
        by_box.setdefault(tuple(ev["box"]), ev)
    referenced = {tuple(c) for ev in events for c in ev.get("children", [])}
    roots = [ev for ev in events if tuple(ev["box"]) not in referenced]

    segments: list[list[TraceNode]] = []
    # Iterative walk (explicit stack) so a deep zoom can't blow the Python recursion limit.
    stack = list(roots)
    while stack:
        ev = stack.pop()
        seg: list[TraceNode] = []
        while True:
            node = _node(ev)
            seg.append(node)
            kids = [by_box[c] for c in node.children if c in by_box]
            if len(kids) == 1:
                ev = kids[0]  # stay on the same single-component chain
                continue
            segments.append(seg)
            stack.extend(kids)  # 0 -> terminal; >= 2 -> each child seeds a new segment
            break
    return segments


# ---------------------------------------------------------------------------
# Per-segment analysis — the peak / overshoot / unimodality read-out.
# ---------------------------------------------------------------------------
def _is_unimodal(vals: list[float], tol: float) -> bool:
    """Rises (within ``tol``) to the max, then falls (within ``tol``). True for len <= 2."""
    if len(vals) <= 2:
        return True
    peak = int(np.argmax(vals))
    left = vals[: peak + 1]
    right = vals[peak:]
    up = all(b >= a - tol for a, b in zip(left, left[1:]))
    down = all(b <= a + tol for a, b in zip(right, right[1:]))
    return up and down


def summarize_segment(nodes: list[TraceNode], *, tol: float = 1e-3) -> SegmentSummary:
    scores = [n.cls_score for n in nodes]
    # Peak = the *deepest* crop achieving the max, so a flat/tied curve peaks at the stop and
    # is never mis-read as overshoot; only a strictly-better shallower crop counts.
    peak_index = len(scores) - 1 - int(np.argmax(scores[::-1]))
    stop_decision = nodes[-1].decision
    unimodal = _is_unimodal(scores, tol)
    overshoot = scores[peak_index] - scores[-1]

    if len(nodes) == 1:
        kind = "single"
    elif overshoot > tol:
        # CLS fell meaningfully after the peak: a shallower ancestor crop scored higher.
        kind = "overshoot"
    elif stop_decision in ("zoom", "empty"):
        # Still wanted to zoom (or gate emptied) but the chain didn't continue here: the
        # child wasn't processed (budget) — don't read it as a clean stop.
        kind = "truncated"
    elif scores[-1] >= scores[-2] - tol:
        # Peak is the stop *and* CLS was still climbing into it: ran out of room -> underzoom.
        kind = "rising"
    else:
        kind = "peaked"

    return SegmentSummary(
        nodes=nodes, peak_index=peak_index, unimodal=unimodal,
        kind=kind, stop_decision=stop_decision,
    )


def summarize_segments(events: list[dict], *, tol: float = 1e-3) -> list[SegmentSummary]:
    return [summarize_segment(seg, tol=tol) for seg in build_segments(events)]


def trajectory_report(events: list[dict], *, min_len: int = 2, tol: float = 1e-3) -> str:
    """Plain-text summary for quick notebook / console inspection (no matplotlib)."""
    segs = summarize_segments(events, tol=tol)
    multi = [s for s in segs if len(s.nodes) >= min_len]
    counts: dict[str, int] = {}
    for s in segs:
        counts[s.kind] = counts.get(s.kind, 0) + 1
    uni = sum(s.unimodal for s in multi)

    lines = [
        f"{len(segs)} segments ({len(multi)} with >= {min_len} nodes); "
        f"kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        f"unimodal: {uni}/{len(multi)} multi-node segments"
        + (f" ({uni / len(multi):.0%})" if multi else ""),
        "",
    ]
    for i, s in enumerate(multi):
        curve = " -> ".join(f"{v:.2f}" for v in s.scores)
        lines.append(
            f"  seg {i:2d}  d{s.depths[0]}..d{s.depths[-1]}  {s.kind:9s}  "
            f"stop={s.stop_decision:11s}  peak@d{s.depths[s.peak_index]}  "
            f"overshoot={s.overshoot:+.3f}  {'uni' if s.unimodal else 'MULTI'}  [{curve}]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot — CLS vs depth, one line per single-component segment.
# ---------------------------------------------------------------------------
_KIND_STYLE = {
    "single": ("0.6", "single node"),
    "rising": ("tab:orange", "rising at stop (underzoom?)"),
    "peaked": ("tab:green", "peaked at stop"),
    "overshoot": ("tab:red", "overshoot (ancestor was better)"),
    "truncated": ("tab:blue", "truncated (budget)"),
}


def plot_cls_trajectories(
    events: list[dict],
    *,
    cls_threshold: float | None = None,
    min_len: int = 1,
    ax=None,
    title: str | None = None,
    tol: float = 1e-3,
):
    """Plot CLS-vs-depth for every single-component segment; star the peak, flag the stop.

    Colour encodes the stop diagnosis (see ``_KIND_STYLE``). The acceptance ``cls_threshold``,
    if given, is drawn as a dashed reference line — leaves are only accepted above it today.
    """
    import matplotlib.pyplot as plt

    segs = [s for s in summarize_segments(events, tol=tol) if len(s.nodes) >= min_len]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    seen_kinds: set[str] = set()
    for s in segs:
        color, _ = _KIND_STYLE.get(s.kind, ("0.4", s.kind))
        label = None
        if s.kind not in seen_kinds:
            label = _KIND_STYLE.get(s.kind, (None, s.kind))[1]
            seen_kinds.add(s.kind)
        ax.plot(s.depths, s.scores, marker="o", ms=4, lw=1.4, color=color, alpha=0.85,
                label=label)
        # Star the peak; outline it red when it isn't the stop (overshoot).
        pk_d, pk_v = s.depths[s.peak_index], s.peak_cls
        ax.plot(pk_d, pk_v, marker="*", ms=13, color=color,
                mec="red" if s.overshoot > tol else "none", mew=1.5, zorder=5)

    if cls_threshold is not None:
        ax.axhline(cls_threshold, ls="--", color="0.5", lw=1,
                   label=f"cls_threshold={cls_threshold:g}")

    multi = [s for s in segs if len(s.nodes) >= 2]
    uni = sum(s.unimodal for s in multi)
    ax.set_xlabel("zoom depth")
    ax.set_ylabel("mean cos(CLS, exemplar)")
    ax.set_title(title or f"CLS trajectory per single-component segment  "
                 f"(unimodal {uni}/{len(multi)})")
    ax.grid(True, alpha=0.3)
    if seen_kinds or cls_threshold is not None:
        ax.legend(fontsize=8, loc="best")
    return ax.figure


def save_cls_trajectory(item, events: list[dict], out_dir, *,
                        cls_threshold: float | None = None) -> Path:
    """Render :func:`plot_cls_trajectories` for one image and save ``<out_dir>/<image_id>.png``.

    Mirrors ``experiments.visualize.save_cascade_trace`` so :mod:`experiments.run` can dump a
    trajectory chart next to the cascade contact sheet under the same trace budget.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: render to file
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_cls_trajectories(events, cls_threshold=cls_threshold,
                                title=f"{item.image_id}  CLS trajectory per segment")
    path = out_dir / f"{item.image_id}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
