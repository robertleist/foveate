"""Reconstruct and plot the re-identification-score (g) trajectory of the recursive zoom.

Step 1 of the g-peak investigation — **no algorithm change**. ``cascade``
already computes the crop's re-id score g at every region (``reid_score``) and hands it to the
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
>>> fig = plot_cls_trajectories(events, crop_sim_floor=cfg.crop_sim_floor)
>>> print(trajectory_report(events))
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Decisions the cascade emits for a region that has >= 2 children (a fork). A segment ends
# at any of these; each child starts a fresh segment. Kept as data so the walk never has to
# re-derive "is this a fork" from n_components.
_FORK_DECISIONS = frozenset({"split", "leaf-cap"})


@dataclass
class TraceNode:
    """One region the cascade processed, as seen through the observer."""

    box: tuple[int, int, int, int]
    depth: int
    reid_score: float
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
        return [n.reid_score for n in self.nodes]

    @property
    def peak_cls(self) -> float:
        return self.nodes[self.peak_index].reid_score

    @property
    def stop_cls(self) -> float:
        return self.nodes[-1].reid_score

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
    """Run :func:`foveate.cascade` with a recording observer attached.

    Returns ``(instances, stats, events)``. Any ``observer=`` passed in ``kwargs`` is chained,
    so an existing trace hook still fires.
    """
    from foveate import cascade

    events, record = recording_observer()
    user_observer = kwargs.pop("observer", None)

    def observer(info):
        record(info)
        if user_observer is not None:
            user_observer(info)

    instances, stats = cascade(
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
        reid_score=float(ev.get("reid_score", float("nan"))),
        decision=ev.get("decision", "?"),
        n_components=int(ev.get("n_components", 0)),
        children=[tuple(c) for c in ev.get("children", [])],
    )


def _reconstruct(events: list[dict]) -> tuple[dict[tuple, dict], list[dict]]:
    """``(box -> event, roots)``. A child box becomes the box of its child region, so an exact
    box map rebuilds parent->child; roots are events whose box is never referenced as a child."""
    by_box: dict[tuple, dict] = {}
    for ev in events:
        by_box.setdefault(tuple(ev["box"]), ev)
    referenced = {tuple(c) for ev in events for c in ev.get("children", [])}
    roots = [ev for ev in events if tuple(ev["box"]) not in referenced]
    return by_box, roots


def build_segments(events: list[dict]) -> list[list[TraceNode]]:
    """Rebuild the zoom tree from observer events and slice it into single-component segments.

    Each segment is a maximal chain of one-child ``zoom`` steps; forks (>= 2 children) end the
    segment and seed a new one per child. For the *connected* root->leaf view use
    :func:`build_paths` instead.
    """
    by_box, roots = _reconstruct(events)

    segments: list[list[TraceNode]] = []
    # Iterative walk (explicit stack) so a deep zoom can't blow the Python recursion limit.
    stack = list(roots)
    while stack:
        ev = stack.pop()
        seg: list[TraceNode] = []
        while True:
            node = _node(ev)
            seg.append(node)
            # A degenerate child box equal to the node's own box (pre-fix cascades could emit
            # one when a split sub-box clipped back to the parent) is a self-loop in the
            # box-keyed tree — following it would walk forever.
            kids = [by_box[c] for c in node.children if c in by_box and c != node.box]
            if len(kids) == 1:
                ev = kids[0]  # stay on the same single-component chain
                continue
            segments.append(seg)
            stack.extend(kids)  # 0 -> terminal; >= 2 -> each child seeds a new segment
            break
    return segments


def build_paths(events: list[dict]) -> list[list[TraceNode]]:
    """Full **connected** root->leaf paths: every chain from the whole-image crop (depth 0) to a
    terminal region (leaf / discard / cap), forks and all. One path per leaf; shared ancestors
    are repeated in each path that passes through them. This is the view to *follow* a single
    instance's zoom from the whole image down to its crop.
    """
    by_box, roots = _reconstruct(events)
    paths: list[list[TraceNode]] = []
    stack: list[list[TraceNode]] = [[_node(r)] for r in roots]
    while stack:
        path = stack.pop()
        # Same self-loop guard as build_segments: never follow a child box equal to the
        # node's own box (pre-fix cascades could emit one; it loops forever here).
        kids = [by_box[c] for c in path[-1].children if c in by_box and c != path[-1].box]
        if not kids:
            paths.append(path)
        else:
            for k in kids:
                stack.append(path + [_node(k)])  # branch: copy prefix per child
    return paths


def _terminal_segment(path: list[TraceNode]) -> list[TraceNode]:
    """The single-component tail of a path — from the last fork's chosen child to the leaf.

    Peak / overshoot are only well posed here: across a fork the curve is confounded by how many
    instances are still in view, so the diagnosis is read off this tail, not the whole path.
    """
    last_fork = 0
    for i, n in enumerate(path[:-1]):
        if len(n.children) >= 2:
            last_fork = i + 1
    return path[last_fork:]


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
    scores = [n.reid_score for n in nodes]
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
    elif stop_decision == "reid-stop":
        # The cascade stopped here *because* the next zoom scored lower — a clean peak by design.
        kind = "peaked"
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
    "single": ("mediumpurple", "single node (leaf at fork)"),
    "rising": ("tab:orange", "rising at stop (underzoom?)"),
    "peaked": ("tab:green", "peaked at stop"),
    "overshoot": ("tab:red", "overshoot (ancestor was better)"),
    "truncated": ("tab:blue", "truncated (budget)"),
}


def plot_cls_trajectories(
    events: list[dict],
    *,
    crop_sim_floor: float | None = None,
    min_len: int = 1,
    ax=None,
    title: str | None = None,
    tol: float = 1e-3,
):
    """Plot CLS-vs-depth for every single-component segment; star the peak, flag the stop.

    Colour encodes the stop diagnosis (see ``_KIND_STYLE``). The acceptance ``crop_sim_floor``,
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

    if crop_sim_floor is not None:
        ax.axhline(crop_sim_floor, ls="--", color="0.5", lw=1,
                   label=f"crop_sim_floor={crop_sim_floor:g}")

    multi = [s for s in segs if len(s.nodes) >= 2]
    uni = sum(s.unimodal for s in multi)
    ax.set_xlabel("zoom depth")
    ax.set_ylabel("mean cos(CLS, exemplar)")
    ax.set_title(title or f"CLS trajectory per single-component segment  "
                 f"(unimodal {uni}/{len(multi)})")
    ax.grid(True, alpha=0.3)
    if seen_kinds or crop_sim_floor is not None:
        ax.legend(fontsize=8, loc="best")
    return ax.figure


# Diagnosis order so the failure modes lead the contact sheet.
_KIND_ORDER = {"overshoot": 0, "rising": 1, "truncated": 2, "peaked": 3, "single": 4}


def plot_cls_tree(
    events: list[dict],
    *,
    crop_sim_floor: float | None = None,
    ax=None,
    tol: float = 1e-3,
    title: str | None = None,
):
    """The whole zoom as **one** node-link tree: x = depth, y = CLS, branches at forks.

    Shared ancestors are drawn once, so M leaves fan out from a common trunk instead of
    producing M plots — trace any leaf by following branches back to the root. Grey edge width
    encodes how many leaves flow through it (thick trunk -> thin twigs). Each leaf's
    single-component tail is coloured by its diagnosis; ``□`` = fork, ``*`` = tail peak (red
    edge if the leaf overshot it), ``o`` = leaf.
    """
    import matplotlib.pyplot as plt

    by_box, _ = _reconstruct(events)
    nodes = {box: _node(ev) for box, ev in by_box.items()}
    paths = build_paths(events)

    # How many leaves flow through each parent->child edge (trunk thickness).
    edge_leaves: Counter = Counter()
    for path in paths:
        for a, b in zip(path, path[1:]):
            edge_leaves[(a.box, b.box)] += 1

    if ax is None:
        max_d = max((n.depth for n in nodes.values()), default=0)
        _, ax = plt.subplots(figsize=(max(7.0, 1.5 * (max_d + 1)), 6.0))

    # 1) Every edge in grey, width ~ #leaves downstream (the branching structure / trunk).
    for nd in nodes.values():
        for cb in nd.children:
            ch = nodes.get(cb)
            if ch is None:
                continue  # child enqueued but never processed (budget) — no node to draw
            lw = 0.8 + 1.3 * np.log2(1 + edge_leaves[(nd.box, ch.box)])
            ax.plot([nd.depth, ch.depth], [nd.reid_score, ch.reid_score],
                    color="0.78", lw=lw, solid_capstyle="round", zorder=1)

    # 2) Each leaf's single-component tail coloured by its diagnosis, on top of the trunk.
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for path in paths:
        tail = _terminal_segment(path)
        summ = summarize_segment(tail, tol=tol)
        counts[summ.kind] = counts.get(summ.kind, 0) + 1
        color = _KIND_STYLE.get(summ.kind, ("0.4", summ.kind))[0]
        label = None
        if summ.kind not in seen:
            label = _KIND_STYLE.get(summ.kind, (None, summ.kind))[1]
            seen.add(summ.kind)
        ax.plot([n.depth for n in tail], [n.reid_score for n in tail],
                color=color, lw=2.0, solid_capstyle="round", zorder=3, label=label)
        pk = tail[summ.peak_index]
        ax.plot(pk.depth, pk.reid_score, "*", ms=12, color=color, zorder=5,
                mec="red" if summ.overshoot > tol else "none", mew=1.2)
        leaf = path[-1]
        ax.plot(leaf.depth, leaf.reid_score, "o", ms=6.5, mfc="white", mec=color, mew=1.8, zorder=5)

    # 3) Fork nodes as open squares (where the single-component reasoning resets).
    for nd in nodes.values():
        if sum(c in nodes for c in nd.children) >= 2:
            ax.plot(nd.depth, nd.reid_score, "s", mfc="none", mec="0.35", ms=7, mew=1.3, zorder=4)

    if crop_sim_floor is not None:
        ax.axhline(crop_sim_floor, ls="--", color="0.5", lw=1, label=f"crop_sim_floor={crop_sim_floor:g}")

    kinds_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    ax.set_xlabel("zoom depth")
    ax.set_ylabel("mean cos(CLS, exemplar)")
    ax.set_title(title or f"CLS zoom tree  ·  {len(paths)} leaves  ·  {kinds_str}")
    ax.margins(x=0.04)
    ax.grid(True, alpha=0.3)
    if seen or crop_sim_floor is not None:
        ax.legend(fontsize=8, loc="best")
    return ax.figure


def plot_leaf_paths(
    events: list[dict],
    *,
    crop_sim_floor: float | None = None,
    max_panels: int = 24,
    ncols: int = 4,
    tol: float = 1e-3,
    suptitle: str | None = None,
):
    """One panel per leaf: the **connected** root->leaf CLS curve, whole image (depth 0) to crop.

    The grey line is the full path; the thick coloured line is the single-component tail where
    peak/overshoot is well defined (forks before it are drawn as dotted verticals + open
    squares, since the curve is confounded across a fork). ★ = peak of the tail (red edge if the
    leaf overshot it), ○ = leaf. Panels are ordered failure-mode first (overshoot, rising, ...).
    """
    import matplotlib.pyplot as plt

    diagnosed = [(p, summarize_segment(_terminal_segment(p), tol=tol)) for p in build_paths(events)]
    diagnosed.sort(key=lambda ps: (_KIND_ORDER.get(ps[1].kind, 9), -len(ps[0])))

    tails = [s for _, s in diagnosed]
    counts: dict[str, int] = {}
    for s in tails:
        counts[s.kind] = counts.get(s.kind, 0) + 1
    multi = [s for s in tails if len(s.nodes) >= 2]
    uni = sum(s.unimodal for s in multi)

    shown = diagnosed[:max_panels]
    n = len(shown)
    ncols = min(ncols, n) or 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), squeeze=False)

    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        path, summ = shown[k]
        depths = [nd.depth for nd in path]
        scores = [nd.reid_score for nd in path]
        color = _KIND_STYLE.get(summ.kind, ("0.4", ""))[0]

        ax.plot(depths, scores, "-o", color="0.6", ms=3, lw=1.1, zorder=2)  # full path (grey)
        tail = _terminal_segment(path)
        ax.plot([nd.depth for nd in tail], [nd.reid_score for nd in tail],
                "-o", color=color, ms=4, lw=2.2, zorder=3)                  # single-component tail
        for nd in path[:-1]:                                               # forks
            if len(nd.children) >= 2:
                ax.axvline(nd.depth, color="0.8", ls=":", lw=1, zorder=1)
                ax.plot(nd.depth, nd.reid_score, "s", mfc="none", mec="0.4", ms=8, zorder=4)
        pk_d, pk_v = tail[summ.peak_index].depth, summ.peak_cls
        ax.plot(pk_d, pk_v, "*", ms=14, color=color, zorder=5,
                mec="red" if summ.overshoot > tol else "none", mew=1.5)
        ax.plot(depths[-1], scores[-1], "o", ms=9, mfc="none", mec=color, mew=2, zorder=5)  # leaf
        if crop_sim_floor is not None:
            ax.axhline(crop_sim_floor, ls="--", color="0.5", lw=0.8)
        ax.set_title(f"leaf d{depths[-1]} · {summ.kind}\n"
                     f"stop={summ.stop_decision} · peak@d{pk_d} · over={summ.overshoot:+.2f}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    kinds_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: kv[0]))
    extra = f"  ·  showing {n}/{len(diagnosed)}" if n < len(diagnosed) else ""
    fig.suptitle((suptitle or "root→leaf CLS paths")
                 + f"  ·  {kinds_str}  ·  tail-unimodal {uni}/{len(multi)}{extra}", fontsize=11)
    fig.supxlabel("zoom depth", fontsize=9)
    fig.supylabel("mean cos(CLS, exemplar)", fontsize=9)
    fig.tight_layout()
    return fig


def save_cls_trajectory(item, events: list[dict], out_dir, *,
                        crop_sim_floor: float | None = None) -> Path:
    """Render :func:`plot_cls_tree` for one image and save ``<out_dir>/<image_id>.png``.

    Mirrors ``experiments.visualize.save_cascade_trace`` so :mod:`experiments.run` can dump the
    one-tree CLS view next to the cascade contact sheet under the same trace budget. One plot per
    image regardless of leaf count (use :func:`plot_leaf_paths` for the per-leaf faceted view).
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: render to file
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_cls_tree(events, crop_sim_floor=crop_sim_floor,
                        title=f"{item.image_id}  CLS zoom tree")
    path = out_dir / f"{item.image_id}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
