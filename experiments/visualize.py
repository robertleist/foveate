"""Render per-image debug overlays for a foveate evaluation.

For in-context instance discovery the single most informative view is the triptych

    prompt (exemplar masks)  ->  ground truth  ->  prediction

side by side on the same image. It surfaces the common failure modes at a glance: a
misaligned/empty exemplar prompt (poisons everything downstream), the cascade collapsing
many instances into one blob (under-segmentation), spraying spurious fragments
(over-segmentation), or systematically low re-id scores.

Pure matplotlib (Agg backend, no display needed). :func:`save_item_overlay` writes one PNG
per image into a directory; :mod:`experiments.run` logs that directory to MLflow as artifacts,
so the panels show up in the run's *Artifacts* tab — and are equally viewable on disk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never to a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.datasets import EvalItem  # noqa: E402
from experiments.eval import ImagePrediction, iou_matrix  # noqa: E402

# A fixed, high-contrast palette so the same instance index keeps its colour across panels.
_PALETTE = np.array(
    [
        [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
        [255, 127, 0], [255, 215, 0], [166, 86, 40], [247, 129, 191],
        [0, 206, 209], [153, 153, 153],
    ],
    dtype=np.float64,
)


def _as_stack(masks) -> np.ndarray:
    """Coerce masks to a ``(N, H, W)`` bool stack (accepts a list or a 2-D/3-D array)."""
    if masks is None:
        return np.zeros((0, 0, 0), dtype=bool)
    if isinstance(masks, (list, tuple)):
        return np.stack([np.asarray(m, dtype=bool) for m in masks]) if masks else np.zeros((0, 0, 0), bool)
    arr = np.asarray(masks)
    if arr.ndim == 2:
        arr = arr[None]
    return arr.astype(bool)


def _overlay(image: np.ndarray, masks: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend each mask onto ``image`` (HxWx3 uint8) with a distinct palette colour + outline."""
    base = image.astype(np.float64)
    if base.ndim == 2:
        base = np.repeat(base[:, :, None], 3, axis=2)
    out = base.copy()
    for i, m in enumerate(masks):
        if not m.any():
            continue
        colour = _PALETTE[i % len(_PALETTE)]
        out[m] = (1 - alpha) * out[m] + alpha * colour
        # 1-px outline: boundary = mask minus its erosion (cheap, dependency-free).
        edge = m & ~(
            np.pad(m, ((1, 0), (0, 0)))[:-1] & np.pad(m, ((0, 1), (0, 0)))[1:]
            & np.pad(m, ((0, 0), (1, 0)))[:, :-1] & np.pad(m, ((0, 0), (0, 1)))[:, 1:]
        )
        out[edge] = colour
    return out.clip(0, 255).astype(np.uint8)


def render_item(item: EvalItem, pred: ImagePrediction, *, max_scores: int = 8):
    """Build a 3-panel debug figure (prompt | ground truth | prediction) for one item."""
    gt = _as_stack(item.gt_masks)
    pred_masks = _as_stack(pred.masks)
    scores = np.asarray(pred.scores, dtype=np.float64).ravel()

    # Prompt panel: cross-image exemplars live on `exemplar_image`; intra prompts on `image`.
    prompt_image = item.exemplar_image if item.exemplar_image is not None else item.image
    exemplars = _as_stack(item.exemplar_masks)

    # Best IoU per prediction against GT — quick read on whether masks actually land on objects.
    iou = iou_matrix(pred_masks, gt) if pred_masks.shape[0] and gt.shape[0] else None
    best_iou = iou.max(axis=1) if iou is not None and iou.size else np.array([])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    kind = "inter" if item.exemplar_image is not None else "intra"
    fig.suptitle(f"{item.image_id}  [{kind}, class {item.class_id}]", fontsize=13)

    axes[0].imshow(_overlay(prompt_image, exemplars))
    axes[0].set_title(f"prompt: {exemplars.shape[0]} exemplar(s)")

    axes[1].imshow(_overlay(item.image, gt))
    axes[1].set_title(f"ground truth: {gt.shape[0]} instance(s)")

    axes[2].imshow(_overlay(item.image, pred_masks))
    order = np.argsort(-scores) if scores.size else np.array([], dtype=int)
    shown = ", ".join(f"{scores[i]:.2f}" for i in order[:max_scores])
    extra = "…" if order.size > max_scores else ""
    miou = f"  best-IoU≤{best_iou.max():.2f}" if best_iou.size else ""
    axes[2].set_title(f"predicted: {pred_masks.shape[0]}  scores=[{shown}{extra}]{miou}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def save_item_overlay(item: EvalItem, pred: ImagePrediction, out_dir: Path) -> Path:
    """Render :func:`render_item` and save it as ``<out_dir>/<image_id>.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = render_item(item, pred)
    path = out_dir / f"{item.image_id}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Cascade trace: one panel per region the recursive zoom processed.
# ---------------------------------------------------------------------------
# discover_instances(..., observer=cb) calls back once per region with a dict holding the
# crop ``box``, the gate ``fg`` (patch grid), the connected-component ``comp_labels`` (patch
# grid), the ``decision`` taken, ``cls_score``, and the child ``children`` boxes it enqueued.
# Collect those dicts and this turns them into a contact sheet of the foveation process.

# What each decision means, for the panel titles.
_DECISIONS = {
    "empty": "gate fired on nothing",
    "split": "≥2 components → zoom each",
    "zoom": "1 component, still shrinking → zoom",
    "leaf": "converged → accepted instance",
    "leaf-cap": "depth/min-crop cap → emit components",
    "clump-split": "rejected clump → watershed split",
    "discard": "converged but rejected → discarded",
}


def _upsample(grid: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upsample a patch-grid array to pixel ``(h, w)`` without cv2/PIL."""
    h, w = hw
    gh, gw = grid.shape
    ys = np.minimum((np.arange(h) * gh // max(h, 1)), gh - 1)
    xs = np.minimum((np.arange(w) * gw // max(w, 1)), gw - 1)
    return grid[ys][:, xs]


def render_cascade_trace(item: EvalItem, events: list[dict], *, max_panels: int = 24):
    """Contact sheet of the recursive zoom: each crop with its gate + components + children.

    Reads the dicts emitted by ``discover_instances``'s ``observer`` hook. Bugs that live in
    the cascade rather than the final masks show up here: the root gate firing on background
    or missing objects, single instances repeatedly re-zoomed, real instances discarded as
    rejected clumps, or the budget exhausting before the frontier drains.
    """
    import matplotlib.patches as mpatches

    events = events[:max_panels]
    n = len(events)
    ncols = min(4, n) or 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.9 * nrows), squeeze=False)
    fig.suptitle(f"{item.image_id}  cascade trace ({n} regions shown)", fontsize=13)

    for k, ax in enumerate(axes.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        if k >= n:
            ax.axis("off")
            continue
        ev = events[k]
        y0, y1, x0, x1 = ev["box"]
        crop = item.image[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]

        # Gate foreground tinted red over the crop.
        fg = ev.get("fg")
        view = crop.astype(np.float64)
        if view.ndim == 2:
            view = np.repeat(view[:, :, None], 3, axis=2)
        if fg is not None and np.asarray(fg).size:
            fg_px = _upsample(np.asarray(fg, dtype=bool), (ch, cw))
            view[fg_px] = 0.5 * view[fg_px] + 0.5 * np.array([255.0, 0.0, 0.0])
        ax.imshow(view.clip(0, 255).astype(np.uint8))

        # Child crops the region enqueued (drawn relative to this crop).
        for (cy0, cy1, cx0, cx1) in ev.get("children", []):
            ax.add_patch(mpatches.Rectangle(
                (cx0 - x0, cy0 - y0), cx1 - cx0, cy1 - cy0,
                fill=False, edgecolor="yellow", linewidth=1.5))

        decision = ev.get("decision", "?")
        ax.set_title(
            f"L{ev.get('level')} d{ev.get('depth')}  {decision}\n"
            f"n={ev.get('n_components')}  cls={ev.get('cls_score', float('nan')):.2f}\n"
            f"{_DECISIONS.get(decision, '')}",
            fontsize=8.5)

    fig.tight_layout()
    return fig


def save_cascade_trace(item: EvalItem, events: list[dict], out_dir: Path) -> Path:
    """Render :func:`render_cascade_trace` and save it as ``<out_dir>/<image_id>.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = render_cascade_trace(item, events)
    path = out_dir / f"{item.image_id}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
