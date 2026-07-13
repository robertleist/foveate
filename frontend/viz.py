"""Pure rendering helpers for the foveate frontend.

Dependency-light by design: only ``numpy`` + ``matplotlib`` (Agg backend) + ``PIL``.
*No* ``streamlit`` import at module top, so this module byte-compiles and imports
without streamlit or model weights installed. The Streamlit app (``frontend.app``)
imports these helpers and feeds them numpy arrays.

Several helpers are lifted/adapted from ``experiments.visualize`` (``_upsample``,
``_overlay``, ``render_cascade_trace``) so the frontend stays consistent with the
offline debug overlays.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _get_cmap(name: str):
    """matplotlib >=3.9 dropped ``cm.get_cmap``; fall back across versions."""
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm

        return cm.get_cmap(name)

# A fixed, high-contrast palette (shared with experiments.visualize) so the same
# instance / cluster index keeps its colour across panels.
_PALETTE = np.array(
    [
        [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
        [255, 127, 0], [255, 215, 0], [166, 86, 40], [247, 129, 191],
        [0, 206, 209], [153, 153, 153],
    ],
    dtype=np.float64,
)

# Decision -> short human-readable note, for cascade trace panel titles.
_DECISIONS = {
    "empty": "gate fired on nothing",
    "split": ">=2 components -> zoom each",
    "zoom": "1 component, still shrinking -> zoom",
    "leaf": "converged, unsplittable / splitting off -> accepted instance",
    "leaf-cap": "min-crop size floor -> emit components",
    "clump-split": "converged -> k=2 split, g-gated next level",
    "discard": "converged but below the crop similarity floor tau_C -> discarded",
    "reid-stop": "no split/zoom child beat this crop -> emit it",
    "reid-worse": "re-id score g dropped below the parent -> discarded (why the cascade stopped)",
    "below-floor": "split sibling below both parent and tau_C -> pruned (stronger siblings continue)",
}


# ---------------------------------------------------------------------------
# Basic grid / mask helpers.
# ---------------------------------------------------------------------------
def upsample_grid(grid: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upsample a patch-grid array to pixel ``(h, w)`` (no cv2/PIL).

    Adapted from ``experiments.visualize._upsample``. Works for any dtype and for
    2-D grids; the last axis is left untouched if the input is 3-D (e.g. an RGB grid).
    """
    grid = np.asarray(grid)
    h, w = hw
    gh, gw = grid.shape[:2]
    ys = np.minimum((np.arange(h) * gh // max(h, 1)), gh - 1)
    xs = np.minimum((np.arange(w) * gw // max(w, 1)), gw - 1)
    return grid[ys][:, xs]


def _as_rgb(image: np.ndarray) -> np.ndarray:
    """Coerce an image to ``(H, W, 3)`` float64 (greyscale -> 3 channels)."""
    base = np.asarray(image).astype(np.float64)
    if base.ndim == 2:
        base = np.repeat(base[:, :, None], 3, axis=2)
    if base.shape[2] == 4:  # drop alpha
        base = base[:, :, :3]
    # Scale [0,1] floats up to [0,255] so blends are in a single range.
    if base.max() <= 1.5:
        base = base * 255.0
    return base


def overlay_mask(
    image: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45
) -> np.ndarray:
    """Blend a single boolean ``mask`` onto ``image`` with ``color`` + a 1-px outline.

    Adapted from ``experiments.visualize._overlay`` (single-mask variant). Returns a
    ``(H, W, 3)`` uint8 RGB image.
    """
    out = _as_rgb(image)
    m = np.asarray(mask, dtype=bool)
    if m.shape != out.shape[:2]:
        m = upsample_grid(m, out.shape[:2]).astype(bool)
    colour = np.asarray(color, dtype=np.float64)
    if m.any():
        out[m] = (1 - alpha) * out[m] + alpha * colour
        # 1-px outline = mask minus its erosion (dependency-free).
        edge = m & ~(
            np.pad(m, ((1, 0), (0, 0)))[:-1] & np.pad(m, ((0, 1), (0, 0)))[1:]
            & np.pad(m, ((0, 0), (1, 0)))[:, :-1] & np.pad(m, ((0, 0), (0, 1)))[:, 1:]
        )
        out[edge] = colour
    return out.clip(0, 255).astype(np.uint8)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Colour an integer label grid with the repeating palette -> ``(H, W, 3)`` uint8.

    Label ``0`` is treated as background and rendered black. Used for cluster maps.
    """
    labels = np.asarray(labels).astype(int)
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    uniq = [u for u in np.unique(labels) if u != 0]
    for k, u in enumerate(uniq):
        out[labels == u] = _PALETTE[k % len(_PALETTE)].astype(np.uint8)
    return out


def heatmap(values: np.ndarray, hw: tuple[int, int], cmap: str = "magma") -> np.ndarray:
    """Float grid -> RGB heatmap via a matplotlib colormap, upsampled to ``hw``.

    Values are min-max normalized to [0, 1] before mapping. Returns ``(h, w, 3)`` uint8.
    """
    vals = np.asarray(values, dtype=np.float64)
    vmin, vmax = float(np.nanmin(vals)) if vals.size else 0.0, float(np.nanmax(vals)) if vals.size else 1.0
    norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
    rgba = _get_cmap(cmap)(np.nan_to_num(norm, nan=0.0))
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return upsample_grid(rgb, hw)


def feature_pca_rgb(feat_grid) -> np.ndarray:
    """PCA an ``(Hp, Wp, D)`` feature grid (torch or np) to 3 dims -> ``(Hp, Wp, 3)`` uint8.

    Uses numpy SVD (no sklearn). Each of the top-3 principal components is min-maxed
    independently to fill the RGB range. Useful for the raw-vs-debiased feature views.
    """
    arr = feat_grid
    # Accept torch tensors without importing torch at module top.
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float64)
    hp, wp, d = arr.shape
    flat = arr.reshape(hp * wp, d)
    flat = flat - flat.mean(axis=0, keepdims=True)
    # Right singular vectors are the principal directions; project onto top 3.
    k = min(3, d)
    _, _, vh = np.linalg.svd(flat, full_matrices=False)
    proj = flat @ vh[:k].T  # (P, k)
    if k < 3:  # pad to 3 channels if D < 3
        proj = np.pad(proj, ((0, 0), (0, 3 - k)))
    pmin = proj.min(axis=0, keepdims=True)
    pmax = proj.max(axis=0, keepdims=True)
    span = np.where(pmax > pmin, pmax - pmin, 1.0)
    rgb = (proj - pmin) / span
    return (rgb.reshape(hp, wp, 3) * 255.0).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Region tree + instance overlays + IoU (pure numpy; used by the step-by-step app).
# ---------------------------------------------------------------------------
def region_tree(events: list[dict]) -> list[tuple[str, dict]]:
    """Reconstruct the parent->child region tree and return it in DFS pre-order.

    A child region is the event whose ``box`` equals a parent's ``children`` entry
    (exact box-tuple match). Roots are events whose box never appears as any event's
    child. Each returned entry is ``(path_id, event)`` where ``path_id`` is a
    hierarchical dotted string: roots are ``"1"``, ``"2"``, ...; the k-th child (by
    order in the parent's ``children`` list, skipping child boxes that have no event)
    is ``parent_path + "." + str(k + 1)``. Example ids: ``"1"``, ``"1.1"``, ``"1.2"``,
    ``"1.2.1"``. Uses an explicit stack (no recursion).
    """
    by_box: dict[tuple, dict] = {}
    for ev in events:
        by_box.setdefault(tuple(ev["box"]), ev)
    referenced = {tuple(c) for ev in events for c in ev.get("children", [])}
    roots = [ev for ev in events if tuple(ev["box"]) not in referenced]

    out: list[tuple[str, dict]] = []
    # Stack of (path_id, event); push roots in reverse so pre-order is 1, 2, ...
    stack: list[tuple[str, dict]] = [
        (str(i + 1), ev) for i, ev in enumerate(roots)
    ][::-1]
    while stack:
        path_id, ev = stack.pop()
        out.append((path_id, ev))
        # Children with an actual event, numbered by their surviving order.
        kids: list[tuple[str, dict]] = []
        k = 0
        for cb in ev.get("children", []):
            if tuple(cb) == tuple(ev["box"]):
                continue  # degenerate self-referential child (pre-fix cascades) — never follow
            child = by_box.get(tuple(cb))
            if child is None:
                continue  # child enqueued but never processed — skip in numbering
            k += 1
            kids.append((f"{path_id}.{k}", child))
        stack.extend(kids[::-1])  # reverse so DFS visits them in order
    return out


def instances_overlay(crop_rgb: np.ndarray, grids, alpha: float = 0.5) -> np.ndarray:
    """Overlay each patch-grid bool in ``grids`` onto ``crop_rgb`` in a palette colour.

    Each grid is upsampled to the crop's ``(H, W)`` and blended with a distinct
    ``_PALETTE`` colour via :func:`overlay_mask`. Empty list -> the crop as RGB uint8.
    """
    out = _as_rgb(crop_rgb).clip(0, 255).astype(np.uint8)
    grids = grids or []
    for k, g in enumerate(grids):
        out = overlay_mask(out, np.asarray(g, dtype=bool),
                           _PALETTE[k % len(_PALETTE)], alpha)
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two full-res boolean masks (empty union -> 0.0)."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return inter / union


def union_mask(masks, hw: tuple[int, int]) -> np.ndarray:
    """Logical-OR of full-res boolean ``masks`` -> ``(H, W)`` bool (empty -> zeros)."""
    out = np.zeros(hw, dtype=bool)
    for m in masks or []:
        out |= np.asarray(m, dtype=bool)
    return out


def annotation_iou(pred_masks, gt_masks, hw: tuple[int, int]) -> dict:
    """IoU summary of predicted instance masks vs a full ground-truth annotation.

    Returns ``{"union_iou", "mean_best_iou", "n_pred", "n_gt"}`` where ``union_iou``
    is the IoU of the OR of all preds vs the OR of all GT, and ``mean_best_iou`` is
    the mean over GT instances of the best IoU against any pred (0 if no pred). Pure
    numpy, O(n_pred * n_gt).
    """
    pred_masks = list(pred_masks or [])
    gt_masks = list(gt_masks or [])
    union_iou = mask_iou(union_mask(pred_masks, hw), union_mask(gt_masks, hw))
    if gt_masks:
        best = []
        for gt in gt_masks:
            best.append(max((mask_iou(gt, p) for p in pred_masks), default=0.0))
        mean_best_iou = float(np.mean(best))
    else:
        mean_best_iou = 0.0
    return {
        "union_iou": union_iou,
        "mean_best_iou": mean_best_iou,
        "n_pred": len(pred_masks),
        "n_gt": len(gt_masks),
    }


# ---------------------------------------------------------------------------
# Cascade trace: one panel per region the recursive zoom processed.
# ---------------------------------------------------------------------------
def render_trace(
    image: np.ndarray, image_id: str, events: list[dict], *, max_panels: int = 24
):
    """Contact sheet of the recursive zoom -- one panel per processed region.

    Adapted from ``experiments.visualize.render_cascade_trace`` but takes a plain
    ``image`` + ``image_id`` instead of an ``EvalItem`` (the original only used
    ``item.image`` and ``item.image_id``). Reads the dicts emitted by
    ``cascade``'s ``observer`` hook: ``box``, ``fg`` (patch grid),
    ``decision``, ``reid_score``, ``children`` boxes. Returns a matplotlib Figure.
    """
    import matplotlib.patches as mpatches

    image = np.asarray(image)
    events = events[:max_panels]
    n = len(events)
    ncols = min(4, n) or 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.9 * nrows), squeeze=False)
    fig.suptitle(f"{image_id}  cascade trace ({n} regions shown)", fontsize=13)

    for k, ax in enumerate(axes.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        if k >= n:
            ax.axis("off")
            continue
        ev = events[k]
        y0, y1, x0, x1 = ev["box"]
        crop = image[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]

        view = _as_rgb(crop)
        fg = ev.get("fg")
        if fg is not None and np.asarray(fg).size:
            fg_px = upsample_grid(np.asarray(fg, dtype=bool), (ch, cw))
            view[fg_px] = 0.5 * view[fg_px] + 0.5 * np.array([255.0, 0.0, 0.0])
        ax.imshow(view.clip(0, 255).astype(np.uint8))

        for (cy0, cy1, cx0, cx1) in ev.get("children", []):
            ax.add_patch(mpatches.Rectangle(
                (cx0 - x0, cy0 - y0), cx1 - cx0, cy1 - cy0,
                fill=False, edgecolor="yellow", linewidth=1.5))

        decision = ev.get("decision", "?")
        reid_score = ev.get("reid_score")
        cls_txt = f"{reid_score:.2f}" if isinstance(reid_score, (int, float)) else "nan"
        ax.set_title(
            f"L{ev.get('level')} d{ev.get('depth')}  {decision}\n"
            f"n={ev.get('n_components')}  cls={cls_txt}\n"
            f"{_DECISIONS.get(decision, '')}",
            fontsize=8.5)

    fig.tight_layout()
    return fig
