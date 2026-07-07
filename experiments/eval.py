"""Instance-segmentation metrics for foveate experiments.

Class-agnostic (the exemplar defines a single concept), mask-based:

* **AP / AP50 / AP75** — COCO-style average precision over mask-IoU thresholds, predictions
  ranked by score, greedy IoU matching. ``box_ap*`` are the same, but over bounding-box IoU
  (boxes derived from the masks) — i.e. detection rather than segmentation AP.
* **Panoptic Quality (PQ)** — ``SQ × RQ`` with the IoU>0.5 unique-matching rule.
* **mean IoU** — mean IoU over matched (TP) pairs.
* **count error** — mean ``|n_pred - n_gt|`` and its relative form.
* **exemplar recovery** — does a discovered instance reproduce the prompt (max IoU)?

Masks are ``(N, H, W)`` boolean/uint8 arrays; everything is pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO's 10 IoU thresholds.
COCO_IOU_THRESHOLDS = np.round(np.arange(0.5, 1.0, 0.05), 2)


@dataclass
class ImagePrediction:
    masks: np.ndarray            # (N, H, W) bool
    scores: np.ndarray           # (N,) float


def _as_bool(masks: np.ndarray) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 2:
        arr = arr[None]
    return arr.astype(bool)


def iou_matrix(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N, H, W)`` predictions and ``(M, H, W)`` GT → ``(N, M)``."""
    pred = _as_bool(pred_masks)
    gt = _as_bool(gt_masks)
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return np.zeros((pred.shape[0], gt.shape[0]), dtype=np.float64)
    p = pred.reshape(pred.shape[0], -1).astype(np.float64)
    g = gt.reshape(gt.shape[0], -1).astype(np.float64)
    inter = p @ g.T                                          # (N, M)
    area_p = p.sum(axis=1, keepdims=True)
    area_g = g.sum(axis=1, keepdims=True)
    union = area_p + area_g.T - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """Tight bounding boxes ``[x0, y0, x1, y1]`` (x1/y1 exclusive) for ``(N, H, W)`` masks.

    Empty masks map to a zero-area box, which yields zero IoU against anything.
    """
    m = _as_bool(masks)
    boxes = np.zeros((m.shape[0], 4), dtype=np.float64)
    for i in range(m.shape[0]):
        ys, xs = np.where(m[i])
        if ys.size == 0:
            continue
        boxes[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return boxes


def box_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` boxes ``[x0, y0, x1, y1]`` → ``(N, M)``."""
    pred = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return np.zeros((pred.shape[0], gt.shape[0]), dtype=np.float64)
    x0 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y0 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x1 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y1 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.clip(x1 - x0, 0.0, None) * np.clip(y1 - y0, 0.0, None)
    area_p = ((pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]))[:, None]
    area_g = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))[None, :]
    union = area_p + area_g - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def box_iou_from_masks(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """Box IoU ``(N, M)`` between predictions and GT, boxes derived from their masks."""
    return box_iou_matrix(masks_to_boxes(pred_masks), masks_to_boxes(gt_masks))


def _match(iou: np.ndarray, order: np.ndarray, threshold: float):
    """Greedy match detections (in ``order``) to GT at ``threshold``. Returns (tp, gt_matched)."""
    n_pred, n_gt = iou.shape
    tp = np.zeros(n_pred, dtype=bool)
    gt_taken = np.zeros(n_gt, dtype=bool)
    for det in order:
        best_j, best_iou = -1, threshold
        for j in range(n_gt):
            if gt_taken[j]:
                continue
            if iou[det, j] >= best_iou:
                best_iou = iou[det, j]
                best_j = j
        if best_j >= 0:
            tp[det] = True
            gt_taken[best_j] = True
    return tp, gt_taken


def _ap_from_pr(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """COCO 101-point interpolated AP from per-detection (score, tp) and total GT count."""
    if n_gt == 0:
        return float("nan")
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = tp[order]
    fp = ~tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # Make precision monotonically decreasing from the right.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    rec_levels = np.linspace(0, 1, 101)
    out = np.zeros_like(rec_levels)
    for i, r in enumerate(rec_levels):
        idx = np.searchsorted(recall, r, side="left")
        if idx < precision.size:
            out[i] = precision[idx]
    return float(out.mean())


def _ap_per_threshold(
    predictions: list["ImagePrediction"],
    ious: list[np.ndarray],
    total_gt: int,
    iou_thresholds: np.ndarray,
) -> dict[float, float]:
    """COCO AP at each IoU threshold, pooling all detections globally, ranked by score.

    ``ious[i]`` is the ``(N_i, M_i)`` IoU matrix (mask or box) for image ``i``.
    """
    per_threshold_ap: dict[float, float] = {}
    for t in iou_thresholds:
        all_scores, all_tp = [], []
        for p, iou in zip(predictions, ious):
            if iou.shape[0] == 0:
                continue
            order = np.argsort(-np.asarray(p.scores))
            tp, _ = _match(iou, order, float(t))
            all_scores.append(np.asarray(p.scores, dtype=np.float64))
            all_tp.append(tp)
        scores = np.concatenate(all_scores) if all_scores else np.array([])
        tp = np.concatenate(all_tp) if all_tp else np.array([], dtype=bool)
        per_threshold_ap[float(t)] = _ap_from_pr(scores, tp, total_gt)
    return per_threshold_ap


def _summarize_ap(per_threshold_ap: dict[float, float]) -> tuple[float, float, float]:
    """(mean AP over thresholds, AP50, AP75) from a per-threshold AP dict."""
    aps = np.array([v for v in per_threshold_ap.values() if not np.isnan(v)])
    ap = float(aps.mean()) if aps.size else float("nan")
    return ap, per_threshold_ap.get(0.5, float("nan")), per_threshold_ap.get(0.75, float("nan"))


@dataclass
class Metrics:
    ap: float = float("nan")
    ap50: float = float("nan")
    ap75: float = float("nan")
    box_ap: float = float("nan")
    box_ap50: float = float("nan")
    box_ap75: float = float("nan")
    mean_iou: float = float("nan")
    pq: float = float("nan")
    sq: float = float("nan")
    rq: float = float("nan")
    count_error: float = float("nan")
    count_error_rel: float = float("nan")
    n_pred: int = 0
    n_gt: int = 0
    per_threshold_ap: dict = field(default_factory=dict)
    box_per_threshold_ap: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        skip = {"per_threshold_ap", "box_per_threshold_ap"}
        d = {k: v for k, v in self.__dict__.items() if k not in skip}
        d.update({f"ap_{t:.2f}": v for t, v in self.per_threshold_ap.items()})
        d.update({f"box_ap_{t:.2f}": v for t, v in self.box_per_threshold_ap.items()})
        return d


def evaluate(
    predictions: list[ImagePrediction],
    gts: list[np.ndarray],
    iou_thresholds: np.ndarray = COCO_IOU_THRESHOLDS,
) -> Metrics:
    """Aggregate class-agnostic instance metrics over a dataset.

    ``predictions[i]`` are the discovered masks+scores for image ``i``; ``gts[i]`` is the
    ``(M, H, W)`` GT masks for that image.
    """
    iou_thresholds = np.asarray(iou_thresholds, dtype=np.float64)
    ious = [iou_matrix(p.masks, g) for p, g in zip(predictions, gts)]
    box_ious = [box_iou_from_masks(p.masks, g) for p, g in zip(predictions, gts)]

    total_gt = int(sum(_as_bool(g).shape[0] for g in gts))

    # AP per threshold, over mask IoU (segmentation) and box IoU (detection); detections are
    # pooled globally and ranked by score.
    per_threshold_ap = _ap_per_threshold(predictions, ious, total_gt, iou_thresholds)
    box_per_threshold_ap = _ap_per_threshold(predictions, box_ious, total_gt, iou_thresholds)
    ap, ap50, ap75 = _summarize_ap(per_threshold_ap)
    box_ap, box_ap50, box_ap75 = _summarize_ap(box_per_threshold_ap)

    # PQ + mean IoU at the 0.5 unique-matching rule.
    tp_iou_sum, n_tp, n_fp, n_fn = 0.0, 0, 0, 0
    for p, iou in zip(predictions, ious):
        n_pred, n_gt = iou.shape
        if n_gt == 0:
            n_fp += n_pred
            continue
        if n_pred == 0:
            n_fn += n_gt
            continue
        order = np.argsort(-np.asarray(p.scores))
        tp, gt_taken = _match(iou, order, 0.5)
        for det in np.where(tp)[0]:
            tp_iou_sum += float(iou[det].max())
        n_tp += int(tp.sum())
        n_fp += int((~tp).sum())
        n_fn += int((~gt_taken).sum())
    sq = tp_iou_sum / n_tp if n_tp else float("nan")
    rq = n_tp / (n_tp + 0.5 * n_fp + 0.5 * n_fn) if (n_tp + n_fp + n_fn) else float("nan")
    pq = sq * rq if (n_tp and not np.isnan(sq)) else (0.0 if (n_fp or n_fn) else float("nan"))
    mean_iou = tp_iou_sum / n_tp if n_tp else float("nan")

    # Count error.
    abs_err, rel_err, n_img = 0.0, 0.0, 0
    n_pred_total = 0
    for p, g in zip(predictions, gts):
        n_pred = _as_bool(p.masks).shape[0] if p.masks is not None else 0
        n_gt = _as_bool(g).shape[0]
        n_pred_total += n_pred
        abs_err += abs(n_pred - n_gt)
        rel_err += abs(n_pred - n_gt) / max(n_gt, 1)
        n_img += 1
    count_error = abs_err / n_img if n_img else float("nan")
    count_error_rel = rel_err / n_img if n_img else float("nan")

    return Metrics(
        ap=ap, ap50=ap50, ap75=ap75,
        box_ap=box_ap, box_ap50=box_ap50, box_ap75=box_ap75,
        mean_iou=mean_iou,
        pq=pq, sq=sq, rq=rq,
        count_error=count_error, count_error_rel=count_error_rel,
        n_pred=n_pred_total, n_gt=total_gt,
        per_threshold_ap=per_threshold_ap,
        box_per_threshold_ap=box_per_threshold_ap,
    )


def exemplar_recovery_iou(predictions: ImagePrediction, exemplar_masks: list[np.ndarray]) -> float:
    """Mean over exemplars of the best IoU achieved by any prediction (does a leaf reproduce
    the prompt?)."""
    if not exemplar_masks:
        return float("nan")
    gt = np.stack([_as_bool(m)[0] if np.asarray(m).ndim == 2 else _as_bool(m) for m in exemplar_masks])
    iou = iou_matrix(predictions.masks, gt)
    if iou.shape[0] == 0:
        return 0.0
    return float(iou.max(axis=0).mean())
