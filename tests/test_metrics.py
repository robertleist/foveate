import numpy as np

from experiments.eval import (
    ImagePrediction,
    box_iou_from_masks,
    evaluate,
    exemplar_recovery_iou,
    iou_matrix,
    masks_to_boxes,
)


def _square(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_iou_matrix_identity():
    a = _square(20, 20, 2, 10, 2, 10)
    iou = iou_matrix(np.stack([a]), np.stack([a]))
    assert iou.shape == (1, 1)
    assert np.isclose(iou[0, 0], 1.0)


def test_iou_disjoint_is_zero():
    a = _square(20, 20, 0, 5, 0, 5)
    b = _square(20, 20, 10, 15, 10, 15)
    assert np.isclose(iou_matrix(np.stack([a]), np.stack([b]))[0, 0], 0.0)


def test_perfect_prediction_scores_one():
    gt = np.stack([_square(40, 40, 2, 12, 2, 12), _square(40, 40, 20, 30, 20, 30)])
    pred = ImagePrediction(masks=gt.copy(), scores=np.array([0.9, 0.8]))
    m = evaluate([pred], [gt])
    assert np.isclose(m.ap50, 1.0)
    assert np.isclose(m.ap, 1.0)
    assert np.isclose(m.box_ap50, 1.0)
    assert np.isclose(m.box_ap, 1.0)
    assert np.isclose(m.mean_iou, 1.0)
    assert np.isclose(m.pq, 1.0)
    assert m.count_error == 0.0
    assert m.n_pred == 2 and m.n_gt == 2


def test_missing_and_extra_predictions():
    gt = np.stack([_square(40, 40, 2, 12, 2, 12), _square(40, 40, 20, 30, 20, 30)])
    # Predict only the first GT, plus a spurious box -> 1 TP, 1 FP, 1 FN.
    pred = ImagePrediction(
        masks=np.stack([gt[0], _square(40, 40, 0, 3, 35, 38)]),
        scores=np.array([0.9, 0.5]),
    )
    m = evaluate([pred], [gt])
    assert m.count_error == 0.0          # 2 predictions vs 2 GT
    assert 0.0 < m.ap50 < 1.0           # one real match, one false positive
    assert m.pq < 1.0


def test_masks_to_boxes():
    m = _square(20, 20, 3, 8, 5, 12)  # rows [3,8), cols [5,12)
    boxes = masks_to_boxes(np.stack([m]))
    assert boxes.shape == (1, 4)
    # [x0, y0, x1, y1] with exclusive x1/y1.
    assert list(boxes[0]) == [5.0, 3.0, 12.0, 8.0]


def test_box_iou_matches_mask_iou_for_rectangles():
    # For axis-aligned rectangular masks, box IoU == mask IoU.
    a = _square(30, 30, 2, 12, 2, 12)
    b = _square(30, 30, 7, 17, 7, 17)
    stack = np.stack([a, b])
    m_iou = iou_matrix(stack, stack)
    b_iou = box_iou_from_masks(stack, stack)
    assert np.allclose(m_iou, b_iou)


def test_box_ap_forgives_ragged_masks():
    # A ragged prediction whose bounding box still matches GT: box AP > mask AP.
    gt_mask = _square(40, 40, 5, 25, 5, 25)
    pred_mask = gt_mask.copy()
    pred_mask[10:20, 10:20] = False  # punch a hole -> lower mask IoU, same box
    pred = ImagePrediction(masks=np.stack([pred_mask]), scores=np.array([0.9]))
    m = evaluate([pred], [np.stack([gt_mask])])
    assert m.box_ap == 1.0            # box is exact -> perfect at every threshold
    assert m.ap < m.box_ap            # hole drops mask IoU at the stricter thresholds


def test_exemplar_recovery():
    ex = _square(30, 30, 5, 15, 5, 15)
    pred = ImagePrediction(masks=np.stack([ex.copy()]), scores=np.array([1.0]))
    assert np.isclose(exemplar_recovery_iou(pred, [ex]), 1.0)
