import numpy as np

from experiments.eval import (
    ImagePrediction,
    evaluate,
    exemplar_recovery_iou,
    iou_matrix,
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


def test_exemplar_recovery():
    ex = _square(30, 30, 5, 15, 5, 15)
    pred = ImagePrediction(masks=np.stack([ex.copy()]), scores=np.array([1.0]))
    assert np.isclose(exemplar_recovery_iou(pred, [ex]), 1.0)
