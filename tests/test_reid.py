import numpy as np
import pytest
import torch

from foveate import Config
from foveate.features import embed_image
from foveate.reid import ReidScorer, build_reid_scorer


def test_cls_scorer_reuses_exemplar_cls_without_embedding(backbone, two_squares):
    """In cls mode a caller's exemplar CLS stack is reused directly — no exemplar embedding."""
    _, ex = two_squares
    cfg = Config(reid_mode="cls")
    stack = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)   # a stand-in (S, D) CLS bank
    scorer = build_reid_scorer(cfg, backbone, np.zeros((4, 4, 3), np.uint8), ex, exemplar_cls=stack)
    assert scorer.mode == "cls" and not scorer.needs_foreground
    assert len(scorer.exemplars) == 3                               # one singleton set per exemplar


def test_cls_scorer_is_topk_mean_of_exemplar_cosines():
    """g in cls mode = top-k mean of the target CLS's cosines to the exemplar CLS stack."""
    exemplars = [e.view(1, -1) for e in torch.eye(3)]              # 3 orthonormal exemplars
    cls = torch.tensor([0.8, 0.6, 0.0])                            # cosines: 0.8, 0.6, 0.0
    assert ReidScorer("cls", 0, exemplars).score(None, cls) == pytest.approx((0.8 + 0.6) / 3)
    assert ReidScorer("cls", 1, exemplars).score(None, cls) == pytest.approx(0.8)
    assert ReidScorer("cls", 2, exemplars).score(None, cls) == pytest.approx(0.7)


def test_masked_modes_need_foreground_cls_does_not():
    assert ReidScorer("cls", 0).needs_foreground is False
    for mode in ("mean", "kmeans", "full"):
        assert ReidScorer(mode, 0).needs_foreground is True


def test_mean_mode_scores_foreground_patches_against_exemplar():
    """mean mode: g = cosine of the mean of the target's FOREGROUND patches to the exemplar mean.
    Only masked (foreground=True) patches count — background patches must not move the score."""
    proto = torch.nn.functional.normalize(torch.randn(4), dim=0)
    scorer = ReidScorer("mean", 0, exemplars=[proto.view(1, 4)], B=None)

    feat = torch.zeros(2, 2, 4)
    feat[0, 0] = proto                                             # one on-concept foreground patch
    feat[1, 1] = -proto                                            # a background patch (excluded)
    fg = np.array([[True, False], [False, False]])
    assert scorer.score(feat, None, fg) == pytest.approx(1.0, abs=1e-5)   # only the fg patch scores


def test_full_mode_is_best_match_per_target_patch():
    """full mode: each target foreground patch takes its best cosine to any exemplar patch, averaged.
    Two exemplar patches (e0, e1); a target of [e0, e1] scores 1.0 (each finds its exact match)."""
    e0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    e1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    scorer = ReidScorer("full", 0, exemplars=[torch.stack([e0, e1])], B=None)
    feat = torch.stack([e0, e1]).view(1, 2, 4)                     # 1x2 grid, both foreground
    fg = np.array([[True, True]])
    assert scorer.score(feat, None, fg) == pytest.approx(1.0, abs=1e-5)


def test_kmeans_k1_equals_mean(backbone, two_squares):
    """reid_mode='kmeans' with k=1 builds the same exemplar representation as reid_mode='mean'."""
    img, ex = two_squares
    mean = build_reid_scorer(Config(reid_mode="mean", debias=False), backbone, img, ex)
    km1 = build_reid_scorer(Config(reid_mode="kmeans", reid_kmeans_k=1, debias=False),
                            backbone, img, ex)
    assert torch.allclose(mean.exemplars[0], km1.exemplars[0], atol=1e-6)


def test_build_masked_scorer_is_extractor_free(backbone, two_squares):
    """build_reid_scorer builds the masked bank straight from the exemplar crop+mask — no Where
    extractor — and scores a target grid+foreground to a finite scalar."""
    img, ex = two_squares
    scorer = build_reid_scorer(Config(reid_mode="full", debias=False), backbone, img, ex)
    assert scorer.mode == "full" and len(scorer.exemplars) == 1
    feat = embed_image(backbone, img[20:40, 20:40], standardize=True)
    fg = np.ones(feat.shape[:2], dtype=bool)
    g = scorer.score(feat, None, fg)
    assert isinstance(g, float) and np.isfinite(g)


def test_build_reid_scorer_rejects_unknown_mode(backbone, two_squares):
    _, ex = two_squares
    cfg = Config.from_dict({"reid_mode": "bogus"})
    with pytest.raises(ValueError, match="reid_mode"):
        build_reid_scorer(cfg, backbone, np.zeros((4, 4, 3), np.uint8), ex)


def test_mode_override_builds_a_masked_scorer_over_a_cls_config(backbone, two_squares):
    """The ``mode`` override lets a cls-config caller build a SECOND masked scorer for the leaf
    confidence — independent of cfg.reid_mode (which stays cls for the recursion signal)."""
    img, ex = two_squares
    cfg = Config(reid_mode="cls", debias=False)
    conf = build_reid_scorer(cfg, backbone, img, ex, mode="full")
    assert conf.mode == "full" and conf.needs_foreground
    # cfg.reid_mode is untouched: the recursion scorer built from the same cfg is still cls.
    recursion = build_reid_scorer(cfg, backbone, img, ex, exemplar_cls=torch.eye(len(ex)))
    assert recursion.mode == "cls"
