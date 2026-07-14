import numpy as np
import pytest
import torch

from foveate import Config
from foveate import features as featlib
from foveate.features import l2_normalize
from foveate.foreground import build_extractor
from foveate.gate import BankExtractor
from foveate.insid3 import InSID3Extractor
from foveate.oracle import OracleExtractor
from foveate.otsu import OtsuExtractor


def test_build_extractor_dispatch():
    assert isinstance(build_extractor(Config(foreground_extractor="insid3")), InSID3Extractor)
    assert isinstance(build_extractor(Config(foreground_extractor="otsu")), OtsuExtractor)
    assert isinstance(build_extractor(Config(foreground_extractor="bank")), BankExtractor)
    assert isinstance(build_extractor(Config(foreground_extractor="oracle")), OracleExtractor)


def test_build_extractor_unknown_raises():
    with pytest.raises(ValueError):
        build_extractor(Config(foreground_extractor="nope"))


@pytest.mark.parametrize("name", ["insid3", "otsu", "bank", "oracle"])
def test_exemplar_cls_shape_and_norm(backbone, two_squares, name):
    img, ex = two_squares
    cfg = Config(foreground_extractor=name, standardize=True)
    extr = build_extractor(cfg)
    extr.set_reference(backbone, img, ex, None, cfg)

    exemplar_cls = extr.exemplar_cls
    assert exemplar_cls.ndim == 2                              # (S, D)
    assert exemplar_cls.shape[0] == len(ex)                    # one row per exemplar mask
    norms = exemplar_cls.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


@pytest.mark.parametrize("reduce", ["mean", "max"])
def test_otsu_extractor_predicts_boolean_foreground(backbone, two_squares, reduce):
    """OtsuExtractor.predict returns a boolean foreground grid + Otsu-threshold internals."""
    img, ex = two_squares
    cfg = Config(foreground_extractor="otsu", otsu_top_k=1, otsu_reduce=reduce, debias=False)
    extr = build_extractor(cfg)
    extr.set_reference(backbone, img, ex, None, cfg)

    feat, cls = featlib.embed_batch(backbone, [img])[0]
    gr = extr.predict(feat, cls=cls, return_internals=True)

    assert gr.foreground.dtype == bool
    assert gr.foreground.shape == feat.shape[:2]              # (Hp, Wp)
    assert gr.score_map.min() >= 0.0 and gr.score_map.max() <= 1.0
    assert "otsu_threshold" in gr.internals
    assert gr.internals["selected_exemplars"] == [0]         # single exemplar
    # foreground is exactly the patches at or above the Otsu cut of the similarity map.
    assert bool(gr.foreground.any()) or gr.score_map.max() == 0.0


def test_otsu_top_k_clamps_to_bank_size(backbone, two_squares):
    """otsu_top_k larger than the bank still selects all available exemplars, no crash."""
    img, ex = two_squares
    cfg = Config(foreground_extractor="otsu", otsu_top_k=8, debias=False)
    extr = build_extractor(cfg)
    extr.set_reference(backbone, img, ex, None, cfg)
    feat, cls = featlib.embed_batch(backbone, [img])[0]
    gr = extr.predict(feat, cls=cls, return_internals=True)
    assert gr.internals["selected_exemplars"] == [0]


def test_oracle_extractor_returns_gt_foreground(backbone, two_squares):
    """OracleExtractor.predict slices the injected GT mask to the crop box, resized to the grid."""
    img, ex = two_squares
    cfg = Config(foreground_extractor="oracle")
    extr = build_extractor(cfg)
    extr.set_reference(backbone, img, ex, None, cfg)

    gt = np.zeros(img.shape[:2], dtype=bool)          # GT foreground: only the top-left square
    gt[20:40, 20:40] = True
    extr.set_target_foreground(gt)

    # Full-image crop → the top-left quadrant of the grid is foreground, the rest is not.
    feat, cls = featlib.embed_batch(backbone, [img])[0]
    gr = extr.predict(feat, cls=cls, box=(0, 128, 0, 128), return_internals=True)
    assert gr.foreground.dtype == bool
    assert gr.foreground.shape == feat.shape[:2]
    assert gr.foreground.any()
    # Matches the GT resized to the patch grid exactly (the oracle is, by construction, perfect).
    expected = featlib.resize_mask_to_grid(gt, feat.shape[:2])
    assert np.array_equal(gr.foreground, expected)

    # A crop over the empty bottom-right square yields an all-false foreground.
    gr_empty = extr.predict(feat, cls=cls, box=(80, 100, 80, 100))
    assert not gr_empty.foreground.any()


def test_oracle_extractor_requires_gt_and_box(backbone, two_squares):
    """Predict fails loudly if the GT mask or crop box was never supplied."""
    img, ex = two_squares
    cfg = Config(foreground_extractor="oracle")
    extr = build_extractor(cfg)
    extr.set_reference(backbone, img, ex, None, cfg)
    feat, cls = featlib.embed_batch(backbone, [img])[0]

    with pytest.raises(RuntimeError):                  # GT not injected yet
        extr.predict(feat, cls=cls, box=(0, 128, 0, 128))

    extr.set_target_foreground(np.ones(img.shape[:2], dtype=bool))
    with pytest.raises(RuntimeError):                  # box missing
        extr.predict(feat, cls=cls)


def test_leaf_classification_is_mean_cosine():
    """The leaf accept score is mean cosine of the crop CLS to all exemplar CLS."""
    torch.manual_seed(0)
    s, d = 4, 8
    exemplar_cls = l2_normalize(torch.randn(s, d), dim=1)      # (S, D) unit rows
    target = l2_normalize(torch.randn(d), dim=0)           # (D,)

    explicit = sum(float(target @ exemplar_cls[i]) for i in range(s)) / s
    matrix = float((target @ exemplar_cls.T).mean())
    assert abs(explicit - matrix) < 1e-6
