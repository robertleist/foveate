import numpy as np
import pytest
import torch

from foveate import Config
from foveate.features import l2_normalize
from foveate.foreground import build_extractor
from foveate.gate import BankExtractor
from foveate.insid3 import InSID3Extractor


def test_build_extractor_dispatch():
    assert isinstance(build_extractor(Config(foreground_extractor="insid3")), InSID3Extractor)
    assert isinstance(build_extractor(Config(foreground_extractor="bank")), BankExtractor)


def test_build_extractor_unknown_raises():
    with pytest.raises(ValueError):
        build_extractor(Config(foreground_extractor="nope"))


@pytest.mark.parametrize("name", ["insid3", "bank"])
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


def test_leaf_classification_is_mean_cosine():
    """The leaf accept score is mean cosine of the crop CLS to all exemplar CLS."""
    torch.manual_seed(0)
    s, d = 4, 8
    exemplar_cls = l2_normalize(torch.randn(s, d), dim=1)      # (S, D) unit rows
    target = l2_normalize(torch.randn(d), dim=0)           # (D,)

    explicit = sum(float(target @ exemplar_cls[i]) for i in range(s)) / s
    matrix = float((target @ exemplar_cls.T).mean())
    assert abs(explicit - matrix) < 1e-6
