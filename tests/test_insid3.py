import numpy as np
import torch

from foveate import Config
from foveate.features import embed_image
from foveate.foreground import GateResult, build_extractor


def _grid_box(r0, r1, c0, c1, hp, wp, img_h=128, img_w=128):
    """Map a full-res pixel box to inclusive-exclusive patch-grid coords."""
    return (int(r0 * hp / img_h), int(np.ceil(r1 * hp / img_h)),
            int(c0 * wp / img_w), int(np.ceil(c1 * wp / img_w)))


def _reference(backbone, two_squares, cfg):
    img, ex = two_squares
    extr = build_extractor(cfg)
    extr.set_reference(backbone, img, ex, None, cfg)
    return extr, img


def test_build_extractor_and_exemplar_cls(backbone, two_squares):
    cfg = Config(foreground_extractor="insid3", standardize=True)
    extr, _ = _reference(backbone, two_squares, cfg)

    assert extr.exemplar_cls is not None
    assert extr.exemplar_cls.ndim == 2                       # (S, D)
    assert extr.exemplar_cls.shape[0] == 1                   # one exemplar mask
    norms = extr.exemplar_cls.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_predict_covers_targets_excludes_distractor(backbone, two_squares):
    cfg = Config(foreground_extractor="insid3", standardize=True)
    extr, img = _reference(backbone, two_squares, cfg)

    res = extr.predict(embed_image(backbone, img, standardize=True))
    assert isinstance(res, GateResult)
    fg = res.foreground
    hp, wp = fg.shape

    red1 = _grid_box(20, 40, 20, 40, hp, wp)
    red2 = _grid_box(80, 100, 80, 100, hp, wp)
    blue = _grid_box(20, 40, 80, 100, hp, wp)

    def covered(b):
        r0, r1, c0, c1 = b
        return int(fg[r0:r1, c0:c1].sum())

    # Qualitative property: foreground lands on both exemplar-colour squares ...
    assert covered(red1) > 0
    assert covered(red2) > 0
    # ... and excludes the blue distractor's patch locations entirely.
    assert covered(blue) == 0


def test_predict_internals_shapes(backbone, two_squares):
    cfg = Config(foreground_extractor="insid3", standardize=True)
    extr, img = _reference(backbone, two_squares, cfg)

    res = extr.predict(embed_image(backbone, img, standardize=True), return_internals=True)
    internals = res.internals
    for key in ("clusters", "forward_sim", "backward_candidates", "seed", "foreground"):
        assert key in internals, key
    hp, wp = res.foreground.shape
    for key in ("clusters", "forward_sim", "backward_candidates", "seed", "foreground"):
        assert internals[key].shape == (hp, wp), key


def test_runs_with_and_without_debias(backbone, two_squares):
    img, _ = two_squares
    for debias in (True, False):
        cfg = Config(foreground_extractor="insid3", standardize=True, debias=debias)
        extr, _ = _reference(backbone, two_squares, cfg)
        res = extr.predict(embed_image(backbone, img, standardize=True))
        assert isinstance(res, GateResult)
        assert res.foreground.dtype == np.bool_
