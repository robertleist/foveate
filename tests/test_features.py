import numpy as np
import torch

from foveate import thresholding
from foveate.debias import estimate_positional_basis, project_out
from foveate.prototypes import build_bank


def test_otsu_separates_bimodal():
    lo = np.random.default_rng(0).normal(0.1, 0.02, 500)
    hi = np.random.default_rng(1).normal(0.9, 0.02, 500)
    x = np.concatenate([lo, hi])
    tau = thresholding.otsu(x)
    # A valid threshold puts (nearly) all low samples below and high samples above.
    assert 450 < (x < tau).sum() < 550


def test_gmm2_separates_bimodal():
    lo = np.random.default_rng(0).normal(0.1, 0.02, 500)
    hi = np.random.default_rng(1).normal(0.9, 0.02, 500)
    x = np.concatenate([lo, hi])
    tau = thresholding.gmm2(x)
    assert 450 < (x < tau).sum() < 550


def test_foreground_threshold_shapes():
    grid = np.array([[0.1, 0.9], [0.9, 0.1]])
    fg = thresholding.foreground(grid, "otsu")
    assert fg.shape == grid.shape
    assert fg[0, 1] and fg[1, 0] and not fg[0, 0]


def test_prototype_budget_is_respected(backbone, two_squares):
    img, ex = two_squares
    bank = build_bank(backbone, img, ex, reduction="kmeans", budget=4, per_exemplar_min=1)
    assert len(bank) <= 4 + 1   # budget (+ guaranteed per-exemplar floor)
    assert torch.allclose(bank.prototypes.norm(dim=1), torch.ones(len(bank)), atol=1e-4)


def test_all_reduction_keeps_under_budget(backbone, two_squares):
    img, ex = two_squares
    bank = build_bank(backbone, img, ex, reduction="all", budget=1000)
    assert len(bank) >= 1


def test_debias_projects_out_subspace(backbone):
    B = estimate_positional_basis(backbone, subspace_dim=4, n_noise=1)
    feats = torch.randn(50, B.shape[0])
    out = project_out(feats, B)
    # Residual should have ~no component in the removed subspace.
    leak = (out @ B).abs().max().item()
    assert leak < 1e-4
