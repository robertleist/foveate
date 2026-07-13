import numpy as np
import pytest

from foveate import Config, cascade
from foveate.foreground import build_extractor, normalize_reference


# ---------------------------------------------------------------------------
# normalize_reference
# ---------------------------------------------------------------------------
def _img(color, box):
    img = np.zeros((128, 128, 3), np.uint8)
    (y0, y1, x0, x1) = box
    img[y0:y1, x0:x1] = color
    m = np.zeros((128, 128), np.uint8)
    m[y0:y1, x0:x1] = 1
    return img, m


def test_normalize_reference_single_image_broadcasts():
    img = np.zeros((8, 8, 3), np.uint8)
    m1 = np.zeros((8, 8), np.uint8); m1[0:2, 0:2] = 1
    m2 = np.zeros((8, 8), np.uint8); m2[4:6, 4:6] = 1
    images, masks = normalize_reference(img, [m1, m2])
    assert len(images) == len(masks) == 2
    assert all(im is img for im in images)               # same image broadcast to both masks


def test_normalize_reference_parallel_lists_and_empty_drop():
    a, ma = _img((220, 0, 0), (0, 20, 0, 20))
    b, mb = _img((0, 0, 220), (40, 60, 40, 60))
    empty = np.zeros((128, 128), np.uint8)
    images, masks = normalize_reference([a, b, a], [ma, empty, mb])   # middle mask empty → dropped
    assert len(images) == len(masks) == 2
    assert images[0] is a and images[1] is a
    assert masks[0].sum() == ma.sum() and masks[1].sum() == mb.sum()


def test_normalize_reference_length_mismatch_raises():
    a = np.zeros((8, 8, 3), np.uint8)
    m = np.ones((8, 8), np.uint8)
    with pytest.raises(ValueError):
        normalize_reference([a, a], [m])                 # 2 images, 1 mask, and not length-1


def test_normalize_reference_all_empty_raises():
    with pytest.raises(ValueError):
        normalize_reference(np.zeros((8, 8, 3), np.uint8), [np.zeros((8, 8), np.uint8)])


# ---------------------------------------------------------------------------
# cascade with exemplars pooled across images
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("extractor", ["insid3", "bank"])
def test_multi_image_exemplars_discover_targets(backbone, two_squares, extractor):
    """Exemplars pooled from two separate images find the red squares in a third target."""
    target, _ = two_squares
    ex_a, mask_a = _img((220, 30, 30), (20, 40, 20, 40))     # red square, image A
    ex_b, mask_b = _img((220, 30, 30), (80, 100, 80, 100))   # red square, image B

    cfg = Config(foreground_extractor=extractor, standardize=False, gate_threshold=0.8,
                 min_crop=24, cascade_min_instance_area=4, crop_sim_floor=0.0)
    instances, stats = cascade(
        backbone, target, [mask_a, mask_b], config=cfg,
        exemplar_images=[ex_a, ex_b],
    )
    assert instances, "no instances discovered from multi-image exemplars"
    for inst in instances:                                    # none should be the blue distractor
        ys, xs = np.where(inst.mask)
        cy, cx = ys.mean(), xs.mean()
        assert not (cy < 60 and cx > 60), "discovered the blue distractor"


def test_exemplar_image_and_images_are_mutually_exclusive(backbone, two_squares):
    target, ex = two_squares
    with pytest.raises(ValueError):
        cascade(backbone, target, ex, config=Config(),
                exemplar_image=target, exemplar_images=[target])


# ---------------------------------------------------------------------------
# Per-crop exemplar selection + dynamic params (INSID3)
# ---------------------------------------------------------------------------
def _insid3_reference(backbone, cfg, images, masks):
    ext = build_extractor(cfg)
    ext.set_reference(backbone, images, masks, None, cfg)
    return ext


def test_selection_picks_the_most_similar_exemplar(backbone):
    """Top-1 selection uses the exemplar whose CLS best matches the crop, not the pooled set."""
    red, red_m = _img((220, 30, 30), (20, 60, 20, 60))
    blue, blue_m = _img((30, 30, 220), (20, 60, 20, 60))
    cfg = Config(standardize=False, insid3_top_k_exemplars=1)
    ext = _insid3_reference(backbone, cfg, [red, blue], [red_m, blue_m])

    import foveate.features as featlib
    (feat, cls) = featlib.embed_batch(backbone, [red], standardize=False)[0]
    _, _, _, _, _, sel, _ = ext._select_reference(cls, feat.device)
    # The reference embeds a red and a blue exemplar crop; a red crop must select the red one.
    assert len(sel) == 1
    red_idx = int(np.argmax((ext.exemplar_cls @ cls).detach().cpu().numpy()))
    assert sel == [red_idx]


def test_top_k_and_pooled_fallback(backbone):
    a, ma = _img((220, 30, 30), (20, 60, 20, 60))
    b, mb = _img((30, 220, 30), (20, 60, 20, 60))
    c, mc = _img((30, 30, 220), (20, 60, 20, 60))
    cfg = Config(standardize=False, insid3_top_k_exemplars=2)
    ext = _insid3_reference(backbone, cfg, [a, b, c], [ma, mb, mc])

    import foveate.features as featlib
    (feat, cls) = featlib.embed_batch(backbone, [a], standardize=False)[0]
    _, _, _, _, _, sel, sim = ext._select_reference(cls, feat.device)
    assert len(sel) == 2 and sim is not None
    # cls=None → pooled fallback over all three exemplars, no similarity signal.
    _, _, _, _, _, sel_all, sim_none = ext._select_reference(None, feat.device)
    assert sel_all == [0, 1, 2] and sim_none is None


def test_dynamic_params_set_tau_and_aggregate_from_similarity(backbone):
    a, ma = _img((220, 30, 30), (20, 60, 20, 60))
    b, mb = _img((30, 30, 220), (20, 60, 20, 60))
    cfg = Config(standardize=False, insid3_top_k_exemplars=1,
                 insid3_dynamic_tau_fg=True, insid3_dynamic_aggt=True)
    ext = _insid3_reference(backbone, cfg, [a, b], [ma, mb])

    import foveate.features as featlib
    (feat, cls) = featlib.embed_batch(backbone, [a], standardize=False)[0]
    _, _, _, tau, aggregate, _, sim = ext._select_reference(cls, feat.device)
    # Inverse coupling: a dissimilar crop splits finely (high tau) and re-merges freely
    # (low aggregate); a frame-filling match splits coarsely.
    assert abs(tau - float(np.clip(cfg.insid3_tau_fg_scale * (1.0 - sim), 0.05, 0.95))) < 1e-6
    assert abs(aggregate - float(np.clip(cfg.insid3_aggt_scale * sim, 0.0, 0.95))) < 1e-6


def test_dynamic_tau_and_aggregate_toggle_independently(backbone):
    a, ma = _img((220, 30, 30), (20, 60, 20, 60))
    cfg = Config(standardize=False, insid3_dynamic_tau_fg=True)  # aggregate stays static
    ext = _insid3_reference(backbone, cfg, [a], [ma])

    import foveate.features as featlib
    (feat, cls) = featlib.embed_batch(backbone, [a], standardize=False)[0]
    _, _, _, tau, aggregate, _, sim = ext._select_reference(cls, feat.device)
    assert abs(tau - float(np.clip(cfg.insid3_tau_fg_scale * (1.0 - sim), 0.05, 0.95))) < 1e-6
    assert aggregate == cfg.insid3_aggt

    ext.cfg.insid3_dynamic_tau_fg = False
    ext.cfg.insid3_dynamic_aggt = True     # now only aggregate is dynamic
    _, _, _, tau, aggregate, _, sim = ext._select_reference(cls, feat.device)
    assert tau == ext.cfg.insid3_tau_fg
    assert abs(aggregate - float(np.clip(ext.cfg.insid3_aggt_scale * sim, 0.0, 0.95))) < 1e-6


def test_single_exemplar_still_uses_dynamic_params(backbone):
    """Regression: with ONE exemplar, dynamic tau/aggregate must still derive from the CLS sim
    (the similarity is computed even though top-k selection is trivial)."""
    a, ma = _img((220, 30, 30), (20, 60, 20, 60))
    cfg = Config(standardize=False,
                 insid3_dynamic_tau_fg=True, insid3_dynamic_aggt=True)
    ext = _insid3_reference(backbone, cfg, [a], [ma])         # single exemplar

    import foveate.features as featlib
    (feat, cls) = featlib.embed_batch(backbone, [a], standardize=False)[0]
    _, _, _, tau, aggregate, sel, sim = ext._select_reference(cls, feat.device)
    assert sel == [0] and sim is not None                    # sim measured for the lone exemplar
    assert abs(tau - float(np.clip(cfg.insid3_tau_fg_scale * (1.0 - sim), 0.05, 0.95))) < 1e-6
    assert abs(aggregate - float(np.clip(cfg.insid3_aggt_scale * sim, 0.0, 0.95))) < 1e-6
    # and it is NOT the static default
    assert tau != cfg.insid3_tau_fg or aggregate != cfg.insid3_aggt


def test_default_k1_single_exemplar_matches_pooled(backbone, two_squares):
    """With one exemplar, selection is a no-op — same reference as the pooled path."""
    target, ex = two_squares
    cfg = Config(standardize=False, min_crop=24, cascade_min_instance_area=4, crop_sim_floor=0.0)
    instances, _ = cascade(backbone, target, ex, config=cfg)
    assert instances
