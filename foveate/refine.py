"""Iterative foreground-crop re-embedding.

DINOv3 resizes *any* input to ``image_size x image_size`` before patchifying, so the
patch grid is a fixed budget (e.g. 64x64). When the foreground occupies a small part
of the frame, most of that budget is spent on background. **Cropping to the
foreground and re-embedding** hands the whole budget to the instances: each instance
now spans many more patches, sharpening intra-instance detail and -- crucially for
individuation -- the seams *between* touching instances.

This module runs that loop:

    round r:  embed(image_r) -> gate -> foreground_r
              crop image_r to the foreground bbox -> image_{r+1}
              (exemplar masks are sliced with the same box, staying in-frame)

It returns every round's intermediates so a notebook can show how the PCA / feature
structure evolves. The loop converges once the foreground already fills the frame
(the crop box stops shrinking) or the exemplar would be cropped out.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from foveate import features as featlib
from foveate.config import Config
from foveate.foreground import GateResult, build_extractor


@dataclass
class RefineRound:
    index: int
    image: np.ndarray                 # the (cropped) image embedded this round
    features: "object"                # (Hp, Wp, D) tensor
    grid_hw: tuple[int, int]
    gate: GateResult
    exemplar_masks: list[np.ndarray]  # exemplar masks in this round's coordinates
    crop_box: tuple[int, int, int, int] | None  # (y0, y1, x0, x1) producing the NEXT round


def foreground_pixel_bbox(
    foreground_grid: np.ndarray,
    image_hw: tuple[int, int],
    pad_frac: float = 0.06,
) -> tuple[int, int, int, int] | None:
    """Bounding box (in image pixels) of the patch-grid foreground, padded.

    Returns ``(y0, y1, x0, x1)`` or ``None`` if the foreground is empty.
    """
    h, w = image_hw
    fg_full = cv2.resize(foreground_grid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(fg_full > 0)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = int(round((y1 - y0) * pad_frac))
    pad_x = int(round((x1 - x0) * pad_frac))
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    return y0, y1, x0, x1


def _box_is_trivial(box: tuple[int, int, int, int], image_hw: tuple[int, int], tol: float = 0.02) -> bool:
    """True if the crop box already (almost) covers the whole frame -> converged."""
    y0, y1, x0, x1 = box
    h, w = image_hw
    return (y1 - y0) >= (1 - tol) * h and (x1 - x0) >= (1 - tol) * w


def iterative_refine(
    backbone,
    image: np.ndarray,
    exemplar_masks: list[np.ndarray],
    params: Config | dict | None = None,
    rounds: int = 4,
    pad_frac: float = 0.06,
    min_box: int = 24,
) -> list[RefineRound]:
    """Run the crop-and-re-embed loop; return one :class:`RefineRound` per iteration.

    Parameters
    ----------
    rounds:
        Maximum number of embeddings (including round 0 on the full image).
    pad_frac:
        Fractional padding added around the foreground bbox before cropping, so
        instances at the foreground edge keep some context.
    min_box:
        Stop if the next crop would be smaller than this many pixels on a side
        (avoids degenerate tiny crops).
    """
    if not isinstance(params, Config):
        params = Config.from_dict(params)

    cur_image = image
    cur_exemplars = exemplar_masks
    history: list[RefineRound] = []
    extractor = build_extractor(params)

    for r in range(rounds):
        feats = featlib.embed_image(backbone, cur_image, standardize=params.standardize)
        # Intra-image each round: reference is the current crop's exemplars in the current crop.
        extractor.set_reference(backbone, cur_image, cur_exemplars, None, params)
        gate = extractor.predict(feats)

        box = None
        if r < rounds - 1:
            cand = foreground_pixel_bbox(gate.foreground, cur_image.shape[:2], pad_frac)
            if cand is not None:
                y0, y1, x0, x1 = cand
                next_exemplars = [m[y0:y1, x0:x1] for m in cur_exemplars]
                big_enough = (y1 - y0) >= min_box and (x1 - x0) >= min_box
                exemplar_kept = all(m.sum() > 0 for m in next_exemplars)
                converged = _box_is_trivial(cand, cur_image.shape[:2])
                if big_enough and exemplar_kept and not converged:
                    box = cand

        history.append(RefineRound(
            index=r,
            image=cur_image,
            features=feats,
            grid_hw=feats.shape[:2],
            gate=gate,
            exemplar_masks=cur_exemplars,
            crop_box=box,
        ))

        if box is None:
            break
        y0, y1, x0, x1 = box
        cur_image = cur_image[y0:y1, x0:x1]
        cur_exemplars = [m[y0:y1, x0:x1] for m in cur_exemplars]

    return history
