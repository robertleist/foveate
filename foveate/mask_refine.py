"""Snap a patch-grid instance mask to the image's own boundaries (roadmap A1.3 / A4.1).

Every mask the cascade emits is built on the encoder's **patch grid**, so its boundary is a
staircase with a step of ``patch_size`` pixels (16 for DINOv3 ViT-S/16). That puts a hard ceiling on
mask IoU, and the ceiling is far lower than it sounds: an instance 2 patches across cannot be
localized to better than roughly half its own width, so mask AP at the strict IoU thresholds
(0.75-0.95) is capped by the representation, not by the algorithm. On the dense LVIS slice, where
instances are 1-2 patches, this dominates the metric.

Refinement is therefore not cosmetic — without it, mask AP measures patch size more than it measures
the method, and tuning against it optimizes the wrong thing.

Methods (``cfg.mask_refine``):

``none``
    Default. The upsampled patch mask, unchanged.
``grabcut``
    Iterated graph-cut with a Gaussian-mixture colour model (Rother et al., 2004), seeded by the
    patch mask: its eroded interior is *definite* foreground, a dilated ring around it is *probable*
    background, and the band between them is what GrabCut decides. Uses the crop's own pixels, needs
    no training and no extra dependency (OpenCV is already required), and it is the standard way to
    turn a coarse region into a pixel-accurate one.

The refinement never *moves* an instance: it only re-decides pixels near the existing boundary, and
any result that collapses (empty, or a large area change) is rejected in favour of the input mask.
That keeps a refinement failure from turning a correct detection into a miss.
"""

from __future__ import annotations

import numpy as np

_METHODS = ("none", "grabcut")


def refine_mask(mask: np.ndarray, image: np.ndarray, cfg) -> np.ndarray:
    """Refine one instance ``mask`` (``(H, W)`` bool) against ``image`` (``(H, W, 3)`` uint8)."""
    method = str(getattr(cfg, "mask_refine", "none"))
    if method == "none":
        return mask
    if method not in _METHODS:
        raise ValueError(f"Unknown mask_refine {method!r}; expected one of {_METHODS}.")
    return _grabcut(mask, image, cfg)


def _grabcut(mask: np.ndarray, image: np.ndarray, cfg) -> np.ndarray:
    import cv2
    from scipy.ndimage import binary_dilation, binary_erosion

    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or image is None or image.size == 0:
        return mask

    band = max(1, int(getattr(cfg, "mask_refine_band", 2)))
    iters = max(1, int(getattr(cfg, "mask_refine_iters", 3)))
    max_change = float(getattr(cfg, "mask_refine_max_change", 0.5))

    # Work in a padded box around the instance: GrabCut is O(pixels), and the boundary is all we
    # re-decide, so running it on the whole image would be wasted work.
    ys, xs = np.where(mask)
    h, w = mask.shape
    pad = band * 4
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    sub = mask[y0:y1, x0:x1]
    img = np.ascontiguousarray(image[y0:y1, x0:x1, :3])
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[:2] != sub.shape or min(sub.shape) < 3:
        return mask

    inner = binary_erosion(sub, iterations=band, border_value=0)
    outer = binary_dilation(sub, iterations=band, border_value=0)
    gc = np.full(sub.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc[outer] = cv2.GC_PR_FGD
    gc[sub] = cv2.GC_PR_FGD
    gc[inner] = cv2.GC_FGD                      # trusted core: never re-decided
    gc[~outer] = cv2.GC_BGD                     # outside the dilated ring: trusted background
    if not (gc == cv2.GC_FGD).any() or not (gc == cv2.GC_BGD).any():
        return mask                             # degenerate seeding (mask fills or vanishes)

    try:
        cv2.grabCut(img, gc, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                    iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask                             # GrabCut can fail on degenerate colour models

    out = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD))
    if not out.any():
        return mask
    # Reject a refinement that changed the area drastically: that is a colour model that latched
    # onto the background, not a better boundary.
    if abs(out.sum() - sub.sum()) / max(sub.sum(), 1) > max_change:
        return mask

    refined = np.zeros_like(mask)
    refined[y0:y1, x0:x1] = out
    return refined
