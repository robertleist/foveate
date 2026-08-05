"""MASK — turn an instance's patch grid into pixels (the fourth slot).

The cascade decides *which* regions are instances; this slot decides what an instance's mask
actually looks like. They are separate failure modes and they need separate ceilings: with Extract,
Stop and Merge oracular on the general slice, recall is 100 % and precision is 100 %, and the whole
remaining AP deficit is the patch staircase — AP50 1.000 against AP 0.827, i.e. everything lost sits
at the strict IoU thresholds where boundary quality is the only thing being measured.

``nearest`` (default)
    The exact patch-grid staircase. Faithful to what the extractor said and nothing more.
``bilinear``
    Resize the {0,1} grid as float and re-binarise at 0.5 — the level set halfway between inside and
    outside. Cheap and a large win, but note what it is *not*: it uses no image evidence whatsoever,
    so it cannot find a boundary the grid did not already imply. It is a smoothing prior, not a
    segmentation step, and it is the floor for this slot rather than the answer.
``oracle``
    The slot's upper bound: replace the **shape** of a detection that already matches a ground-truth
    instance with that instance's mask. Deliberately **shape-only** — it may not rescue a detection
    that does not already match at ``oracle_upsample_iou``, because an oracle allowed to snap any
    mask to its nearest object would silently do the Merge slot's work (on the dense slice ~44 % of
    detections are clipped fragments; snapping each to its object would drive recall to 100 % and
    attribute a *search* gain to boundary quality). What it isolates is exactly the boundary error
    and nothing else.

Not implemented here, and the reason this is a slot rather than a boolean: guided/joint-bilateral
filtering against the crop's own pixels, and feature upsampling (roadmap §A1.3), which makes the
*grid* finer instead of interpolating a coarse one. Both are registry entries away.

:mod:`foveate.mask_refine` (``mask_refine: grabcut``) is a *separate* post-hoc step that re-decides a
band around the boundary from colour. It overlaps with this slot conceptually and should probably be
folded into it once there is a second real arm here; measured on the oracle runs it currently hurts
(general AP 0.648 → 0.593), so there is no rush.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from foveate import features as featlib

_Box = tuple[int, int, int, int]


@runtime_checkable
class MaskUpsampler(Protocol):
    """Strategy interface for the Mask stage."""

    def upsample(self, grid: np.ndarray, box: _Box, mask=None) -> np.ndarray:
        """A ``(Hp, Wp)`` bool instance grid over crop ``box`` → a ``uint8`` mask of the crop's size.

        ``box`` is ``(y0, y1, x0, x1)`` in original-image coordinates; the returned mask is
        crop-local, which is what the cascade pastes into the full-image mask.

        ``mask`` is the extractor's own crop-local **pixel** mask when it produced one (a segmenter
        arm). The real rules hand it straight back — there is nothing a grid upsample can add to a
        mask that was already made at pixel resolution — but the slot still sees it, so an oracle can
        bound this stage even for an extractor that bypasses the interpolation.
        """
        ...


class NearestUpsampler:
    """``nearest`` (default) — the exact patch-grid staircase."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def upsample(self, grid: np.ndarray, box: _Box, mask=None) -> np.ndarray:
        if mask is not None:
            return np.asarray(mask, dtype=np.uint8)
        y0, y1, x0, x1 = box
        return featlib.upsample_mask(grid, (x1 - x0, y1 - y0), bilinear=False)


class BilinearUpsampler(NearestUpsampler):
    """``bilinear`` — resize as float, re-binarise at 0.5. A smoothing prior, not image evidence."""

    def upsample(self, grid: np.ndarray, box: _Box, mask=None) -> np.ndarray:
        if mask is not None:
            return np.asarray(mask, dtype=np.uint8)
        y0, y1, x0, x1 = box
        return featlib.upsample_mask(grid, (x1 - x0, y1 - y0), bilinear=True)


class OracleUpsampler(NearestUpsampler):
    """``oracle`` — the shape of the GT instance a detection *already* matches.

    Starts from the configured base upsample (``oracle_upsample_base``), finds the GT instance that
    mask overlaps most, and returns that instance's mask restricted to the crop — but only when the
    base mask already clears ``oracle_upsample_iou``. Below it the base mask is returned untouched,
    so a fragment stays a fragment and a false positive stays a false positive.

    The GT is injected per image by the cascade (``cascade(..., gt_foreground=...)``, the same
    payload the other oracle slots consume) via :meth:`set_target_instances`.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._labels: np.ndarray | None = None
        self._base = (BilinearUpsampler(cfg)
                      if str(getattr(cfg, "oracle_upsample_base", "bilinear")) == "bilinear"
                      else NearestUpsampler(cfg))

    def set_target_instances(self, gt: np.ndarray) -> None:
        self._labels = np.asarray(gt).astype(np.int32)

    def upsample(self, grid: np.ndarray, box: _Box, mask=None) -> np.ndarray:
        base = self._base.upsample(grid, box, mask)
        if self._labels is None:
            raise RuntimeError(
                "OracleUpsampler used before set_target_instances — the cascade must receive "
                "gt_foreground for the oracle Mask slot."
            )
        y0, y1, x0, x1 = box
        sub = self._labels[y0:y1, x0:x1]
        m = base.astype(bool)
        if not m.any() or sub.size == 0:
            return base

        # Best-overlapping instance, by IoU against the mask as it would have been emitted.
        best_lab, best_iou = 0, 0.0
        for lab in np.unique(sub[m]):
            if lab == 0:
                continue
            g = sub == lab
            inter = int(np.logical_and(m, g).sum())
            iou = inter / max(int(np.logical_or(m, g).sum()), 1)
            if iou > best_iou:
                best_lab, best_iou = int(lab), iou
        if best_lab == 0 or best_iou < float(getattr(self.cfg, "oracle_upsample_iou", 0.5)):
            return base                                  # not already a match → do not rescue it
        return (sub == best_lab).astype(np.uint8)


#: ``cfg.mask_upsample`` → implementation.
_UPSAMPLERS = {
    "nearest": NearestUpsampler,
    "bilinear": BilinearUpsampler,
    "oracle": OracleUpsampler,
}


def build_mask_upsampler(cfg) -> MaskUpsampler:
    """Build the Mask strategy named by ``cfg.mask_upsample``."""
    name = str(cfg.mask_upsample)
    try:
        return _UPSAMPLERS[name](cfg)
    except KeyError:
        raise ValueError(
            f"Unknown mask_upsample {name!r}; expected one of {sorted(_UPSAMPLERS)}."
        ) from None
