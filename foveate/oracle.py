"""An oracle **Where** extractor — the upper-bound ablation on foreground quality.

Every learned extractor (INSID3, Otsu, the bank gate) answers *"which patches of this crop are the
concept?"* imperfectly, and that error propagates into the Extract / Split stages downstream. This
extractor removes the error entirely: it returns the **ground-truth** class foreground of whatever
crop the cascade hands it, by slicing the target image's GT mask to the crop box and resizing it to
the patch grid. Comparing a run of this against the real extractors isolates how much of the final
metric gap is the *Where* stage's fault versus the rest of the cascade (re-identification, splitting,
NMS) — the standard "perfect-foreground" oracle ablation.

The GT foreground is the union of the target image's GT instance masks of the class, injected once
per target image via :meth:`set_target_foreground` (the cascade forwards it from the eval item). The
exemplar bank (:attr:`exemplar_cls`) is still built from the exemplar crops exactly like the other
extractors, because the re-identification score ``g`` (:mod:`foveate.reid`) is independent of the
*Where* stage and must be computed identically across extractors — only the foreground is oracled,
not ``g``.
"""

from __future__ import annotations

import numpy as np
import torch

from foveate import features as featlib
from foveate.foreground import GateResult, normalize_reference


def _mask_bbox(mask: np.ndarray, pad_frac: float) -> tuple[int, int, int, int]:
    """Padded bbox of a binary mask (patch-scale framing shared with the cascade's crops)."""
    ys, xs = np.where(mask)
    h, w = mask.shape
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    py, px = int((y1 - y0) * pad_frac), int((x1 - x0) * pad_frac)
    return max(0, y0 - py), min(h, y1 + py), max(0, x0 - px), min(w, x1 + px)


class OracleExtractor:
    """Ground-truth foreground extractor behind the :class:`ForegroundExtractor` interface.

    ``set_reference`` builds the exemplar CLS bank (for the re-id score ``g``) exactly like the other
    extractors; ``set_target_foreground`` injects the target image's GT class mask; ``predict``
    returns that GT sliced to the queried crop box — so the foreground is always perfect.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.exemplar_cls: torch.Tensor | None = None      # (S, D) L2-normalized per-exemplar CLS
        self._gt: np.ndarray | None = None                 # (H, W) bool full-image GT class foreground

    # ------------------------------------------------------------------ reference
    def set_reference(
        self,
        backbone,
        ref_image: "np.ndarray | list[np.ndarray]",
        ref_masks: list[np.ndarray],
        negative_masks: list[np.ndarray] | None,
        cfg,
    ) -> None:
        """Build the per-exemplar CLS bank from a padded crop of each exemplar mask.

        Only the CLS tokens are needed (the foreground is oracled, so no foreground prototypes are
        built). ``negative_masks`` is accepted for interface parity and ignored."""
        images, valid = normalize_reference(ref_image, ref_masks)
        boxes = [_mask_bbox(m, cfg.pad_frac) for m in valid]
        crops = [img[y0:y1, x0:x1] for img, (y0, y1, x0, x1) in zip(images, boxes)]
        embedded = featlib.embed_batch(backbone, crops, chunk=8, standardize=cfg.standardize)
        cls_stack = torch.stack([cls for _, cls in embedded])
        self.exemplar_cls = featlib.l2_normalize(cls_stack, dim=1).to(embedded[0][0].device)  # (S, D)

    # ------------------------------------------------------------------ target GT
    def set_target_foreground(self, gt_foreground: np.ndarray) -> None:
        """Inject the target image's GT class foreground (``(H, W)`` bool) for the current image.

        Called once per target image by the cascade. It is transient per-image state — a cached
        extractor reused across inter-protocol targets simply has it overwritten each image."""
        self._gt = np.asarray(gt_foreground).astype(bool)

    # ------------------------------------------------------------------- predict
    def predict(
        self, target_feat: torch.Tensor, *, cls: torch.Tensor | None = None,
        box: tuple[int, int, int, int] | None = None, return_internals: bool = False,
    ) -> GateResult:
        """Return the GT foreground of the crop ``box`` resized to ``target_feat``'s patch grid."""
        if self._gt is None:
            raise RuntimeError(
                "OracleExtractor.predict called before set_target_foreground — the cascade must "
                "receive gt_foreground for the oracle extractor."
            )
        if box is None:
            raise RuntimeError(
                "OracleExtractor.predict needs the crop box to slice the GT foreground; the cascade "
                "must pass box=region.box."
            )
        hp, wp, _ = target_feat.shape
        y0, y1, x0, x1 = box
        crop_gt = self._gt[y0:y1, x0:x1]
        foreground = (
            featlib.resize_mask_to_grid(crop_gt, (hp, wp))
            if crop_gt.any() else np.zeros((hp, wp), dtype=bool)
        )
        score_map = foreground.astype(np.float32)                # perfect confidence: 1 on the GT

        internals: dict = {}
        if return_internals:
            internals = {
                "forward_sim": score_map,
                "candidate_mask": foreground,
                "foreground": foreground,
                "tau_used": 0.5,                                 # nominal cut for the viz caption
            }
        return GateResult(
            foreground=foreground,
            score_map=score_map,
            exemplar_cls=self.exemplar_cls.cpu().numpy(),
            internals=internals,
        )
