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

import cv2
import numpy as np
import torch

from foveate import features as featlib
from foveate.foreground import GateResult, normalize_reference


def cfg_mode(cfg) -> str:
    """Oracle downsampling semantics: ``any`` (proposal, default) or ``center``."""
    return str(getattr(cfg, "oracle_coverage", "any"))


def mask_bbox(mask: np.ndarray, pad_frac: float) -> tuple[int, int, int, int]:
    """Padded bbox of a binary mask (patch-scale framing shared with the cascade's crops)."""
    ys, xs = np.where(mask)
    h, w = mask.shape
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    py, px = int((y1 - y0) * pad_frac), int((x1 - x0) * pad_frac)
    return max(0, y0 - py), min(h, y1 + py), max(0, x0 - px), min(w, x1 + px)


#: Pre-rename alias (this used to be private to this module).
_mask_bbox = mask_bbox


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
        self._gt_labels: np.ndarray | None = None          # (H, W) int per-instance GT ids (0 = bg)

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
        boxes = [mask_bbox(m, cfg.pad_frac) for m in valid]
        crops = [img[y0:y1, x0:x1] for img, (y0, y1, x0, x1) in zip(images, boxes)]
        embedded = featlib.embed_batch(backbone, crops, chunk=8, standardize=cfg.standardize)
        cls_stack = torch.stack([cls for _, cls in embedded])
        self.exemplar_cls = featlib.l2_normalize(cls_stack, dim=1).to(embedded[0][0].device)  # (S, D)

    # ------------------------------------------------------------------ target GT
    def set_target_foreground(self, gt_foreground: np.ndarray) -> None:
        """Inject the target image's GT class foreground for the current image.

        ``gt_foreground`` is either a ``(H, W)`` bool union mask or a ``(H, W)`` int instance-label
        map (``0`` = background, ``i`` = the ``i``-th GT instance) — the label map additionally
        carries the per-instance identity the :class:`OracleCCExtractor` needs. A bool mask is just
        the single-instance case (every foreground pixel labelled ``1``).

        Called once per target image by the cascade. It is transient per-image state — a cached
        extractor reused across inter-protocol targets simply has it overwritten each image."""
        labels = np.asarray(gt_foreground).astype(np.int32)
        self._gt_labels = labels
        self._gt = labels > 0

    @property
    def gt_labels(self) -> "np.ndarray | None":
        """The injected ``(H, W)`` instance-label map, or ``None`` before ``set_target_foreground``.

        Read by the monolithic :class:`foveate.extract.OracleExtractor`, which composes this class
        for the exemplar bank and the mask → grid path and adds the per-instance decomposition.
        """
        return self._gt_labels

    def _cls_bank(self) -> "np.ndarray | None":
        """The exemplar CLS stack as numpy, or ``None`` when no reference was set.

        The monolithic Extract-slot oracle composes this class for its foreground alone and reads
        ``g`` from its own scorer, so a bank-less instance is a legitimate state — ``predict`` must
        not require a reference it does not use.
        """
        return None if self.exemplar_cls is None else self.exemplar_cls.cpu().numpy()

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
        # ``any``-overlap, not centre sampling: the Where stage answers "where COULD the concept
        # be", and the cascade then foveates onto what it proposes. Centre sampling deletes anything
        # smaller than a patch before the recursion ever sees it, which is not a Where error the
        # oracle should inherit — it is a downsampling artefact. Under any-overlap every instance
        # marks at least one patch, so an upper-bound Where really is an upper bound.
        foreground = (
            featlib.resize_mask_to_grid(crop_gt, (hp, wp), mode=cfg_mode(self.cfg))
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
            exemplar_cls=self._cls_bank(),
            internals=internals,
        )


def _instance_seams(lab_grid: np.ndarray) -> np.ndarray:
    """Patches on a border between two *different* GT instances (8-neighbourhood).

    A patch is a seam if any of its 8 neighbours carries a different non-zero instance id. Removing
    the seam patches opens a ≥1-patch background gap between adjacent instances, so the cascade's
    connected-components Extract stage (whether 4- or 8-connectivity) recovers each instance as its
    own component instead of fusing touching ones.
    """
    hp, wp = lab_grid.shape
    padded = np.pad(lab_grid, 1)                                  # zero border → edges never seam-match
    seam = np.zeros((hp, wp), dtype=bool)
    core = lab_grid
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neigh = padded[1 + dy:1 + dy + hp, 1 + dx:1 + dx + wp]
            seam |= (core > 0) & (neigh > 0) & (core != neigh)
    return seam


class OracleCCExtractor(OracleExtractor):
    """Oracle foreground that also **pre-separates touching instances** into distinct components.

    The plain :class:`OracleExtractor` returns the GT *union*, so two GT instances that touch fall in
    one connected component and the cascade must tease them apart with the learned Split stage. This
    variant instead uses the GT per-instance identity (the label map from
    :meth:`set_target_foreground`) to carve the seam patches between different instances, so the
    Extract stage's connected components recover each instance directly — the upper bound where **both**
    the foreground and the instance boundaries are perfect. Comparing it against ``oracle`` isolates
    how much the pipeline is limited by imperfect instance separation versus imperfect foreground.
    """

    def predict(
        self, target_feat: torch.Tensor, *, cls: torch.Tensor | None = None,
        box: tuple[int, int, int, int] | None = None, return_internals: bool = False,
    ) -> GateResult:
        """GT foreground of the crop with inter-instance seams carved so CC separates instances."""
        if self._gt_labels is None:
            raise RuntimeError(
                "OracleCCExtractor.predict called before set_target_foreground — the cascade must "
                "receive gt_foreground for the oracle extractor."
            )
        if box is None:
            raise RuntimeError(
                "OracleCCExtractor.predict needs the crop box to slice the GT foreground; the cascade "
                "must pass box=region.box."
            )
        hp, wp, _ = target_feat.shape
        y0, y1, x0, x1 = box
        crop_labels = self._gt_labels[y0:y1, x0:x1]
        # Nearest-resize the integer label map onto the patch grid: a patch takes the instance id at
        # its source-pixel centre, so ``lab_grid > 0`` is exactly the plain oracle's union foreground
        # (same centre-pixel sampling) while carrying which instance each patch belongs to.
        if crop_labels.any():
            lab_grid = featlib.resize_labels_to_grid(crop_labels, (hp, wp))
        else:
            lab_grid = np.zeros((hp, wp), dtype=crop_labels.dtype)
        foreground = (lab_grid > 0) & ~_instance_seams(lab_grid)
        score_map = foreground.astype(np.float32)

        internals: dict = {}
        if return_internals:
            internals = {
                "forward_sim": score_map,
                "candidate_mask": foreground,
                "foreground": foreground,
                "instance_labels": lab_grid,             # per-patch GT instance id (viz)
                "tau_used": 0.5,
            }
        return GateResult(
            foreground=foreground,
            score_map=score_map,
            exemplar_cls=self._cls_bank(),
            internals=internals,
        )
