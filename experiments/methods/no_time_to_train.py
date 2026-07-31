"""NoTimeToTrainMethod — "No Time to Train!" (Espinosa et al.) as an in-context baseline.

NTT is a *training-free* reference-based instance segmenter: it uses a frozen DINO encoder to
match a target image against a small memory of reference-instance features, turns the best
matches into point prompts, and hands those to a promptable SAM to cut out instance masks.
The upstream repo (https://github.com/miquel-espinosa/no-time-to-train) is a PyTorch-Lightning
framework with DINOv2/SAM2 git submodules and COCO-pickle memory banks — none of which maps onto
this harness's per-image :class:`~experiments.methods.base.Method` interface. So this is a
faithful *reimplementation* of the core algorithm on Hugging Face parts: the repo's own DINO
backbone (:func:`experiments.methods.base.build_backbone`, DINOv2 or DINOv3) for features and HF
``facebook/sam2-hiera-large`` (``Sam2Model``) for prompt→mask. It sits beside ``sam3`` and
``semantic_cc`` in the same MLflow experiment. If the SAM 2 classes aren't importable this module
raises :class:`ImportError` at import time and the package auto-import skips it gracefully.

Pipeline (single class per :class:`EvalItem`, so NTT's multi-class memory collapses to one class):

1. **Memory bank** — embed the reference image (intra: the target itself; inter: the support
   ``exemplar_image``), take the DINO patch features under the exemplar masks, and reduce them to
   ``kmeans_k`` L2-normalized cluster centers (:func:`build_memory`). These centers are the whole
   "concept": one modal center per visual sub-part, matching the paper's representation
   aggregation (a single mean is the ``k == 1`` special case).
2. **Semantic matching → point prompts** — embed the target, score every target patch by its max
   cosine similarity to the centers (:func:`similarity_heatmap`), keep the top ``num_points``
   patches above ``point_thr``, and map their grid cells to target-image pixel coordinates
   (:func:`grid_topk_points`, :func:`grid_points_to_pixels`).
3. **Segment** — prompt SAM 2 with each point as its own positive object (``multimask_output``);
   keep the highest-SAM-IoU mask per point.
4. **Classify + dedup** — score each candidate mask by the mean center-similarity of the target
   patches under it (:func:`score_masks`), drop masks below ``score_thr`` or ``min_area``, then
   greedily suppress near-duplicate / nested masks (:func:`suppress_masks`) by IoU and
   intersection-over-self containment. Surviving masks + scores are the prediction.

The matching/aggregation/suppression logic is pure NumPy/torch at module level so it is unit-tested
without downloading SAM 2 or DINO weights; only :meth:`NoTimeToTrainMethod.predict` touches models.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

# Heavy dep at import time -> tolerant auto-import (see methods/__init__.py) skips this baseline
# if SAM 2 isn't installed, instead of breaking every run.
from transformers.models.sam2 import Sam2Model, Sam2Processor  # noqa: E402

from experiments.datasets import EvalItem
from experiments.methods.base import (
    Method,
    MethodPrediction,
    build_backbone,
    register_method,
)
from foveate import features as featlib


# ---------------------------------------------------------------------------
# Pure matching logic — now owned by :mod:`foveate.ntt`, which is also the Extract-slot arm.
# Re-exported here so the single-pass baseline and the foveated arm are provably the same
# algorithm rather than two implementations that drift.
# ---------------------------------------------------------------------------
from foveate.ntt import (  # noqa: E402
    build_memory,
    grid_points_to_pixels,
    grid_topk_points,
    score_masks,
    similarity_heatmap,
    spherical_kmeans,
    suppress_masks,
)

# ---------------------------------------------------------------------------
# Method
# ---------------------------------------------------------------------------
@register_method("no_time_to_train")
class NoTimeToTrainMethod(Method):
    """DINO-memory + SAM 2 reference-based segmentation ("No Time to Train!"), intra + inter."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.backbone = build_backbone(config.get("backbone", {}))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        mc = self.method_config
        self.standardize = bool(mc.get("standardize", True))     # matches semantic_cc / cascade
        self.kmeans_k = int(mc.get("kmeans_k", 8))               # memory centers per class
        self.num_points = int(mc.get("num_points", 100))         # top-k matched query points
        self.point_thr = float(mc.get("point_thr", 0.5))         # min cosine sim to keep a point
        self.score_thr = float(mc.get("score_thr", 0.5))         # min semantic score to keep a mask
        self.iou_thr = float(mc.get("iou_thr", 0.8))             # duplicate-suppression IoU
        self.containment_thr = float(mc.get("containment_thr", 0.9))  # nested-fragment suppression
        self.min_area = int(mc.get("min_area", 4))               # drop masks smaller than this (px)
        self.point_batch = int(mc.get("point_batch", 128))       # SAM 2 prompts per forward
        self.sam2_model = str(mc.get("sam2_model", "facebook/sam2-hiera-large"))

        token = _hf_token()
        self.processor = Sam2Processor.from_pretrained(self.sam2_model, token=token)
        self.model = Sam2Model.from_pretrained(self.sam2_model, token=token).to(self.device).eval()
        # Inter reuses one exemplar memory per (support image, class); cache so the reference is
        # embedded and clustered once per class instead of once per target image (mirrors
        # semantic_cc's reference cache).
        self._mem_cache: dict = {}

    # -- reference memory (intra: per-image; inter: cached per support image + class) ----------
    def _memory_for(self, item: EvalItem) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Return ``(centers, ref_grid_if_intra, dino_forwards)`` for ``item``.

        For intra the reference *is* the target image, so we embed it here and hand the grid back
        to reuse for target matching (one DINO forward total). For inter the reference is the
        support image; its centers are cached across all targets of the class (0 forwards on a hit).
        """
        if item.exemplar_image is None:                          # intra: reference == target
            ref_grid = featlib.embed_image(self.backbone, item.image, standardize=self.standardize)
            centers = build_memory(ref_grid, item.exemplar_masks, kmeans_k=self.kmeans_k)
            return centers, ref_grid, 1
        key = (id(item.exemplar_image), item.class_id)
        centers = self._mem_cache.get(key)
        if centers is None:
            ref_grid = featlib.embed_image(
                self.backbone, item.exemplar_image, standardize=self.standardize
            )
            centers = build_memory(ref_grid, item.exemplar_masks, kmeans_k=self.kmeans_k)
            self._mem_cache[key] = centers
            return centers, None, 1
        return centers, None, 0

    # -- SAM 2: point prompts -> best mask per point ------------------------------------------
    def _segment_points(
        self, image: np.ndarray, points_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prompt SAM 2 with each point as its own positive object; keep its best multimask.

        Encodes the image once and reuses the embeddings across point batches. Returns
        ``(masks (N, H, W) bool, sam_ious (N,))`` aligned with ``points_xy`` (empty if no points).
        """
        h, w = image.shape[:2]
        if points_xy.shape[0] == 0:
            return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.float32)

        base = self.processor(images=[image], return_tensors="pt")
        original_sizes = base["original_sizes"]
        with torch.inference_mode():
            image_embeddings = self.model.get_image_embeddings(base["pixel_values"].to(self.device))

        masks_out: list[np.ndarray] = []
        ious_out: list[np.ndarray] = []
        for start in range(0, points_xy.shape[0], self.point_batch):
            chunk = points_xy[start:start + self.point_batch]
            # One object per point: input_points (1, P, 1, 2), input_labels (1, P, 1) all positive.
            pts = [[[[float(x), float(y)]] for x, y in chunk]]
            lbls = [[[1] for _ in chunk]]
            inputs = self.processor(
                input_points=pts, input_labels=lbls,
                original_sizes=original_sizes, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                out = self.model(
                    input_points=inputs["input_points"],
                    input_labels=inputs["input_labels"],
                    image_embeddings=image_embeddings,
                    multimask_output=True,
                )
            # pred_masks (1, P, M, h_low, w_low); post_process -> (P, M, H, W) bool at full res.
            proc = self.processor.post_process_masks(
                out.pred_masks, original_sizes, binarize=True
            )[0]
            ious = out.iou_scores[0]                              # (P, M)
            best = ious.argmax(dim=1)                             # best multimask per point
            proc = proc[torch.arange(proc.shape[0]), best]       # (P, H, W)
            masks_out.append(proc.cpu().numpy().astype(bool))
            ious_out.append(ious[torch.arange(ious.shape[0]), best].float().cpu().numpy())

        return np.concatenate(masks_out, axis=0), np.concatenate(ious_out, axis=0)

    def predict(
        self, item: EvalItem, observer: Callable[[dict], None] | None = None
    ) -> MethodPrediction:
        h, w = item.image.shape[:2]
        centers, ref_grid, n_embeds = self._memory_for(item)
        if centers.shape[0] == 0:                                 # no reference foreground patches
            return MethodPrediction(masks=np.zeros((0, h, w), bool), scores=np.zeros((0,)), n_embeds=n_embeds)

        # Target features: reuse the intra reference grid (same image); else embed the target.
        if ref_grid is not None:
            tar_grid = ref_grid
        else:
            tar_grid = featlib.embed_image(self.backbone, item.image, standardize=self.standardize)
            n_embeds += 1
        hp, wp = tar_grid.shape[:2]

        # Semantic matching -> top-k query points -> pixel coordinates.
        heat = similarity_heatmap(tar_grid, centers)
        rc = grid_topk_points(heat, self.num_points, self.point_thr)
        points_xy = grid_points_to_pixels(rc, (hp, wp), (h, w))

        masks, _sam_ious = self._segment_points(item.image, points_xy)
        if masks.shape[0] == 0:
            return MethodPrediction(masks=np.zeros((0, h, w), bool), scores=np.zeros((0,)), n_embeds=n_embeds)

        # Classify each candidate by center-similarity of the target patches under it.
        mask_grids = torch.stack([
            torch.from_numpy(featlib.resize_mask_to_grid(m, (hp, wp))) for m in masks
        ]).to(tar_grid.device)
        scores = score_masks(tar_grid, mask_grids, centers).cpu().numpy()

        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        keep = (scores >= self.score_thr) & (areas >= self.min_area)
        masks, scores = masks[keep], scores[keep]
        if masks.shape[0] == 0:
            return MethodPrediction(masks=np.zeros((0, h, w), bool), scores=np.zeros((0,)), n_embeds=n_embeds)

        kept = suppress_masks(
            masks, scores, iou_thr=self.iou_thr, containment_thr=self.containment_thr
        )
        return MethodPrediction(
            masks=masks[kept], scores=scores[kept].astype(np.float64), n_embeds=n_embeds
        )

    def param_blocks(self) -> dict[str, dict[str, Any]]:
        return {
            "no_time_to_train": {
                "sam2_model": self.sam2_model,
                "standardize": self.standardize,
                "kmeans_k": self.kmeans_k,
                "num_points": self.num_points,
                "point_thr": self.point_thr,
                "score_thr": self.score_thr,
                "iou_thr": self.iou_thr,
                "containment_thr": self.containment_thr,
                "min_area": self.min_area,
            }
        }


def _hf_token() -> str | None:
    """HF access token from the environment, loading ``./.env`` first (mirrors sam3._hf_token)."""
    try:
        from pathlib import Path

        from dotenv import load_dotenv

        if Path(".env").exists():
            load_dotenv(".env", override=False)
    except ImportError:
        pass
    return os.environ.get("HF_TOKEN")
