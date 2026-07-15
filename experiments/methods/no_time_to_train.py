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
# Pure matching logic (no model needed — unit-tested in isolation)
# ---------------------------------------------------------------------------
def spherical_kmeans(feats: torch.Tensor, k: int, *, n_iter: int = 25, seed: int = 0) -> torch.Tensor:
    """Cosine (spherical) k-means over ``feats`` ``(M, D)`` → ``(k', D)`` L2-normalized centers.

    ``feats`` are assumed L2-normalized (patch features from :func:`foveate.features.embed_image`),
    so a dot product is cosine similarity. ``k`` is clamped to ``M`` (can't have more centers than
    points); empty clusters keep their previous center rather than collapsing to zero. Mirrors the
    upstream ``kmeans`` (argmax-cosine assignment, mean-then-renormalize update).
    """
    m, d = feats.shape
    k = int(min(k, m))
    if k <= 0:
        return feats.new_zeros((0, d))
    g = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.randperm(m, generator=g)[:k].to(feats.device)
    centers = F.normalize(feats[init], p=2, dim=-1)
    for _ in range(n_iter):
        assign = (feats @ centers.t()).argmax(dim=1)          # [M] nearest center per patch
        new = centers.clone()
        for j in range(k):
            sel = feats[assign == j]
            if sel.shape[0] > 0:
                new[j] = sel.mean(dim=0)
        centers = F.normalize(new, p=2, dim=-1)
    return centers


def build_memory(
    ref_grid: torch.Tensor, exemplar_masks: list[np.ndarray], *, kmeans_k: int
) -> torch.Tensor:
    """Reduce the reference foreground patches to ``kmeans_k`` cluster centers.

    ``ref_grid`` is the ``(Hp, Wp, D)`` L2-normalized patch grid of the reference image; each mask
    in ``exemplar_masks`` is a full-resolution boolean mask on that image. Returns ``(k', D)``
    normalized centers (``k' == 0`` if no mask covers any patch — the caller then predicts nothing).
    """
    patches = featlib.stack_exemplar_patches(ref_grid, exemplar_masks)   # (M, D), normalized
    if patches.shape[0] == 0:
        return ref_grid.new_zeros((0, ref_grid.shape[-1]))
    return spherical_kmeans(patches, kmeans_k)


def similarity_heatmap(tar_grid: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Per-patch max cosine similarity of the target grid to the memory centers → ``(Hp, Wp)``.

    Both inputs are L2-normalized, so ``tar @ centers.T`` is cosine similarity in ``[-1, 1]``; we
    take the max over centers (best-matching modal sub-part) per patch. Empty memory → all ``-1``.
    """
    hp, wp, d = tar_grid.shape
    if centers.shape[0] == 0:
        return tar_grid.new_full((hp, wp), -1.0)
    sim = tar_grid.reshape(-1, d) @ centers.t()          # (Hp*Wp, k)
    return sim.max(dim=1).values.reshape(hp, wp)


def grid_topk_points(heat: torch.Tensor, num_points: int, thr: float) -> np.ndarray:
    """Top-``num_points`` grid cells of ``heat`` with similarity ``>= thr`` → ``(N, 2)`` ``[row, col]``.

    Fewer than ``num_points`` rows come back if the threshold prunes them (``N`` may be 0). Ordered
    by descending similarity so downstream suppression sees the strongest matches first.
    """
    hp, wp = heat.shape
    flat = heat.reshape(-1)
    n = int(min(num_points, flat.numel()))
    vals, idx = torch.topk(flat, k=n)
    keep = vals >= thr
    idx = idx[keep]
    rows = (idx // wp).cpu().numpy()
    cols = (idx % wp).cpu().numpy()
    return np.stack([rows, cols], axis=1).astype(np.int64) if idx.numel() else np.zeros((0, 2), np.int64)


def grid_points_to_pixels(
    rc: np.ndarray, grid_hw: tuple[int, int], img_hw: tuple[int, int]
) -> np.ndarray:
    """Map ``[row, col]`` patch-grid cells to ``[x, y]`` pixel coordinates at each cell center.

    ``grid_hw`` is ``(Hp, Wp)``; ``img_hw`` is the target image ``(H, W)``. A patch cell ``(r, c)``
    maps to the pixel at its center: ``x = (c + 0.5) / Wp * W``, ``y = (r + 0.5) / Hp * H``.
    """
    hp, wp = grid_hw
    h, w = img_hw
    if rc.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    xs = (rc[:, 1].astype(np.float32) + 0.5) / wp * w
    ys = (rc[:, 0].astype(np.float32) + 0.5) / hp * h
    return np.stack([xs, ys], axis=1).astype(np.float32)


def score_masks(
    tar_grid: torch.Tensor, mask_grids: torch.Tensor, centers: torch.Tensor
) -> torch.Tensor:
    """Semantic score in ``[0, 1]`` for each candidate mask against the memory centers.

    ``mask_grids`` is ``(N, Hp, Wp)`` boolean (candidate masks resized to the patch grid). Each
    mask's score is the mean over its foreground patches of the max center cosine similarity,
    mapped from ``[-1, 1]`` to ``[0, 1]``. A mask with no foreground patch scores 0.
    """
    n = mask_grids.shape[0]
    if n == 0 or centers.shape[0] == 0:
        return tar_grid.new_zeros((n,))
    hp, wp, d = tar_grid.shape
    patch_sim = (tar_grid.reshape(-1, d) @ centers.t()).max(dim=1).values  # (Hp*Wp,) in [-1,1]
    flat_masks = mask_grids.reshape(n, -1).to(patch_sim.dtype)             # (N, Hp*Wp)
    counts = flat_masks.sum(dim=1)
    summed = flat_masks @ patch_sim                                        # (N,)
    mean_sim = torch.where(counts > 0, summed / counts.clamp(min=1), counts.new_full((n,), -1.0))
    return ((mean_sim + 1.0) * 0.5).clamp(0.0, 1.0)


def suppress_masks(
    masks: np.ndarray, scores: np.ndarray, *, iou_thr: float, containment_thr: float
) -> np.ndarray:
    """Greedy score-ranked suppression of duplicate/nested masks → indices to keep (desc. score).

    Walking masks from highest score down, a candidate is dropped if, against any already-kept
    mask, its IoU exceeds ``iou_thr`` (near-duplicate) or its intersection-over-self exceeds
    ``containment_thr`` (it is mostly contained inside a stronger mask — a fragment/part). Pure
    geometry, vectorized over pixels; ``masks`` is ``(N, H, W)`` bool.
    """
    n = masks.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    flat = masks.reshape(n, -1).astype(np.float32)
    areas = flat.sum(axis=1)                          # (N,)
    inter = flat @ flat.T                             # (N, N) pairwise intersection
    order = np.argsort(-scores)                       # strongest first
    kept: list[int] = []
    for i in order:
        if areas[i] <= 0:
            continue
        drop = False
        for j in kept:
            ai, aj, ij = areas[i], areas[j], inter[i, j]
            iou = ij / (ai + aj - ij + 1e-6)
            containment = ij / (ai + 1e-6)            # fraction of i inside the kept mask j
            if iou > iou_thr or containment > containment_thr:
                drop = True
                break
        if not drop:
            kept.append(int(i))
    return np.asarray(kept, dtype=np.int64)


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
