"""NTT as an Extract slot — "No Time to Train!" (Espinosa et al.), foveated.

A *monolithic* extractor: it answers "which instances of the concept are on this crop?" in one step,
with no per-patch foreground in between. That is the whole reason the Where × Extract split was
collapsed (roadmap §1.5) — NTT produces instance masks directly, and the old contract would have
forced them through a foreground and re-split them.

The algorithm, per crop:

1. **Memory** (once, at ``set_reference``) — the exemplars' DINO patch features, reduced to
   ``ntt_kmeans_k`` L2-normalized spherical-k-means centers. One modal center per visual sub-part.
2. **Match** — score every patch of the crop by its max cosine to those centers, and take the
   top-``ntt_num_points`` above ``ntt_point_thr`` as point prompts.
3. **Segment** — prompt SAM 2 with each point as its own positive object, keep the best multimask.
4. **Classify + dedup** — score each mask by the mean center-similarity of the patches under it,
   drop the weak and the tiny, then suppress duplicates and nested fragments.

**What foveating it changes.** Step 2 runs on the crop's patch grid, which the cascade has *already*
embedded for the re-identification score — so the DINO half of NTT is free, and the marginal cost of
a crop is one SAM 2 forward. Step 1 is paid once per image. And the recursion hands NTT a tighter
crop each level, so the same fixed patch budget is spent on progressively fewer objects: the
matching heat-map that has to separate 30 instances in one full-frame pass has to separate two or
three at depth 3. That is the meta-algorithm claim — *the same base extractor, foveated* — with the
base extractor's own hyperparameters untouched.

The matching logic is pure NumPy/torch at module level so it is unit-testable without downloading
SAM 2; only :class:`NTTExtractor` touches a model. :mod:`experiments.methods.no_time_to_train` (the
single-pass NTT *baseline*) re-exports these, so the baseline and the foveated arm are provably the
same algorithm.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from foveate import features as featlib
from foveate.extract import (
    ExtractResult,
    build_exemplar_bank,
    scale_matched_view,
    scale_octave,
)


# ---------------------------------------------------------------------------
# Pure matching logic (no model needed)
# ---------------------------------------------------------------------------
def spherical_kmeans(feats: torch.Tensor, k: int, *, n_iter: int = 25, seed: int = 0) -> torch.Tensor:
    """Cosine (spherical) k-means over ``feats`` ``(M, D)`` → ``(k', D)`` L2-normalized centers.

    ``feats`` are assumed L2-normalized (patch features from :func:`foveate.features.embed_image`),
    so a dot product is cosine similarity. ``k`` is clamped to ``M`` (can't have more centers than
    points); empty clusters keep their previous center rather than collapsing to zero.
    """
    m = feats.shape[0]
    k = int(min(max(k, 1), m))
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(m, generator=g)[:k].to(feats.device)
    centers = feats[idx].clone()
    for _ in range(n_iter):
        assign = (feats @ centers.t()).argmax(dim=1)
        for j in range(k):
            sel = feats[assign == j]
            if sel.shape[0]:
                centers[j] = featlib.l2_normalize(sel.mean(dim=0), dim=0)
    return centers


def build_memory(ref_grid: torch.Tensor, exemplar_masks: list[np.ndarray], *,
                 kmeans_k: int) -> torch.Tensor:
    """Reduce the reference foreground patches to ``kmeans_k`` cluster centers.

    ``ref_grid`` is the ``(Hp, Wp, D)`` L2-normalized patch grid of the reference image; each mask is
    a full-resolution boolean mask on that image. ``(k', D)`` normalized centers, or ``(0, D)`` when
    no mask covers a patch (the caller then proposes nothing).
    """
    patches = featlib.stack_exemplar_patches(ref_grid, exemplar_masks)   # (M, D), normalized
    if patches.shape[0] == 0:
        return ref_grid.new_zeros((0, ref_grid.shape[-1]))
    return spherical_kmeans(patches, kmeans_k)


def similarity_heatmap(tar_grid: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Per-patch max cosine similarity of the target grid to the memory centers → ``(Hp, Wp)``.

    Both inputs are L2-normalized, so ``tar @ centers.T`` is cosine similarity in ``[-1, 1]``; the
    max over centers is the best-matching modal sub-part. Empty memory → all ``-1``.
    """
    hp, wp, d = tar_grid.shape
    if centers.shape[0] == 0:
        return tar_grid.new_full((hp, wp), -1.0)
    sim = tar_grid.reshape(-1, d) @ centers.t()          # (Hp*Wp, k)
    return sim.max(dim=1).values.reshape(hp, wp)


def grid_topk_points(heat: torch.Tensor, num_points: int, thr: float) -> np.ndarray:
    """Top-``num_points`` grid cells of ``heat`` with similarity ``>= thr`` → ``(N, 2)`` ``[row, col]``.

    Ordered by descending similarity so downstream suppression sees the strongest matches first.
    """
    hp, wp = heat.shape
    flat = heat.reshape(-1)
    n = int(min(num_points, flat.numel()))
    vals, idx = torch.topk(flat, k=n)
    idx = idx[vals >= thr]
    if idx.numel() == 0:
        return np.zeros((0, 2), np.int64)
    rows = (idx // wp).cpu().numpy()
    cols = (idx % wp).cpu().numpy()
    return np.stack([rows, cols], axis=1).astype(np.int64)


def grid_points_to_pixels(rc: np.ndarray, grid_hw: tuple[int, int],
                          img_hw: tuple[int, int]) -> np.ndarray:
    """Map ``[row, col]`` patch cells to ``[x, y]`` pixels at each cell **centre**.

    The same centre convention as :func:`foveate.features.resize_labels_to_grid` — a cell stands for
    the pixel interval it covers, and is addressed at that interval's middle.
    """
    hp, wp = grid_hw
    h, w = img_hw
    if rc.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    xs = (rc[:, 1].astype(np.float32) + 0.5) / wp * w
    ys = (rc[:, 0].astype(np.float32) + 0.5) / hp * h
    return np.stack([xs, ys], axis=1).astype(np.float32)


def score_masks(tar_grid: torch.Tensor, mask_grids: torch.Tensor,
                centers: torch.Tensor) -> torch.Tensor:
    """Semantic score in ``[0, 1]`` per candidate mask: mean center-similarity of its patches."""
    n = mask_grids.shape[0]
    if n == 0 or centers.shape[0] == 0:
        return tar_grid.new_zeros((n,))
    hp, wp, d = tar_grid.shape
    patch_sim = (tar_grid.reshape(-1, d) @ centers.t()).max(dim=1).values  # (Hp*Wp,) in [-1,1]
    flat_masks = mask_grids.reshape(n, -1).to(patch_sim.dtype)             # (N, Hp*Wp)
    counts = flat_masks.sum(dim=1)
    summed = flat_masks @ patch_sim
    mean_sim = torch.where(counts > 0, summed / counts.clamp(min=1), counts.new_full((n,), -1.0))
    return ((mean_sim + 1.0) * 0.5).clamp(0.0, 1.0)


def suppress_masks(masks: np.ndarray, scores: np.ndarray, *, iou_thr: float,
                   containment_thr: float) -> np.ndarray:
    """Greedy score-ranked suppression of duplicate/nested masks → indices to keep (desc. score)."""
    n = masks.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    flat = masks.reshape(n, -1).astype(np.float32)
    areas = flat.sum(axis=1)
    inter = flat @ flat.T
    kept: list[int] = []
    for i in np.argsort(-scores):
        if areas[i] <= 0:
            continue
        drop = False
        for j in kept:
            ai, aj, ij = areas[i], areas[j], inter[i, j]
            if (ij / (ai + aj - ij + 1e-6) > iou_thr) or (ij / (ai + 1e-6) > containment_thr):
                drop = True
                break
        if not drop:
            kept.append(int(i))
    return np.asarray(kept, dtype=np.int64)


# ---------------------------------------------------------------------------
# The Extract slot
# ---------------------------------------------------------------------------
class NTTExtractor:
    """``ntt`` — DINO memory matching + SAM 2, run on every crop the cascade visits.

    ``set_reference`` builds the memory centers and the exemplar CLS bank (the latter for ``g``,
    which is independent of this slot and must be comparable across arms); ``extract`` matches,
    prompts and returns instances at pixel resolution alongside their patch grids.

    :attr:`n_segment_calls` counts SAM 2 forwards, which is the cost axis this arm adds and the
    number §A4.2 has to report next to ``n_embeds``.
    """

    def __init__(self, cfg) -> None:
        from transformers.models.sam2 import Sam2Model, Sam2Processor   # deferred heavy import

        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.exemplar_cls: torch.Tensor | None = None
        self._centers: torch.Tensor | None = None      # memory at the reference's own framing
        self._backbone = None                          # kept to re-embed scale-matched views
        self._ref: list = []                           # (image, mask) pairs the memory is built from
        self.feature_stats = None                      # fixed (mean, std); read by the cascade
        self._by_octave: dict[int, torch.Tensor] = {}  # scale-matched memories, cached per octave
        self.n_segment_calls = 0
        self.processor = Sam2Processor.from_pretrained(cfg.sam2_model)
        self.model = Sam2Model.from_pretrained(cfg.sam2_model).to(self.device).eval()

    # ------------------------------------------------------------------ reference
    def set_reference(self, backbone, ref_image, ref_masks, negative_masks, cfg) -> None:
        """Build the memory centers from the exemplars, and the CLS bank for ``g``.

        Keeps the backbone and the reference pairs so the memory can be rebuilt at the framing of
        whatever crop is being extracted (:meth:`_memory_for`) — a memory is only comparable to a
        crop that frames its content at a similar scale.
        """
        self.exemplar_cls, images, masks = build_exemplar_bank(backbone, ref_image, ref_masks, cfg)
        self._backbone = backbone
        self._ref = list(zip(images, masks))
        if cfg.standardize and cfg.standardize_stats == "reference":
            # One fixed feature space for the whole run, taken from the first exemplar crop, so every
            # crop's features live in the same coordinates regardless of how it is framed.
            from foveate.oracle import mask_bbox

            img, m = self._ref[0]
            y0, y1, x0, x1 = mask_bbox(m, cfg.pad_frac)
            raw = backbone(backbone.preprocess(img[y0:y1, x0:x1]))[0]
            self.feature_stats = featlib.grid_stats(raw.permute(1, 2, 0).float())
        self._centers = self._build_memory(None)

    def _build_memory(self, target_hw) -> torch.Tensor:
        """Cluster the exemplar patches, optionally seen at ``target_hw``'s framing."""
        from foveate.oracle import mask_bbox

        cfg = self.cfg
        chunks = []
        for img, m in self._ref:
            if target_hw is not None:
                view, view_mask = scale_matched_view(img, m, target_hw)
            elif cfg.ntt_crop_reference:
                y0, y1, x0, x1 = mask_bbox(m, cfg.pad_frac)
                view, view_mask = img[y0:y1, x0:x1], m[y0:y1, x0:x1]
            else:
                view, view_mask = img, m
            if not view_mask.any():
                continue
            grid = featlib._standardize_and_norm(
                self._backbone(self._backbone.preprocess(view))[0], cfg.standardize,
                self.feature_stats)
            patches = featlib.stack_exemplar_patches(grid, [view_mask])
            if patches.shape[0]:
                chunks.append(patches)
        if not chunks:
            return self.exemplar_cls.new_zeros((0, self.exemplar_cls.shape[-1]))
        return spherical_kmeans(torch.cat(chunks, dim=0), cfg.ntt_kmeans_k)

    def _memory_for(self, target_hw) -> torch.Tensor:
        """The memory to match this crop against — scale-matched when ``ntt_scale_matched``.

        Rebuilding a view per crop would cost an encoder forward per crop, so views are bucketed by
        octave: the recursion halves a crop rather than nudging it, so a handful of buckets covers a
        whole image.
        """
        if not self.cfg.ntt_scale_matched:
            return self._centers
        key = scale_octave(target_hw)
        mem = self._by_octave.get(key)
        if mem is None:
            mem = self._by_octave[key] = self._build_memory(target_hw)
        return mem

    # ------------------------------------------------------------------- segment
    def _segment_points(self, image: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        """Prompt SAM 2 with each point as its own positive object → ``(N, H, W)`` bool masks.

        Degenerate crops are refused rather than segmented: the recursion can derive a child box a
        few pixels on a side from a sliver of an instance grid, and SAM 2 on a 3-pixel-tall image
        both means nothing and comes back with its axes swapped. The cascade's size floor stops such
        a crop from being *descended*, but it is still extracted once first.
        """
        h, w = image.shape[:2]
        if points_xy.shape[0] == 0 or min(h, w) < self.cfg.ntt_min_crop_side:
            return np.zeros((0, h, w), dtype=bool)

        base = self.processor(images=[image], return_tensors="pt")
        original_sizes = base["original_sizes"]
        with torch.inference_mode():
            embeddings = self.model.get_image_embeddings(base["pixel_values"].to(self.device))
        self.n_segment_calls += 1

        out_masks: list[np.ndarray] = []
        for start in range(0, points_xy.shape[0], self.cfg.ntt_point_batch):
            chunk = points_xy[start:start + self.cfg.ntt_point_batch]
            pts = [[[[float(x), float(y)]] for x, y in chunk]]     # one object per point
            lbls = [[[1] for _ in chunk]]
            inputs = self.processor(input_points=pts, input_labels=lbls,
                                    original_sizes=original_sizes,
                                    return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model(input_points=inputs["input_points"],
                                 input_labels=inputs["input_labels"],
                                 image_embeddings=embeddings, multimask_output=True)
            proc = self.processor.post_process_masks(out.pred_masks, original_sizes,
                                                     binarize=True)[0]
            best = out.iou_scores[0].argmax(dim=1)                 # best multimask per point
            proc = proc[torch.arange(proc.shape[0]), best]
            out_masks.append(proc.cpu().numpy().astype(bool))
        masks = np.concatenate(out_masks, axis=0)
        # The contract with the cascade is a crop-shaped mask; anything else is a processor quirk on
        # an odd aspect ratio, and pasting it into the crop would raise deep inside ``emit``.
        if masks.shape[1:] != (h, w):
            masks = np.stack([
                cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                for m in masks
            ]) if masks.shape[0] else np.zeros((0, h, w), dtype=bool)
        return masks

    # ------------------------------------------------------------------- extract
    def extract(self, feat, *, cls=None, box=None, image=None,
                return_internals=False) -> ExtractResult:
        cfg = self.cfg
        hp, wp = feat.shape[:2]
        empty = np.zeros((hp, wp), dtype=bool)
        if self._centers is None:
            raise RuntimeError("NTTExtractor used before set_reference.")
        if image is None:
            raise RuntimeError(
                "NTTExtractor.extract needs the crop pixels; the cascade must pass image=."
            )

        centers = self._memory_for(image.shape[:2])
        heat = similarity_heatmap(feat, centers)
        score_map = ((heat + 1.0) * 0.5).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
        pts = grid_points_to_pixels(
            grid_topk_points(heat, cfg.ntt_num_points, cfg.ntt_point_thr),
            (hp, wp), image.shape[:2],
        )
        masks = self._segment_points(image, pts)
        if masks.shape[0] == 0:
            return ExtractResult([], empty, score_map, {"heat": score_map} if return_internals else {})

        grids = np.stack([featlib.resize_mask_to_grid(m, (hp, wp), mode="any") for m in masks])
        scores = score_masks(feat, torch.from_numpy(grids).to(feat.device), centers)
        scores = scores.cpu().numpy()
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        ok = np.where((scores >= cfg.ntt_score_thr) & (areas >= cfg.ntt_min_area))[0]
        if ok.size == 0:
            return ExtractResult([], empty, score_map, {"heat": score_map} if return_internals else {})

        keep = ok[suppress_masks(masks[ok], scores[ok], iou_thr=cfg.ntt_iou_thr,
                                 containment_thr=cfg.ntt_containment_thr)]
        instances = [grids[i] for i in keep if grids[i].any()]
        pixel = [masks[i] for i in keep if grids[i].any()]
        foreground = np.logical_or.reduce(instances) if instances else empty
        internals = {"heat": score_map, "foreground": foreground} if return_internals else {}
        return ExtractResult(instances=instances, foreground=foreground, score_map=score_map,
                             internals=internals, masks=pixel or None)

    def param_blocks(self) -> dict[str, Any]:
        return {"sam2_model": self.cfg.sam2_model, "n_segment_calls": self.n_segment_calls}
