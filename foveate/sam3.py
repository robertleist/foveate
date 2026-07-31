"""SAM 3 as an Extract slot — a promptable foundation segmenter, foveated.

The second *monolithic* extractor, and the one the Where × Extract collapse was really aimed at
(roadmap §1.5): SAM 3 returns instance masks with scores directly, so flattening them into a
per-patch foreground and re-splitting them — which the old contract required — would have thrown
away the instance information it had just produced and paid to rebuild it worse.

Prompting, per crop: paste the exemplar crop and the current crop side by side on one canvas, prompt
with the exemplar-side boxes, and crop the returned masks back to the target half. INSID3 found this
*concatenation* protocol beats video-style propagation for cross-image visual exemplars, and it has
the property this slot needs — it works identically whether the exemplars live on the target image
(intra) or on a separate support image (inter), so the arm is one code path rather than two.

**What foveating it changes.** SAM 3 sees a canvas whose target half is a *crop*, not the whole
frame, so the object it is asked about occupies far more of its input each level down. The base
model, its weights and its thresholds are untouched — the only thing that changes is the framing,
which is exactly the comparison the paper wants: *the same base extractor, once vs. foveated*, and
the answer to the §A5.4 "why not just tile at high resolution" rebuttal.

**Cost.** One SAM 3 forward per visited crop, counted in :attr:`n_segment_calls`. That is a real and
large cost next to ``n_embeds`` and it belongs in §A4.2 beside the accuracy — an adaptive-compute
claim that hides the segmenter calls is not an adaptive-compute claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from foveate import features as featlib
from foveate.extract import ExtractResult, build_exemplar_bank


# ---------------------------------------------------------------------------
# Pure geometry (no model needed)
# ---------------------------------------------------------------------------
def masks_to_boxes(masks: list[np.ndarray]) -> list[list[int]]:
    """Tight ``xyxy`` pixel boxes for each non-empty mask; empty masks are skipped.

    SAM 3's geometry encoder cannot use a degenerate box, so a mask that covers nothing simply does
    not become a prompt.
    """
    boxes: list[list[int]] = []
    for mask in masks:
        ys, xs = np.where(np.asarray(mask, dtype=bool))
        if ys.size == 0:
            continue
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
    return boxes


def build_concat_canvas(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, int]:
    """Paste ``left`` and ``right`` side by side → ``(canvas, x_offset_of_right)``.

    The canvas is as tall as the taller image and both are placed at the top-left of their half, so
    ``left``'s coordinates are unchanged (its prompt boxes need no transform) and ``right``'s are
    shifted by ``x_offset`` only in x.
    """
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    canvas = np.zeros((max(lh, rh), lw + rw, 3), dtype=np.uint8)
    canvas[:lh, :lw] = left[..., :3]
    canvas[:rh, lw:lw + rw] = right[..., :3]
    return canvas, lw


def crop_canvas_masks_to_target(masks: np.ndarray, scores: np.ndarray, x_offset: int,
                                target_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Cut canvas-space masks back to the target half, dropping anything that landed on the left.

    A mask with no pixels right of ``x_offset`` is a detection of the *exemplar*, not of the target,
    and must not be returned — otherwise the prompt would count as its own answer.
    """
    h, w = target_hw
    if masks.shape[0] == 0:
        return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.float64)
    sub = np.asarray(masks, dtype=bool)[:, :h, x_offset:x_offset + w]
    keep = sub.reshape(sub.shape[0], -1).any(axis=1)
    return sub[keep], np.asarray(scores, dtype=np.float64)[keep]


# ---------------------------------------------------------------------------
# The Extract slot
# ---------------------------------------------------------------------------
class SAM3Extractor:
    """``sam3`` — SAM 3 prompted with the exemplar boxes on a concatenated canvas, per crop."""

    def __init__(self, cfg) -> None:
        from transformers.models.sam3 import Sam3Model, Sam3Processor   # deferred heavy import

        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.exemplar_cls: torch.Tensor | None = None
        self._exemplar: np.ndarray | None = None          # the exemplar crop pasted on every canvas
        self._boxes: list[list[int]] = []                 # its prompt boxes, in canvas coords
        self.n_segment_calls = 0
        self.processor = Sam3Processor.from_pretrained(cfg.sam3_model)
        self.model = Sam3Model.from_pretrained(cfg.sam3_model).to(self.device).eval()

    # ------------------------------------------------------------------ reference
    def set_reference(self, backbone, ref_image, ref_masks, negative_masks, cfg) -> None:
        """Cache one exemplar canvas half and its prompt boxes, plus the CLS bank for ``g``.

        The exemplars are cropped to their joint bounding box so the canvas carries the concept at a
        comparable scale to the target crop rather than a whole frame around it — the same reason
        the reference crops are padded by ``pad_frac`` everywhere else.
        """
        self.exemplar_cls, images, masks = build_exemplar_bank(backbone, ref_image, ref_masks, cfg)
        # One shared support half: the first reference image, cropped to cover every exemplar on it.
        img = images[0]
        on_first = [m for i, m in zip(images, masks) if i is img]
        union = np.logical_or.reduce(on_first)
        ys, xs = np.where(union)
        h, w = union.shape
        py, px = int((ys.max() - ys.min()) * cfg.pad_frac), int((xs.max() - xs.min()) * cfg.pad_frac)
        y0, y1 = max(0, int(ys.min()) - py), min(h, int(ys.max()) + 1 + py)
        x0, x1 = max(0, int(xs.min()) - px), min(w, int(xs.max()) + 1 + px)
        self._exemplar = np.asarray(img)[y0:y1, x0:x1]
        self._boxes = masks_to_boxes([m[y0:y1, x0:x1] for m in on_first])

    # ------------------------------------------------------------------- segment
    def _segment(self, canvas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One SAM 3 forward on ``canvas`` with the cached exemplar boxes as positive prompts."""
        kwargs: dict[str, Any] = {"images": [canvas], "text": "visual", "return_tensors": "pt"}
        if self._boxes:
            kwargs["input_boxes"] = [self._boxes]
            kwargs["input_boxes_labels"] = torch.tensor([[1] * len(self._boxes)], dtype=torch.int64)
        inputs = self.processor(**kwargs).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        self.n_segment_calls += 1
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=self.cfg.sam3_threshold,
            mask_threshold=self.cfg.sam3_mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        return results["masks"].cpu().numpy().astype(bool), results["scores"].cpu().numpy()

    # ------------------------------------------------------------------- extract
    def extract(self, feat, *, cls=None, box=None, image=None,
                return_internals=False) -> ExtractResult:
        hp, wp = feat.shape[:2]
        empty = np.zeros((hp, wp), dtype=bool)
        if self._exemplar is None:
            raise RuntimeError("SAM3Extractor used before set_reference.")
        if image is None:
            raise RuntimeError(
                "SAM3Extractor.extract needs the crop pixels; the cascade must pass image=."
            )

        canvas, x_offset = build_concat_canvas(self._exemplar, image)
        canvas_masks, canvas_scores = self._segment(canvas)
        masks, scores = crop_canvas_masks_to_target(canvas_masks, canvas_scores, x_offset,
                                                    image.shape[:2])
        if masks.shape[0] == 0:
            return ExtractResult([], empty, np.zeros((hp, wp), np.float32), {})

        grids = [featlib.resize_mask_to_grid(m, (hp, wp), mode="any") for m in masks]
        keep = [i for i, g in enumerate(grids) if g.any()]
        instances = [grids[i] for i in keep]
        pixel = [masks[i] for i in keep]
        foreground = np.logical_or.reduce(instances) if instances else empty
        # Per-patch confidence: the score of the strongest instance covering each patch.
        score_map = np.zeros((hp, wp), np.float32)
        for i in keep:
            score_map = np.maximum(score_map, grids[i].astype(np.float32) * float(scores[i]))
        internals = {"foreground": foreground, "forward_sim": score_map} if return_internals else {}
        return ExtractResult(instances=instances, foreground=foreground, score_map=score_map,
                             internals=internals, masks=pixel or None)

    def param_blocks(self) -> dict[str, Any]:
        return {"sam3_model": self.cfg.sam3_model, "n_segment_calls": self.n_segment_calls}
