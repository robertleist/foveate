"""SAM3Method — Meta's SAM 3 as an in-context instance-segmentation baseline.

SAM 3 is a promptable foundation model that accepts a text concept plus visual prompts
(boxes/points/masks) and returns instance masks with scores. We wrap it behind the
:class:`~experiments.methods.base.Method` interface so it sits beside foveate in the paper's
comparison. The heavy weights (``facebook/sam3``) load once in ``__init__``; if the SAM 3
classes aren't importable this module raises :class:`ImportError` at import time and the
package's auto-import (see ``methods/__init__.py``) skips it gracefully.

Two cross-image protocols are exposed via the ``inter_mode`` method param:

* ``"concat"`` (default) — the important one. INSID3 found that for cross-image exemplars,
  *concatenating* the support and target images into one canvas and prompting SAM 3 with the
  exemplar boxes beats video-style propagation. We paste the exemplar on the LEFT and the
  target on the RIGHT of a padded canvas, prompt with the exemplar-side boxes, then crop the
  returned masks back to the target region and drop anything that landed in the exemplar half.
* ``"text"`` — ignore exemplar geometry; prompt with the class name as a text concept only.
  Reported as a separate, purely text-driven variant.

Intra-image (``exemplar_image is None``) always prompts SAM 3 on the single image with boxes
derived from the exemplar masks.

Score calibration: SAM 3 scores are ``sigmoid(class) * sigmoid(presence)`` — a product of two
probabilities, so they sit low. The HF-calibrated detection ``threshold`` default is 0.3;
0.5 over-filters and the model "finds almost nothing". ``mask_threshold`` (0.5) binarizes each
kept instance's mask.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from experiments.datasets import EvalItem
from experiments.methods.base import Method, MethodPrediction, register_method

# Raise ImportError at module import time if SAM 3 isn't available, so the package's
# tolerant auto-import degrades the run gracefully instead of crashing it.
import torch  # noqa: E402
from transformers.models.sam3 import Sam3Model, Sam3Processor  # noqa: E402


# ---------------------------------------------------------------------------
# Pure geometry helpers (no model needed — unit-tested in isolation)
# ---------------------------------------------------------------------------
def masks_to_boxes(masks: list[np.ndarray]) -> list[list[int]]:
    """Tight ``xyxy`` pixel boxes for each non-empty boolean mask.

    Empty masks are skipped (SAM 3's geometry encoder can't use a degenerate box). Boxes are
    inclusive-min / exclusive-max in pixel coords of whichever image the mask lives in.
    """
    boxes: list[list[int]] = []
    for mask in masks:
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        boxes.append([x0, y0, x1, y1])
    return boxes


def build_concat_canvas(
    exemplar_image: np.ndarray, target_image: np.ndarray
) -> tuple[np.ndarray, int]:
    """Paste ``exemplar`` (left) and ``target`` (right) side by side on one padded canvas.

    Why: for cross-image exemplars, SAM 3 works best when the visual prompt and the query live
    in the *same* frame (INSID3's "concatenate then prompt with an exemplar" finding). The
    exemplar sits at the origin so its boxes are unchanged; the target is offset horizontally by
    the exemplar's width. Both images are top-aligned and the shorter one is zero-padded to the
    max height, so the target keeps its original ``(H, W)`` for an exact crop-back later.

    Returns ``(canvas, x_offset)`` where ``x_offset`` is the target's left edge on the canvas.
    """
    eh, ew = exemplar_image.shape[:2]
    th, tw = target_image.shape[:2]
    canvas_h = max(eh, th)
    canvas_w = ew + tw
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=exemplar_image.dtype)
    canvas[:eh, :ew] = exemplar_image          # exemplar at origin (its boxes stay valid)
    canvas[:th, ew:ew + tw] = target_image     # target offset right by the exemplar width
    return canvas, ew


def crop_canvas_masks_to_target(
    canvas_masks: np.ndarray,
    canvas_scores: np.ndarray,
    x_offset: int,
    target_hw: tuple[int, int],
    *,
    min_target_area: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop canvas-space masks back to the target region and drop exemplar-side detections.

    ``canvas_masks`` is ``(N, canvas_h, canvas_w)`` bool from ``post_process`` over the concat
    canvas. Each mask is cropped to the target window ``[x_offset : x_offset + tw]`` (top rows
    ``0 : th``); detections whose in-target area is below ``min_target_area`` (empty, or living
    almost entirely in the exemplar half) are dropped. Surviving masks come back at the target's
    original ``(th, tw)`` shape, aligned with their scores.
    """
    th, tw = target_hw
    if canvas_masks.shape[0] == 0:
        return (np.zeros((0, th, tw), dtype=bool), np.zeros((0,), dtype=canvas_scores.dtype))

    # Crop to the target window; the canvas is at least (th, x_offset + tw), so this is in-bounds.
    cropped = canvas_masks[:, :th, x_offset:x_offset + tw].astype(bool)
    keep = cropped.reshape(cropped.shape[0], -1).sum(axis=1) >= min_target_area
    return cropped[keep], canvas_scores[keep]


@register_method("sam3")
class SAM3Method(Method):
    """SAM 3 promptable segmentation as an in-context baseline (intra + cross-image)."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # SAM 3 scores are sigmoid(class)*sigmoid(presence) -> low; 0.3 is HF-calibrated.
        self.threshold = float(self.method_config.get("threshold", 0.3))
        self.mask_threshold = float(self.method_config.get("mask_threshold", 0.5))
        self.inter_mode = str(self.method_config.get("inter_mode", "concat"))
        if self.inter_mode not in ("concat", "text"):
            raise ValueError(f"inter_mode must be 'concat' or 'text', got {self.inter_mode!r}")

        token = _hf_token()
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=token)
        self.model = Sam3Model.from_pretrained("facebook/sam3", token=token).to(self.device)

    # -- SAM 3 forward + post-process on one image ----------------------------
    def _segment(
        self, image: np.ndarray, text: str, boxes: list[list[int]] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run SAM 3 on a single image with an optional set of positive exemplar boxes."""
        proc_kwargs: dict[str, Any] = {"images": [image], "text": text, "return_tensors": "pt"}
        if boxes:
            # Positive exemplars -> label 1; labels is a LongTensor of shape (batch, num_boxes).
            labels = torch.tensor([[1] * len(boxes)], dtype=torch.int64)
            proc_kwargs["input_boxes"] = [boxes]
            proc_kwargs["input_boxes_labels"] = labels

        inputs = self.processor(**proc_kwargs)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        masks = results["masks"].cpu().numpy().astype(bool)
        scores = results["scores"].cpu().numpy()
        return masks, scores

    def predict(
        self, item: EvalItem, observer: Callable[[dict], None] | None = None
    ) -> MethodPrediction:
        h, w = item.image.shape[:2]

        if item.exemplar_image is None:
            # INTRA: exemplars are in item.image; prompt with their boxes on the same image.
            boxes = masks_to_boxes(item.exemplar_masks)
            masks, scores = self._segment(item.image, "visual", boxes)

        elif self.inter_mode == "text":
            # Text variant: ignore exemplar geometry, prompt with the class name (or "visual").
            text = item.class_name or "visual"
            masks, scores = self._segment(item.image, text, boxes=None)

        else:  # inter_mode == "concat"
            # Concat variant: exemplar (left) + target (right) on one canvas; prompt with the
            # exemplar-side boxes (unchanged at the origin), then crop masks back to the target.
            canvas, x_offset = build_concat_canvas(item.exemplar_image, item.image)
            boxes = masks_to_boxes(item.exemplar_masks)  # in exemplar coords == canvas coords
            canvas_masks, canvas_scores = self._segment(canvas, "visual", boxes)
            masks, scores = crop_canvas_masks_to_target(
                canvas_masks, canvas_scores, x_offset, (h, w)
            )

        if masks.shape[0] == 0:
            masks = np.zeros((0, h, w), dtype=bool)
            scores = np.zeros((0,), dtype=np.float64)
        # SAM 3 doesn't expose a backbone-forward count comparable to foveate's, so report 0.
        return MethodPrediction(masks=masks, scores=scores.astype(np.float64), n_embeds=0)

    def param_blocks(self) -> dict[str, dict[str, Any]]:
        return {
            "sam3": {
                "threshold": self.threshold,
                "mask_threshold": self.mask_threshold,
                "inter_mode": self.inter_mode,
            }
        }


def _hf_token() -> str | None:
    """HF access token from the environment, loading ``./.env`` first if python-dotenv is here.

    Mirrors ``run._load_dotenv``: env vars already set take precedence over the file.
    """
    try:
        from dotenv import load_dotenv
        from pathlib import Path

        if Path(".env").exists():
            load_dotenv(".env", override=False)
    except ImportError:
        pass
    return os.environ.get("HF_TOKEN")
