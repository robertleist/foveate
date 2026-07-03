"""SemanticCCMethod — the naive single-pass "semantic + connected components" baseline.

Purpose in the paper: isolate the value of foveate's recursion + individuation. This baseline
does the "WHERE is the class" step **once** — a single INSID3 foreground pass over the whole
target image, no cascade recursion, no zoom — then splits that one foreground into instances
with plain connected-components labelling instead of foveate's marker-controlled watershed +
CLS-guided splitting. If foveate's individuation adds nothing over connected components, this
baseline would match it, so it must be a faithful single-pass version of the same WHERE step.

WHERE step (shared with the main method for a fair ablation):
    We reuse foveate's INSID3 foreground extractor directly — the exact three lines that both
    :func:`foveate.pipeline.run` and the root of :func:`foveate.cascade.discover_instances` use to
    get the class region on a grid, minus everything after it:

        feats = foveate.features.embed_image(backbone, image, standardize=cfg.standardize)
        extractor = foveate.foreground.build_extractor(cfg)   # cfg.foreground_extractor == "insid3"
        extractor.set_reference(backbone, ref_image, exemplar_masks, None, cfg)
        gate = extractor.predict(feats)                       # GateResult: foreground + score_map

    ``gate.foreground`` is the ``(Hp, Wp)`` boolean class region *before* any individuation, and
    ``gate.score_map`` is the ``(Hp, Wp)`` soft per-patch class confidence in ``[0, 1]``. We take
    those and stop — we do NOT call :func:`foveate.pipeline.run` (it runs clustering /
    individuation / merge / watershed after the gate) and we do NOT call ``discover_instances``
    (recursive zoom). Calling the extractor's ``predict`` once, on the full image, is the cleanest
    way to obtain a single-pass class foreground with no recursion.

Cross-image (inter) is wired through :meth:`InSID3Extractor.set_reference`, whose ``ref_image``
argument is the image the exemplar masks live on: for intra ``exemplar_image is None`` so the
reference is the target image itself; for inter we pass ``item.exemplar_image`` so the reference
is built from the support image while the foreground is predicted on the (disjoint) target grid.

WHAT step (this baseline's whole contribution): binarize the foreground, run
:func:`scipy.ndimage.label` connected-components on it (each component = one predicted instance),
drop components below ``min_area`` pixels, and score each by the mean foreground confidence under
it. This deliberately has no recursion and no individuation — that is the point of the ablation.
"""

from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np
from scipy.ndimage import generate_binary_structure, label

from experiments.datasets import EvalItem
from experiments.methods.base import (
    Method,
    MethodPrediction,
    build_backbone,
    register_method,
)
from foveate import Config
from foveate import features as featlib
from foveate.foreground import build_extractor


def foreground_to_instances(
    foreground: np.ndarray,
    score_map: np.ndarray | None,
    *,
    min_area: int,
    connectivity: int,
    score_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a boolean foreground into instance masks by connected components.

    ``foreground`` is ``(H, W)`` bool (already binarized and at the resolution the returned masks
    should have). ``score_map`` is an optional ``(H, W)`` float confidence in the same resolution,
    used only when ``score_mode == "mean_prob"``. Returns ``(masks (N, H, W) bool, scores (N,))``;
    ``N`` may be 0. This is the deterministic core of the baseline and is unit-tested directly.
    """
    h, w = foreground.shape
    if not foreground.any():
        return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.float64)

    # 4- or 8-connectivity for the labelling — 8 is the default (matches the cascade's _CONN8).
    structure = generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labels, n = label(foreground, structure=structure)

    masks: list[np.ndarray] = []
    scores: list[float] = []
    for cid in range(1, n + 1):
        comp = labels == cid
        if int(comp.sum()) < min_area:
            continue
        masks.append(comp)
        if score_mode == "mean_prob" and score_map is not None:
            scores.append(float(score_map[comp].mean()))
        else:
            scores.append(1.0)

    if not masks:
        return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.float64)
    return np.stack(masks), np.asarray(scores, dtype=np.float64)


@register_method("semantic_cc")
class SemanticCCMethod(Method):
    """Single-pass INSID3 foreground + connected-components instances (no recursion)."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.backbone = build_backbone(config.get("backbone", {}))
        # Share the WHERE params with the main method so the ablation is fair (same INSID3 gate).
        self.foveate_config = Config.from_dict(config.get("foveate", {}))

        self.min_area = int(self.method_config.get("min_area", 10))
        self.threshold = float(self.method_config.get("threshold", 0.5))
        self.connectivity = int(self.method_config.get("connectivity", 8))
        if self.connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {self.connectivity}")
        self.score_mode = str(self.method_config.get("score_mode", "mean_prob"))
        # Cross-image (inter) reuses one exemplar bank for every target of a class; cache the
        # reference-set extractor so set_reference (which embeds every exemplar crop) runs once
        # per class instead of once per target image.
        self._ref_cache: dict = {}

    def _reference_extractor(self, item: EvalItem):
        """Build (or fetch from cache) the INSID3 extractor with its reference set for ``item``.

        Intra exemplars live on the per-image target, so there is nothing to reuse — a fresh
        extractor is built each call. Inter exemplars live on a shared support image, so the
        extractor is cached by ``(support image, class)`` and reused across all targets.
        """
        cfg = self.foveate_config
        # ref_image is where the exemplar masks live: the target itself (intra) or the support.
        ref_image = item.image if item.exemplar_image is None else item.exemplar_image
        if item.exemplar_image is None:                 # intra: per-image exemplars, no caching
            extractor = build_extractor(cfg)
            extractor.set_reference(self.backbone, ref_image, item.exemplar_masks, None, cfg)
            return extractor
        key = (id(item.exemplar_image), item.class_id)
        extractor = self._ref_cache.get(key)
        if extractor is None:
            extractor = build_extractor(cfg)
            extractor.set_reference(self.backbone, ref_image, item.exemplar_masks, None, cfg)
            self._ref_cache[key] = extractor
        return extractor

    def predict(
        self, item: EvalItem, observer: Callable[[dict], None] | None = None
    ) -> MethodPrediction:
        cfg = self.foveate_config
        h, w = item.image.shape[:2]

        # --- single-pass WHERE: INSID3 foreground on the FULL target image, no recursion ---
        feats = featlib.embed_image(self.backbone, item.image, standardize=cfg.standardize)
        # The reference (exemplar bank) is built once and, for inter, reused across targets.
        extractor = self._reference_extractor(item)
        n_embeds = 1                                   # one target forward; reference embeds are
        # extractor-internal and not tracked here, so this is a lower bound like the pipeline's.
        gate = extractor.predict(feats)                # single-pass; no cls => pooled reference

        # gate.foreground is already a boolean class mask; gate.score_map is the soft confidence.
        # score_map is min-max normalized to [0, 1], so no separate threshold is applied to it —
        # `threshold` only matters if a future soft-map extractor returns an unbinarized map.
        fg_grid = np.asarray(gate.foreground, dtype=bool)
        score_grid = np.asarray(gate.score_map, dtype=np.float64)

        # Upsample the patch-level grid to full image resolution (nearest, like the pipeline).
        fg_full = cv2.resize(
            fg_grid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        score_full = cv2.resize(
            score_grid.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(np.float64)

        masks, scores = foreground_to_instances(
            fg_full, score_full, min_area=self.min_area,
            connectivity=self.connectivity, score_mode=self.score_mode,
        )
        return MethodPrediction(masks=masks, scores=scores, n_embeds=n_embeds)

    def param_blocks(self) -> dict[str, dict[str, Any]]:
        # Log this baseline's resolved params plus the shared foveate WHERE config, so a
        # semantic_cc run is directly comparable to the foveate run beside it in MLflow.
        return {
            "semantic_cc": {
                "min_area": self.min_area,
                "threshold": self.threshold,
                "connectivity": self.connectivity,
                "score_mode": self.score_mode,
            },
            "foveate": self.foveate_config.to_dict(),
        }
