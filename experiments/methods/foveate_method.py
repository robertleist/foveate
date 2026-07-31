"""FoveateMethod — the default method: recursive foveated instance discovery.

Wraps :func:`foveate.cascade` behind the :class:`~experiments.methods.base.Method`
interface exactly as the pre-abstraction runner did: the ``backbone:`` block builds the
encoder, the ``foveate:`` block resolves into one :class:`foveate.Config`, and the observer
callback is threaded through so the runner's cascade-trace rendering keeps working.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from experiments import eval as evallib
from experiments.datasets import EvalItem
from experiments.methods.base import (
    Method,
    MethodPrediction,
    build_backbone,
    register_method,
)
from foveate import Config, cascade
from foveate.extract import build_extractor, resolve_extractor_name


@register_method("foveate")
class FoveateMethod(Method):
    """Recursive prototype-guided instance discovery from frozen DINOv3 features."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.backbone = build_backbone(config.get("backbone", {}))
        self.foveate_config = Config.from_dict(config.get("foveate", {}))
        # The runner's CLS-trajectory rendering reads the acceptance floor off the method.
        self.crop_sim_floor = self.foveate_config.crop_sim_floor
        # Cross-image (inter) reuses one exemplar bank for every target of a class; cache the
        # Extract slot (whose set_reference embeds every exemplar crop) so that cost is paid once.
        self._ref_cache: dict = {}

    def _cached_extractor(self, item: EvalItem):
        """Return a reference-set Extract slot for ``item``, or ``None`` to build it per-image.

        Only the cross-image (inter) protocol reuses a bank across target images — its exemplars
        live on a shared support image. Intra exemplars live on the (per-image) target itself, so
        there is nothing to cache and we let the cascade build the reference as before.
        """
        if item.exemplar_image is None:
            return None                                 # intra: exemplars are image-specific
        key = (id(item.exemplar_image), item.class_id)
        ext = self._ref_cache.get(key)
        if ext is None:
            ext = build_extractor(self.foveate_config)
            ext.set_reference(self.backbone, item.exemplar_image, item.exemplar_masks,
                              None, self.foveate_config)
            self._ref_cache[key] = ext
        return ext

    def predict(
        self, item: EvalItem, observer: Callable[[dict], None] | None = None
    ) -> MethodPrediction:
        # Oracle ablations: feed the cascade the target's GT class foreground as an instance-label
        # map (0 = bg, i = the i-th GT instance). The oracle *Extract* slot consumes it (whether as
        # the monolithic GT extractor or as a composite whose Where half is oracular), and so does
        # the oracle *Stop* rule (box<->instance isolation in place of g). For every other slot
        # combination it stays None so the discovery is honest.
        cfg = self.foveate_config
        oracle = (resolve_extractor_name(cfg) == "oracle"
                  or cfg.foreground_extractor in ("oracle", "oracle_cc")
                  or cfg.stop_rule == "oracle"
                  or cfg.merge_rule == "oracle"
                  or cfg.mask_upsample == "oracle")
        gt_foreground = None
        if oracle and len(item.gt_masks):
            gt_foreground = np.zeros(item.image.shape[:2], dtype=np.int32)
            for i, m in enumerate(item.gt_masks, start=1):
                gt_foreground[m.astype(bool)] = i          # later instances win on overlap
        instances, stats = cascade(
            self.backbone, item.image, item.exemplar_masks, config=self.foveate_config,
            exemplar_image=item.exemplar_image, extractor=self._cached_extractor(item),
            gt_foreground=gt_foreground, observer=observer,
        )
        h, w = item.image.shape[:2]
        if instances:
            masks = np.stack([inst.mask.astype(bool) for inst in instances])
            scores = np.array([inst.score for inst in instances], dtype=np.float64)
            # Detection box for box AP. ``"mask"`` (default) = the tight box of the emitted mask,
            # which is what every detection benchmark compares against — COCO-FSOD, RF20-VL and
            # CD-FSOD are all box AP, and the GT side is always a tight box. ``"crop"`` reports the
            # final crop the cascade converged on instead: useful as a *diagnostic* of how well the
            # recursion frames an object, but it is padded by ``pad_frac``/``crop_dilate``, so
            # scoring it as a detection penalises us for our own padding.
            if str(self.foveate_config.report_boxes) == "crop":
                boxes = np.array(
                    [(inst.box[2], inst.box[0], inst.box[3], inst.box[1]) for inst in instances],
                    dtype=np.float64,
                )
            else:
                boxes = evallib.masks_to_boxes(masks)
        else:
            masks = np.zeros((0, h, w), dtype=bool)
            scores = np.zeros((0,), dtype=np.float64)
            boxes = np.zeros((0, 4), dtype=np.float64)
        return MethodPrediction(masks=masks, scores=scores, n_embeds=stats.n_embeds, boxes=boxes)

    def param_blocks(self) -> dict[str, dict[str, Any]]:
        # Log the *resolved* foveate Config (defaults included), matching the pre-abstraction
        # runner so old MLflow runs stay directly comparable.
        return {"foveate": self.foveate_config.to_dict()}
