"""FoveateMethod — the default method: recursive foveated instance discovery.

Wraps :func:`foveate.discover_instances` behind the :class:`~experiments.methods.base.Method`
interface exactly as the pre-abstraction runner did: the ``backbone:`` block builds the
encoder, the ``foveate:`` block resolves into one :class:`foveate.Config`, and the observer
callback is threaded through so the runner's cascade-trace rendering keeps working.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from experiments.datasets import EvalItem
from experiments.methods.base import (
    Method,
    MethodPrediction,
    build_backbone,
    register_method,
)
from foveate import Config, discover_instances
from foveate.foreground import build_extractor


@register_method("foveate")
class FoveateMethod(Method):
    """Recursive prototype-guided instance discovery from frozen DINOv3 features."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.backbone = build_backbone(config.get("backbone", {}))
        self.foveate_config = Config.from_dict(config.get("foveate", {}))
        # The runner's CLS-trajectory rendering reads the acceptance floor off the method.
        self.cls_threshold = self.foveate_config.cls_threshold
        # Cross-image (inter) reuses one exemplar bank for every target of a class; cache the
        # extractor (whose set_reference embeds every exemplar crop) so that cost is paid once.
        self._ref_cache: dict = {}

    def _cached_extractor(self, item: EvalItem):
        """Return a reference-set extractor for ``item``, or ``None`` to build it per-image.

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
        instances, stats = discover_instances(
            self.backbone, item.image, item.exemplar_masks, config=self.foveate_config,
            exemplar_image=item.exemplar_image, extractor=self._cached_extractor(item),
            observer=observer,
        )
        h, w = item.image.shape[:2]
        if instances:
            masks = np.stack([inst.mask.astype(bool) for inst in instances])
            scores = np.array([inst.score for inst in instances], dtype=np.float64)
            # Detection box = the final crop insid3 converged on (Instance.box is (y0,y1,x0,x1)),
            # reordered to the eval's [x0,y0,x1,y1]. This is scored for box AP instead of the
            # tight mask box.
            boxes = np.array(
                [(inst.box[2], inst.box[0], inst.box[3], inst.box[1]) for inst in instances],
                dtype=np.float64,
            )
        else:
            masks = np.zeros((0, h, w), dtype=bool)
            scores = np.zeros((0,), dtype=np.float64)
            boxes = np.zeros((0, 4), dtype=np.float64)
        return MethodPrediction(masks=masks, scores=scores, n_embeds=stats.n_embeds, boxes=boxes)

    def param_blocks(self) -> dict[str, dict[str, Any]]:
        # Log the *resolved* foveate Config (defaults included), matching the pre-abstraction
        # runner so old MLflow runs stay directly comparable.
        return {"foveate": self.foveate_config.to_dict()}
