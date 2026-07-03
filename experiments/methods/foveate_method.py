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


@register_method("foveate")
class FoveateMethod(Method):
    """Recursive prototype-guided instance discovery from frozen DINOv3 features."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.backbone = build_backbone(config.get("backbone", {}))
        self.foveate_config = Config.from_dict(config.get("foveate", {}))
        # The runner's CLS-trajectory rendering reads the acceptance floor off the method.
        self.cls_threshold = self.foveate_config.cls_threshold

    def predict(
        self, item: EvalItem, observer: Callable[[dict], None] | None = None
    ) -> MethodPrediction:
        instances, stats = discover_instances(
            self.backbone, item.image, item.exemplar_masks, config=self.foveate_config,
            exemplar_image=item.exemplar_image, observer=observer,
        )
        h, w = item.image.shape[:2]
        if instances:
            masks = np.stack([inst.mask.astype(bool) for inst in instances])
            scores = np.array([inst.score for inst in instances], dtype=np.float64)
        else:
            masks = np.zeros((0, h, w), dtype=bool)
            scores = np.zeros((0,), dtype=np.float64)
        return MethodPrediction(masks=masks, scores=scores, n_embeds=stats.n_embeds)

    def param_blocks(self) -> dict[str, dict[str, Any]]:
        # Log the *resolved* foveate Config (defaults included), matching the pre-abstraction
        # runner so old MLflow runs stay directly comparable.
        return {"foveate": self.foveate_config.to_dict()}
