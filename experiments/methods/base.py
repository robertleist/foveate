"""Method abstraction — one interface for foveate and the paper's baselines.

The runner (:mod:`experiments.run`) is method-agnostic: it builds one :class:`Method` per run
from the full experiment config and calls :meth:`Method.predict` per :class:`EvalItem`;
metrics, npz dumps, overlays and MLflow logging are shared across methods.

A method registers itself under a config name and is selected by the experiment config's
``method:`` block (absent block => ``foveate``, so existing configs keep working)::

    @register_method("sam3")
    class SAM3Method(Method):
        def __init__(self, config: dict[str, Any]):   # full experiment config dict
            super().__init__(config)                  # pull what you need (method/backbone/...)

        def predict(self, item, observer=None):
            ...

    # in the YAML:
    method:
      type: sam3

Modules in :mod:`experiments.methods` are auto-imported (see ``__init__.py``), so adding a
baseline is one new file in this package — no registry or ``__init__`` edits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from experiments.datasets import EvalItem


@dataclass
class MethodPrediction:
    """One image's predictions plus the per-image cost the runner accounts for."""

    masks: np.ndarray            # (N, H, W) bool
    scores: np.ndarray           # (N,) float
    n_embeds: int = 0            # backbone forwards spent (0 when the method doesn't track it)
    boxes: np.ndarray | None = None   # (N, 4) [x0,y0,x1,y1] detection boxes for box AP; None →
    #                                   the runner scores the tight mask box instead
    n_leaf_calls: int = 0        # LEAF Extract slot invocations (foveate's cfg.leaf_extractor).
    #                              Tracked separately from n_embeds because it is the cost that
    #                              scales with the EXPENSIVE extractor, and the whole claim of the
    #                              leaf slot is that it is O(leaves) rather than O(crops visited).


class Method(ABC):
    """A segmentation approach evaluated by the experiment runner.

    Built once per run from the *full* experiment config dict; each subclass pulls the blocks
    it needs (its own ``method:`` options, ``backbone:``, ``foveate:``, ...). ``method_config``
    is the ``method:`` block with the dispatch ``type`` key already removed.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.method_config = {
            k: v for k, v in dict(config.get("method") or {}).items() if k != "type"
        }

    @abstractmethod
    def predict(
        self, item: EvalItem, observer: Callable[[dict], None] | None = None
    ) -> MethodPrediction:
        """Segment all instances of the exemplar concept in ``item.image``.

        ``observer`` is foveate's cascade trace callback (one event dict per visited region);
        non-foveate methods simply ignore it — the runner only renders traces that were filled.
        """
        ...

    def param_blocks(self) -> dict[str, dict[str, Any]]:
        """Extra nested param blocks to log to MLflow, e.g. ``{"foveate": resolved_config}``.

        The runner already logs the raw ``backbone``/``data``/``eval``/``method`` config blocks;
        override this to additionally log *resolved* defaults (kept for run comparability).
        """
        return {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_METHODS: dict[str, type[Method]] = {}


def register_method(name: str):
    """Class decorator registering a :class:`Method` under a config ``method.type`` name."""

    def decorator(cls: type[Method]) -> type[Method]:
        _METHODS[name] = cls
        return cls

    return decorator


def build_method(config: dict[str, Any]) -> Method:
    """Instantiate the method selected by ``config["method"]["type"]`` (default ``foveate``)."""
    kind = dict(config.get("method") or {}).get("type", "foveate")
    if kind not in _METHODS:
        raise ValueError(
            f"unknown method type {kind!r}; registered methods: {sorted(_METHODS)}. "
            "Add a module in experiments/methods/ with @register_method."
        )
    return _METHODS[kind](config)


# ---------------------------------------------------------------------------
# Shared backbone construction (foveate + feature-based baselines)
# ---------------------------------------------------------------------------
def build_backbone(spec: dict[str, Any]):
    """Build a patch-feature backbone from a ``backbone:`` config block."""
    spec = dict(spec or {})
    kind = spec.pop("type", "dino")
    if kind == "mock":
        from foveate import MockBackbone

        return MockBackbone(**spec)
    if kind in ("dino", "dinov3"):
        from foveate import DINOv3Backbone

        return DINOv3Backbone(**spec)
    raise ValueError(f"unknown backbone type {kind!r}")


__all__ = [
    "EvalItem",
    "Method",
    "MethodPrediction",
    "build_backbone",
    "build_method",
    "register_method",
]
