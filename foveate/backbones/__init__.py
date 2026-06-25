"""Backbone implementations.

``DINOv3Backbone`` is the default (requires the ``[dino]`` extra: ``transformers`` +
``huggingface-hub``). ``MockBackbone`` needs only numpy/torch and produces colour-based
features — enough to exercise the whole pipeline in tests and demos without weights.
"""

from foveate.backbones.mock import MockBackbone

__all__ = ["MockBackbone", "DINOv3Backbone"]


def __getattr__(name):
    # Lazy import so importing foveate.backbones doesn't pull in transformers.
    if name == "DINOv3Backbone":
        from foveate.backbones.dinov3 import DINOv3Backbone

        return DINOv3Backbone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
