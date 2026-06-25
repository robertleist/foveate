"""foveate — training-free, recursive, prototype-guided instance discovery from frozen
DINOv3 features.

A foveated zoom: starting from the whole image, the cascade recursively directs the encoder's
fixed patch budget at candidate regions (a connected-components fixed point), re-identifies the
exemplar concept (CLS / prototype similarity), and refines each instance's border — discovering
*every* instance of a prompted morphology within a single image (or across images).

Lineage: inspired by INSID3 (CVPR 2026), which produces a single cross-image mask in one
forward; foveate adds recursion + individuation to turn that into instance discovery.

Quickstart
----------
>>> from foveate import discover_instances, Config, DINOv3Backbone
>>> backbone = DINOv3Backbone()
>>> instances, stats = discover_instances(backbone, image, exemplar_masks, Config())

Stages (single-pass :func:`run` pipeline, exposed for the notebooks): features → gate →
clustering → individuation → merge. The recursive entry point is :func:`discover_instances`.
"""

from foveate.cascade import discover_instances
from foveate.config import Config, InSID3Params
from foveate.pipeline import InSID3Result, run
from foveate.types import Backbone, CascadeStats, DiscoveredInstance, Instance, Stats

__all__ = [
    "discover_instances",
    "Config",
    "Instance",
    "Stats",
    "Backbone",
    "run",
    "InSID3Result",
    # Backward-compatible aliases.
    "InSID3Params",
    "DiscoveredInstance",
    "CascadeStats",
    "DINOv3Backbone",
    "MockBackbone",
]


def __getattr__(name):
    # Lazy so `import foveate` never requires transformers (the [dino] extra).
    if name in ("DINOv3Backbone", "MockBackbone"):
        import foveate.backbones as backbones

        return getattr(backbones, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
