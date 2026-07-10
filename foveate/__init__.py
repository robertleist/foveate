"""foveate — Foveate: a training-free, in-context, recursive method for instance segmentation
from frozen DINOv3 features (paper: *Foveate*, CV4E @ ECCV 2026).

A foveated zoom: starting from the whole image, the cascade recursively directs the encoder's
fixed patch budget at candidate regions (a connected-components fixed point), re-identifies the
exemplar concept against the exemplar bank (crop similarity of CLS tokens), and separates each
instance — discovering *every* instance of a prompted concept within a single image (or across
images).

Lineage: inspired by INSID3, which produces a single cross-image mask in one forward; Foveate
adds the recursive cascade + splitting to turn that into instance discovery.

Quickstart
----------
>>> from foveate import foveate_cascade, Config, DINOv3Backbone
>>> backbone = DINOv3Backbone()
>>> instances, stats = foveate_cascade(backbone, image, exemplar_masks, Config())

The recursive entry point is :func:`foveate_cascade` (the Foveate cascade of Algorithm 1). A
single-pass :func:`run` pipeline (features → gate → clustering → individuation → merge) is kept
for the notebooks / ablations.
"""

from foveate.cascade import foveate_cascade
from foveate.config import Config, InSID3Params
from foveate.pipeline import InSID3Result, run
from foveate.types import Backbone, CascadeStats, DiscoveredInstance, Instance, Stats

# Pre-rename public name (the cascade entry point used to be ``discover_instances``).
discover_instances = foveate_cascade

__all__ = [
    "foveate_cascade",
    "Config",
    "Instance",
    "Stats",
    "Backbone",
    "run",
    "InSID3Result",
    # Backward-compatible aliases.
    "discover_instances",
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
