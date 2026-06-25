"""Mask-based instance-segmentation datasets for foveate experiments.

A :class:`DatasetSource` describes one dataset format (COCO, PanNuke, ...); the shared
:class:`InstanceDataset` adds seeded image selection and a positive-unlabelled (PU) instance
split on top. Each :class:`InstanceSample` carries the image, per-instance binary masks, a
per-class semantic mask stack, and the PU partition (``train`` instances double as exemplar
prompts; ``val`` / ``unlabelled`` are held out for evaluation).
"""

from data.sample import GTInstance, InstanceSample
from data.config import DataConfig
from data.source import (
    AnnotationMeta,
    DatasetSource,
    ImageMeta,
    InstanceMask,
)
from data.registry import (
    build_source,
    list_datasets,
    register_dataset,
)

# Importing the sources package registers all built-in datasets by name.
from data import sources  # noqa: F401
from data.sources import COCOSource, PanNukeSource

from data.dataset import InstanceDataset

__all__ = [
    "InstanceSample",
    "GTInstance",
    "DataConfig",
    "DatasetSource",
    "ImageMeta",
    "AnnotationMeta",
    "InstanceMask",
    "InstanceDataset",
    "COCOSource",
    "PanNukeSource",
    "register_dataset",
    "build_source",
    "list_datasets",
]
