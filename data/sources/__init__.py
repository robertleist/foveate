"""Built-in dataset sources.

Importing this package registers all built-in sources with the registry.
New sources should be imported here so they are available by name.
"""

from data.sources.coco import COCOSource
from data.sources.pannuke import PanNukeSource
from data.sources.synthetic import SyntheticSource

__all__ = ["COCOSource", "PanNukeSource", "SyntheticSource"]
