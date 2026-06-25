"""Dataset registry — refer to datasets by name in configs.

Register a source once::

    @register_dataset("coco")
    class COCOSource(DatasetSource):
        ...

and build it from a :class:`~data.config.DataConfig`::

    source = build_source(cfg)

Built-in sources are imported (and thus registered) in ``data/sources/__init__.py``.
"""

from __future__ import annotations

from typing import Dict, List, Type

from data.source import DatasetSource

_REGISTRY: Dict[str, Type[DatasetSource]] = {}


def register_dataset(name: str):
    """Class decorator that registers a :class:`DatasetSource` under ``name``."""

    def decorator(cls: Type[DatasetSource]) -> Type[DatasetSource]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def list_datasets() -> List[str]:
    """Names of all registered datasets, sorted."""
    return sorted(_REGISTRY)


def get_source_class(name: str) -> Type[DatasetSource]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. Registered datasets: {list_datasets()}. "
            "Register a new one with @register_dataset(name)."
        )
    return _REGISTRY[name]


def build_source(cfg) -> DatasetSource:
    """Instantiate the registered source for ``cfg.name`` from a DataConfig."""
    return get_source_class(cfg.name).from_config(cfg)
