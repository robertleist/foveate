"""Method registry package — drop a module in this directory to add a baseline.

Every submodule is auto-imported below so its ``@register_method`` decorator runs; a new
baseline never edits this file. Imports are tolerant: a baseline whose optional heavy
dependency (e.g. SAM3 weights/library) isn't installed is skipped with a one-line warning
instead of breaking every run.
"""

from __future__ import annotations

import importlib
import pkgutil

from experiments.methods.base import (
    Method,
    MethodPrediction,
    build_backbone,
    build_method,
    register_method,
)

__all__ = [
    "Method",
    "MethodPrediction",
    "build_backbone",
    "build_method",
    "register_method",
]

for _mod in pkgutil.iter_modules(__path__):
    if _mod.name == "base" or _mod.name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{_mod.name}")
    except ImportError as exc:  # optional dep missing — skip the baseline, keep the rest
        print(f"[methods] skipping {_mod.name!r} (missing dependency): {exc}")
