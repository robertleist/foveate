"""Ablation sweeps — run :func:`experiments.run.run_experiment` over a grid, one MLflow run each.

A sweep file is a base experiment config plus a ``sweep:`` block mapping dotted config keys to
lists of values; the cartesian product is expanded and each combination runs as its own MLflow
run (so they group under one experiment and compare directly). Example sweep block::

    sweep:
      foveate.gate_threshold_mode: [static, otsu, gmm2]
      foveate.prototype_reduction: [all, kmeans]

    python -m experiments.ablations --config configs/ablation_thresholds.yaml
"""

from __future__ import annotations

import argparse
import copy
import itertools
from typing import Any

import yaml

from experiments.run import run_experiment


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    node = config
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def expand_sweep(base: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one config per point in the cartesian product of the ``sweep:`` block."""
    base = copy.deepcopy(base)          # never mutate the caller's dict
    sweep = base.pop("sweep", {}) or {}
    if not sweep:
        return [base]
    keys = list(sweep)
    grids = [sweep[k] if isinstance(sweep[k], list) else [sweep[k]] for k in keys]
    base_name = base.get("run_name", "ablation")
    configs = []
    for combo in itertools.product(*grids):
        cfg = copy.deepcopy(base)
        suffix = []
        for k, v in zip(keys, combo):
            _set_dotted(cfg, k, v)
            suffix.append(f"{k.split('.')[-1]}={v}")
        tag = "_".join(suffix)
        cfg["run_name"] = f"{base_name}__{tag}"
        configs.append(cfg)
    return configs


def run_sweep(base: dict[str, Any]) -> list[dict[str, Any]]:
    configs = expand_sweep(base)
    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n=== [{i}/{len(configs)}] {cfg['run_name']} ===")
        results.append({"run_name": cfg["run_name"], **run_experiment(cfg)})
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a foveate ablation sweep.")
    parser.add_argument("--config", required=True, help="Sweep config (base + sweep: block).")
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    run_sweep(base)


if __name__ == "__main__":
    main()
