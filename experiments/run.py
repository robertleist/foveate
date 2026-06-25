"""Config-driven experiment runner: discover instances over a dataset, score, log to MLflow.

    python -m experiments.run --config configs/coco_baseline.yaml
    python -m experiments.run --config configs/coco_baseline.yaml --set foveate.gate_threshold=0.6

A run: build the backbone + dataset from the YAML, thread one :class:`~foveate.config.Config`
through :func:`foveate.discover_instances` for every image, compute the metrics in
:mod:`experiments.eval`, and log params + metrics + per-image cost (embeds, runtime) and the
saved masks to MLflow. Use the ``mock`` backbone (no weights) for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from data import DataConfig
from experiments import eval as evallib
from experiments.datasets import (
    EvalItem,
    build_datasets,
    build_support_index,
    iter_inter_items,
    iter_intra_items,
)
from foveate import Config, discover_instances


# ---------------------------------------------------------------------------
# Backbone construction
# ---------------------------------------------------------------------------
def build_backbone(spec: dict[str, Any]):
    spec = dict(spec or {})
    kind = spec.pop("type", "dino")
    if kind == "mock":
        from foveate import MockBackbone

        return MockBackbone(**spec)
    if kind in ("dino", "dinov3"):
        from foveate import DINOv3Backbone

        return DINOv3Backbone(**spec)
    raise ValueError(f"unknown backbone type {kind!r}")


# ---------------------------------------------------------------------------
# MLflow (optional)
# ---------------------------------------------------------------------------
def _load_dotenv(dotenv_path: str | None = None) -> None:
    """Load a .env file if python-dotenv is available; silently skip otherwise."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    from pathlib import Path as _Path
    path = _Path(dotenv_path) if dotenv_path else _Path(".env")
    if path.exists():
        load_dotenv(path, override=False)  # env already set takes precedence


@contextmanager
def _mlflow_run(mlflow_cfg: dict[str, Any], run_name: str | None):
    cfg = dict(mlflow_cfg or {})
    if not cfg.get("enabled", True):
        yield None
        return
    try:
        import mlflow
    except ImportError:
        print("[run] mlflow not installed; skipping tracking.")
        yield None
        return

    _load_dotenv(cfg.get("dotenv"))

    # YAML tracking_uri takes precedence over MLFLOW_TRACKING_URI env var
    tracking_uri = cfg.get("tracking_uri") or None
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # YAML username/password take precedence over env vars
    import os
    if cfg.get("username"):
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", cfg["username"])
    if cfg.get("password"):
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", cfg["password"])

    mlflow.set_experiment(cfg.get("experiment_name", "foveate"))
    with mlflow.start_run(run_name=run_name):
        yield mlflow


def _flatten_params(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_params(key, v))
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(map(str, v))
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run one experiment from a parsed config dict; return the metrics dict.

    Evaluates the ``intra`` (same-image) and/or ``inter`` (cross-image) protocols selected by
    ``eval.targets`` and returns their metrics with ``intra_`` / ``inter_`` prefixes.
    """
    backbone = build_backbone(config.get("backbone", {}))
    data_cfg = DataConfig.from_dict(config["data"])
    foveate_cfg = Config.from_dict(config.get("foveate", {}))
    eval_cfg = config.get("eval", {}) or {}
    max_exemplars = eval_cfg.get("max_exemplars", 3)
    limit = eval_cfg.get("limit")
    targets = eval_cfg.get("targets", ["intra"])

    intra_ds, inter_ds = build_datasets(data_cfg)

    output_dir = Path(config.get("output_dir", "runs/latest"))
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {}
    if "intra" in targets:
        items = iter_intra_items(intra_ds, max_exemplars=max_exemplars, limit=limit)
        result.update(_evaluate(backbone, items, foveate_cfg, output_dir / "intra", "intra"))
    if "inter" in targets:
        if inter_ds is None:
            print("[run] inter eval requested but interval_images == 0; skipping.")
        else:
            support = build_support_index(intra_ds)
            items = iter_inter_items(inter_ds, support, max_exemplars=max_exemplars, limit=limit)
            result.update(_evaluate(backbone, items, foveate_cfg, output_dir / "inter", "inter"))

    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    with _mlflow_run(config.get("mlflow", {}), config.get("run_name")) as mlflow:
        if mlflow is not None:
            params = {}
            params.update(_flatten_params("backbone", config.get("backbone", {})))
            params.update(_flatten_params("data", config["data"]))
            params.update(_flatten_params("foveate", foveate_cfg.to_dict()))
            params.update(_flatten_params("eval", eval_cfg))
            mlflow.log_params(params)
            mlflow.log_metrics({k: float(v) for k, v in result.items()
                                if isinstance(v, (int, float)) and not np.isnan(float(v))})
            mlflow.log_artifacts(str(output_dir))

    return result


def _evaluate(backbone, items, cfg: Config, output_dir: Path, prefix: str) -> dict[str, Any]:
    """Discover over ``items``, save per-image masks, and return ``{prefix}_<metric>`` results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[evallib.ImagePrediction] = []
    gts: list[np.ndarray] = []
    recoveries: list[float] = []
    total_embeds, total_time, n_items = 0, 0.0, 0

    for item in items:
        pred, stats, elapsed = _discover_one(backbone, item, cfg)
        predictions.append(pred)
        gts.append(item.gt_masks)
        if item.exemplar_image is None:           # recovery only meaningful for same-image prompts
            recoveries.append(evallib.exemplar_recovery_iou(pred, item.exemplar_masks))
        total_embeds += stats.n_embeds
        total_time += elapsed
        n_items += 1
        np.savez_compressed(
            output_dir / f"{item.image_id}.npz",
            masks=pred.masks, scores=pred.scores, gt=item.gt_masks, class_id=item.class_id,
        )

    metrics = evallib.evaluate(predictions, gts)
    out = {f"{prefix}_{k}": v for k, v in metrics.to_dict().items()}
    out[f"{prefix}_exemplar_recovery_iou"] = (
        float(np.nanmean(recoveries)) if recoveries else float("nan")
    )
    out[f"{prefix}_mean_embeds"] = total_embeds / max(n_items, 1)
    out[f"{prefix}_mean_runtime_s"] = total_time / max(n_items, 1)
    out[f"{prefix}_n_images"] = n_items
    return out


def _discover_one(backbone, item: EvalItem, cfg: Config):
    t0 = time.perf_counter()
    instances, stats = discover_instances(
        backbone, item.image, item.exemplar_masks, config=cfg,
        exemplar_image=item.exemplar_image,
    )
    elapsed = time.perf_counter() - t0
    h, w = item.image.shape[:2]
    if instances:
        masks = np.stack([inst.mask.astype(bool) for inst in instances])
        scores = np.array([inst.score for inst in instances], dtype=np.float64)
    else:
        masks = np.zeros((0, h, w), dtype=bool)
        scores = np.zeros((0,), dtype=np.float64)
    return evallib.ImagePrediction(masks=masks, scores=scores), stats, elapsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for ov in overrides:
        key, _, raw = ov.partition("=")
        node = config
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)   # parse ints/floats/bools/null
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a foveate experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="Override a config value, e.g. --set foveate.gate_threshold=0.6")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = _apply_overrides(config, args.overrides)
    run_experiment(config)


if __name__ == "__main__":
    main()
