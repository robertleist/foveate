"""Config-driven experiment runner: discover instances over a dataset, score, log to MLflow.

    python -m experiments.run --config configs/coco_baseline.yaml
    python -m experiments.run --config configs/coco_baseline.yaml --set foveate.gate_threshold=0.6

A run: build the method (default: foveate) + dataset from the YAML, call
:meth:`~experiments.methods.base.Method.predict` for every image, compute the metrics in
:mod:`experiments.eval`, and log params + metrics + per-image cost (embeds, runtime) and the
saved masks to MLflow. Use the ``mock`` backbone (no weights) for smoke tests.

Baselines plug in via the ``method:`` config block (see :mod:`experiments.methods`); configs
without one run the foveate pipeline exactly as before.
"""

from __future__ import annotations

import argparse
import json
import tempfile
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
from experiments.methods import Method, build_method
from experiments.methods.base import build_backbone  # noqa: F401  (backward-compat re-export)


# ---------------------------------------------------------------------------
# Progress / logging
# ---------------------------------------------------------------------------
def _progress(iterable, desc: str, total: int | None):
    """Wrap ``iterable`` in a tqdm bar (per-image progress + cost postfix).

    tqdm is an ``exp`` extra, not a core dependency, so fall back to the bare iterable
    (with a one-line start log) when it's not installed.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        print(f"[run] {desc}: processing{f' {total}' if total else ''} images...")
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit="img", dynamic_ncols=True, leave=True)


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
    if config.get("sweep"):
        print("[run] warning: this config has a 'sweep:' block but run_experiment runs a single "
              "config — use experiments.ablations.run_sweep (or the run.py CLI) to expand it.")
    method = build_method(config)
    data_cfg = DataConfig.from_dict(config["data"])
    eval_cfg = config.get("eval", {}) or {}
    max_exemplars = eval_cfg.get("max_exemplars", 3)
    limit = eval_cfg.get("limit")
    targets = eval_cfg.get("targets", ["intra"])
    # Per-image debug overlays (prompt | GT | prediction). int caps how many to render,
    # True/"all" renders every image, False/0/None disables. Logged to MLflow as artifacts.
    visualize = eval_cfg.get("visualize", False)
    # Per-image cascade trace (contact sheet of the recursive zoom). Same budget semantics.
    cascade_trace = eval_cfg.get("cascade_trace", False)

    intra_ds, inter_ds = build_datasets(data_cfg)

    # Stage artifacts in a fresh per-run temp dir so MLflow is the single source of truth.
    # A persistent output_dir was reused across runs without being cleared, so log_artifacts
    # uploaded stale masks/overlays from earlier datasets; a temp dir is empty every time.
    backbone_kind = (config.get("backbone") or {}).get("type", "dino")
    method_kind = (config.get("method") or {}).get("type", "foveate")
    result: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="foveate-run-") as tmp:
        output_dir = Path(tmp)

        print(f"[run] '{config.get('run_name', 'run')}': method={method_kind} "
              f"backbone={backbone_kind} data={data_cfg.name} targets={targets} "
              f"intra_pool={len(intra_ds)} inter_pool={len(inter_ds) if inter_ds else 0} "
              f"-> {output_dir}")

        if "intra" in targets:
            items = iter_intra_items(intra_ds, max_exemplars=max_exemplars, limit=limit)
            result.update(_evaluate(method, items, output_dir / "intra", "intra",
                                    visualize=visualize, cascade_trace=cascade_trace, total=limit))
        if "inter" in targets:
            if inter_ds is None:
                print("[run] inter eval requested but interval_images == 0; skipping.")
            else:
                support = build_support_index(intra_ds)
                items = iter_inter_items(inter_ds, support, max_exemplars=max_exemplars, limit=limit)
                result.update(_evaluate(method, items, output_dir / "inter", "inter",
                                        visualize=visualize, cascade_trace=cascade_trace, total=limit))

        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))

        with _mlflow_run(config.get("mlflow", {}), config.get("run_name")) as mlflow:
            if mlflow is not None:
                params = {}
                params.update(_flatten_params("backbone", config.get("backbone", {})))
                params.update(_flatten_params("data", config["data"]))
                # Resolved per-method blocks (e.g. the full foveate Config, defaults included)
                # keep new runs comparable with pre-abstraction MLflow runs.
                for block_name, block in method.param_blocks().items():
                    params.update(_flatten_params(block_name, block))
                params.update(_flatten_params("method", config.get("method", {}) or {}))
                params.update(_flatten_params("eval", eval_cfg))
                mlflow.log_params(params)
                mlflow.log_metrics({k: float(v) for k, v in result.items()
                                    if isinstance(v, (int, float)) and not np.isnan(float(v))})
                mlflow.log_artifacts(str(output_dir))

    return result


def _evaluate(method: Method, items, output_dir: Path, prefix: str,
              visualize: Any = False, cascade_trace: Any = False,
              total: int | None = None) -> dict[str, Any]:
    """Predict over ``items``, save per-image masks, and return ``{prefix}_<metric>`` results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[evallib.ImagePrediction] = []
    gts: list[np.ndarray] = []
    recoveries: list[float] = []
    total_embeds, total_time, n_items = 0, 0.0, 0
    total_pred = 0

    def _budget(flag: Any) -> float:
        return float("inf") if flag in (True, "all") else int(flag or 0)

    viz_budget = _budget(visualize)
    trace_budget = _budget(cascade_trace)
    viz_dir = output_dir / "viz"
    trace_dir = output_dir / "viz" / "cascade"
    traj_dir = output_dir / "viz" / "trajectory"

    bar = _progress(items, desc=prefix, total=total)
    for item in bar:
        trace: list | None = [] if n_items < trace_budget else None
        pred, n_embeds, elapsed = _discover_one(method, item, trace=trace)
        predictions.append(pred)
        gts.append(item.gt_masks)
        if item.exemplar_image is None:           # recovery only meaningful for same-image prompts
            recoveries.append(evallib.exemplar_recovery_iou(pred, item.exemplar_masks))
        total_embeds += n_embeds
        total_time += elapsed
        total_pred += pred.masks.shape[0]
        n_items += 1
        np.savez_compressed(
            output_dir / f"{item.image_id}.npz",
            masks=pred.masks, scores=pred.scores, gt=item.gt_masks, class_id=item.class_id,
        )
        if n_items <= viz_budget:
            from experiments.visualize import save_item_overlay
            save_item_overlay(item, pred, viz_dir)
        if trace:   # non-empty only when the method drove the observer (i.e. foveate)
            from experiments.visualize import save_cascade_trace
            from experiments.trajectories import save_cls_trajectory
            save_cascade_trace(item, trace, trace_dir)
            save_cls_trajectory(item, trace, traj_dir,
                                cls_threshold=getattr(method, "cls_threshold", 0.5))
        # Live cost readout on the bar: predictions/image, embeds/image, sec/image.
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(pred=f"{total_pred / n_items:.1f}",
                            embeds=f"{total_embeds / n_items:.0f}",
                            s_img=f"{total_time / n_items:.2f}", refresh=False)

    if n_items == 0:
        print(f"[run] {prefix}: no eligible images (no prompt/target pair); skipping.")
        return {}

    metrics = evallib.evaluate(predictions, gts)
    out = {f"{prefix}_{k}": v for k, v in metrics.to_dict().items()}
    out[f"{prefix}_exemplar_recovery_iou"] = (
        float(np.nanmean(recoveries)) if recoveries else float("nan")
    )
    out[f"{prefix}_mean_embeds"] = total_embeds / max(n_items, 1)
    out[f"{prefix}_mean_runtime_s"] = total_time / max(n_items, 1)
    out[f"{prefix}_n_images"] = n_items
    print(f"[run] {prefix}: {n_items} imgs | AP={metrics.ap:.3f} AP50={metrics.ap50:.3f} "
          f"PQ={metrics.pq:.3f} mIoU={metrics.mean_iou:.3f} "
          f"count_err={metrics.count_error:.2f} | {total_embeds / max(n_items, 1):.0f} embeds/img")
    return out


def _discover_one(method: Method, item: EvalItem, trace: list | None = None):
    # When `trace` is a list, collect a lightweight observer event per region (drop the heavy
    # `feat` tensor; keep the grids/boxes the cascade visualization needs). Methods that don't
    # cascade simply never call the observer, so the trace stays empty and nothing is rendered.
    observer = None
    if trace is not None:
        def observer(info: dict) -> None:
            trace.append({k: v for k, v in info.items() if k != "feat"})
    t0 = time.perf_counter()
    pred = method.predict(item, observer=observer)
    elapsed = time.perf_counter() - t0
    return evallib.ImagePrediction(masks=pred.masks, scores=pred.scores), pred.n_embeds, elapsed


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

    # A `sweep:` block expands to a grid of runs. Dispatch here so the single runner does the
    # right thing too — otherwise the block is silently ignored and only the base config runs.
    if config.get("sweep"):
        from experiments.ablations import run_sweep

        n = len(config["sweep"])
        print(f"[run] '{args.config}' has a sweep block ({n} swept key(s)); expanding the grid.")
        run_sweep(config)
        return

    run_experiment(config)


if __name__ == "__main__":
    main()
