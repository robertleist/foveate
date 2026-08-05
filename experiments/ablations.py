"""Ablation sweeps — run :func:`experiments.run.run_experiment` over a grid, one MLflow run each.

Two ways to say what to run, and they answer different questions.

``sweep:`` — **a grid.** Dotted config keys to lists of values; the cartesian product is expanded
and each combination runs as its own MLflow run (so they group under one experiment and compare
directly). This is the right shape for tuning: how does AP move across a knob's range?

::

    sweep:
      foveate.gate_threshold_mode: [static, otsu, gmm2]
      foveate.prototype_reduction: [all, kmeans]

``arms:`` — **named configurations.** A list of ``{name, set}`` entries, each a dict of dotted
overrides applied to the base. This is the right shape for *attribution*, where the grid is the
wrong object: the headroom table sets every slot to its oracle and then knocks out **one knob at a
time**, so the interesting configs are 1 + K, not 2^K, and each has a name that means something.

::

    arms:
      - name: all-oracle                       # the ceiling: no override at all
      - name: naive-where
        set: {foveate.foreground_extractor: otsu}

Both may appear: every arm is crossed with the grid. Run either with::

    python -m experiments.ablations --config configs/ablation/knob_cost_lvis.yaml --out results.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
from pathlib import Path
from typing import Any

import yaml

from experiments.run import run_experiment, use_utf8_stdio


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    node = config
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def expand_arms(base: dict[str, Any]) -> list[dict[str, Any]]:
    """One config per entry of the ``arms:`` block; the base itself when there is none.

    An arm is ``{name: <str>, set: {<dotted key>: <value>}}``. ``set`` may be omitted — that is the
    control arm (the base config unchanged), which is what the all-oracle ceiling row is.
    """
    base = copy.deepcopy(base)
    arms = base.pop("arms", None)
    if not arms:
        return [base]
    base_name = base.get("run_name", "ablation")
    out = []
    for i, arm in enumerate(arms):
        cfg = copy.deepcopy(base)
        for key, value in (arm.get("set") or {}).items():
            _set_dotted(cfg, key, value)
        cfg["run_name"] = f"{base_name}__{arm.get('name', f'arm{i}')}"
        # Keep the arm's identity out of the config knobs and in the MLflow tags, so a results
        # table can group by arm without re-deriving it from the override dict.
        cfg.setdefault("tags", {})["arm"] = str(arm.get("name", f"arm{i}"))
        out.append(cfg)
    return out


def expand_sweep(base: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one config per point in ``arms:`` x the cartesian product of ``sweep:``."""
    if base.get("arms"):
        return [cfg for arm in expand_arms(base) for cfg in expand_sweep(arm)]
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


def run_sweep(base: dict[str, Any], out: Path | None = None) -> list[dict[str, Any]]:
    """Run every expanded config; optionally write the metrics table to ``out`` as CSV.

    The CSV is written **after every arm**, not at the end: an ablation is a long batch and a crash
    on row 5 must not cost the four rows already paid for.
    """
    configs = expand_sweep(base)
    results: list[dict[str, Any]] = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n=== [{i}/{len(configs)}] {cfg['run_name']} ===")
        row = {"run_name": cfg["run_name"], "arm": (cfg.get("tags") or {}).get("arm", "")}
        try:
            row.update(run_experiment(cfg))
        except Exception as exc:  # noqa: BLE001 — one arm failing must not kill the ablation
            import traceback
            traceback.print_exc()
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
        if out is not None:
            _write_csv(results, out)
    if out is not None:
        print(f"\n[ablations] wrote {len(results)} row(s) -> {out}")
        print(_summary(results))
    return results


#: The columns a knob-cost table is read by, in the order it is read: accuracy first, then what
#: that accuracy cost. Everything else the runs logged stays in the CSV behind them.
_HEADLINE = ("arm", "run_name", "inter_ap", "inter_ap50", "inter_ap75", "inter_pq",
             "inter_mean_iou", "inter_count_error", "inter_mean_embeds", "inter_mean_leaf_calls",
             "inter_median_runtime_s", "intra_ap", "intra_ap50", "intra_mean_embeds",
             "intra_mean_leaf_calls", "intra_median_runtime_s", "error")


def _write_csv(results: list[dict[str, Any]], out: Path) -> None:
    keys = list(_HEADLINE) + sorted({k for r in results for k in r} - set(_HEADLINE))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _summary(results: list[dict[str, Any]]) -> str:
    """A fixed-width headline table — the thing you actually read off the terminal."""
    cols = [c for c in _HEADLINE if any(r.get(c) not in (None, "") for r in results)]
    widths = [max(len(c), *(len(_fmt(r.get(c))) for r in results)) for c in cols]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cols, widths)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(_fmt(r.get(c)).ljust(w) for c, w in zip(cols, widths)) for r in results]
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))


def main(argv: list[str] | None = None) -> None:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description="Run a foveate ablation sweep.")
    parser.add_argument("--config", required=True,
                        help="Ablation config (base + a sweep: and/or arms: block).")
    parser.add_argument("--out", default=None,
                        help="Write the per-arm metrics table here as CSV (updated after each arm).")
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    run_sweep(base, out=Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
