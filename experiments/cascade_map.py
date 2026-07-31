"""Draw the cascade: every crop it visited, coloured by how deep in the recursion it was.

The trace tooling already renders a contact sheet — one panel per crop — which shows *what* each
crop saw but not *where the recursion went*. This puts the whole search on the image at once: crop
boxes over the frame, coloured from the root (0) to the deepest crop (1). Reading it you can see
directly whether the cascade foveated onto objects or wandered, and where it spent its budget.

    python -m experiments.cascade_map --config configs/ablation/lvis_general_a1.yaml --limit 2 \
        --set foveate.extractor=oracle --out runs/maps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: Decisions whose region is emitted as one or more instances.
_TERMINAL = frozenset({"leaf", "leaf-cap", "reid-stop"})


def _overlay(image: np.ndarray, masks, colors, alpha: float = 0.55) -> np.ndarray:
    """Tint ``image`` with one colour per mask."""
    out = np.asarray(image, dtype=np.float32).copy()
    for m, c in zip(masks, colors):
        m = np.asarray(m, dtype=bool)
        if not m.any():
            continue
        out[m] = (1 - alpha) * out[m] + alpha * np.asarray(c[:3], np.float32) * 255.0
    return out.astype(np.uint8)


def save_cascade_map(image, trace, gt_masks, pred_masks, path: Path, *, title: str = "") -> Path:
    """Three panels: ground truth, the recursion coloured by depth, and the predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Rectangle

    events = [e for e in trace if e.get("box") is not None]
    depths = [int(e.get("depth", 0)) for e in events] or [0]
    dmax = max(max(depths), 1)
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=dmax)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    for ax in axes:
        ax.imshow(image)
        ax.set_xticks([]); ax.set_yticks([])

    gt = [np.asarray(m, bool) for m in (gt_masks if gt_masks is not None else [])]
    axes[0].imshow(_overlay(image, gt, [plt.get_cmap("tab20")(i % 20) for i in range(len(gt))]))
    axes[0].set_title(f"ground truth — {len(gt)} instances")

    # -- the search, coloured by depth ------------------------------------------------------
    # Deep boxes are drawn last and thinner so they stay visible inside their ancestors.
    for e in sorted(events, key=lambda e: int(e.get("depth", 0))):
        y0, y1, x0, x1 = e["box"]
        d = int(e.get("depth", 0))
        terminal = e.get("decision") in _TERMINAL
        axes[1].add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=cmap(norm(d)),
            linewidth=max(0.6, 2.4 - 0.22 * d), linestyle="-" if terminal else (0, (4, 2)),
            alpha=0.95,
        ))
    axes[1].set_title(f"cascade — {len(events)} crops, depth 0…{dmax}\n"
                      f"solid = emitted here, dashed = descended")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[1], fraction=0.046, pad=0.02,
                 label="recursion depth")

    pred = [np.asarray(m, bool) for m in (pred_masks if pred_masks is not None else [])]
    axes[2].imshow(_overlay(image, pred, [plt.get_cmap("tab20")(i % 20) for i in range(len(pred))]))
    axes[2].set_title(f"predictions — {len(pred)} instances")

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def run(config: dict, limit: int, out: Path, target: str = "intra") -> list[Path]:
    import copy

    from data import DataConfig
    from experiments.datasets import (
        build_datasets,
        build_support_index,
        iter_inter_items,
        iter_intra_items,
    )
    from experiments.methods import build_method

    cfg = copy.deepcopy(config)
    cfg.pop("sweep", None)
    cfg["mlflow"] = {"enabled": False}
    method = build_method(cfg)
    intra, inter = build_datasets(DataConfig.from_dict(cfg["data"]))
    max_ex = cfg.get("eval", {}).get("max_exemplars", 5)
    # Some venues are cross-image by construction (``known_ratio: 0`` leaves no intra prompts), so
    # the map has to follow whichever protocol the config is written for.
    items = (iter_inter_items(inter, build_support_index(intra), max_exemplars=max_ex, limit=limit)
             if target == "inter"
             else iter_intra_items(intra, max_exemplars=max_ex, limit=limit))

    name = str(cfg.get("foveate", {}).get("extractor")
               or cfg.get("foveate", {}).get("foreground_extractor", "composite"))
    written: list[Path] = []
    for item in items:
        trace: list[dict] = []
        pred = method.predict(item, observer=lambda ev: trace.append(
            {k: v for k, v in ev.items() if k in ("box", "decision", "depth")}))
        written.append(save_cascade_map(
            item.image, trace, item.gt_masks, pred.masks,
            out / f"{cfg['data'].get('label', 'data')}_{item.image_id}_{name}.png",
            title=f"{name} — image {item.image_id} — {pred.n_embeds} encoder passes",
        ))
        print(f"[map] {written[-1]}")
    return written


def main(argv: list[str] | None = None) -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("runs/maps"))
    ap.add_argument("--target", default="intra", choices=("intra", "inter"))
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args(argv)

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    for ov in args.overrides:
        key, _, raw = ov.partition("=")
        node = config
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    run(config, args.limit, args.out, args.target)


if __name__ == "__main__":
    main()
