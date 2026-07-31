"""Why was this ground-truth instance not detected? (roadmap A0)

The all-oracle runs showed that ~32 % of GT instances get **no prediction at all**, even with a
perfect Where, a perfect Extract and a perfect Stop. "Not detected" is not one failure, though — it
is three, with three different fixes:

``never_framed``
    No crop the cascade visited ever framed the instance. The recursion did not *look* there, so the
    problem is in the **search**: the proposals, the survivor rule, or the budget.
``not_emitted``
    A crop did frame it, but no mask came out of that crop (or the mask was too poor to match). The
    problem is in the **emit path**: the size filters, the mask upsampling, the component that was
    chosen to emit.
``suppressed``
    A mask matching the instance *was* emitted and then removed by the final NMS. The problem is
    **ranking**: a worse detection outscored a better one.

Everything is measured against the same match threshold as AP (IoU ≥ 0.5), so the counts add up to
the AP50 recall exactly.

The pre-merge detections are obtained by running the cascade with ``merge_rule: none`` and then
applying the configured Merge rule afterwards, so both sets come from one forward pass and differ
*only* by that rule.

    python -m experiments.miss_diagnostics --config configs/ablation/lvis_dense_a1.yaml --limit 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_EPS = 1e-9


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    return inter / (np.count_nonzero(a) + np.count_nonzero(b) - inter + _EPS)


def _mask_box(mask: np.ndarray):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _box_iou(a, b) -> float:
    iy0, iy1 = max(a[0], b[0]), min(a[1], b[1])
    ix0, ix1 = max(a[2], b[2]), min(a[3], b[3])
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    if inter == 0:
        return 0.0
    area_a = (a[1] - a[0]) * (a[3] - a[2])
    area_b = (b[1] - b[0]) * (b[3] - b[2])
    return inter / (area_a + area_b - inter + _EPS)


def _contained_frac(mask: np.ndarray, box) -> float:
    """Fraction of ``mask``'s pixels lying inside ``box``."""
    total = np.count_nonzero(mask)
    if total == 0:
        return 0.0
    y0, y1, x0, x1 = box
    return np.count_nonzero(mask[y0:y1, x0:x1]) / total


@dataclass
class MissBreakdown:
    """Per-GT-instance outcome, pooled over images."""

    n_gt: int = 0
    detected: int = 0
    suppressed: int = 0
    not_emitted: int = 0
    never_framed: int = 0
    #: Of the never-framed ones: how many were at least *inside* some visited crop (the recursion
    #: passed over them but never tightened onto them) — search that started but did not finish.
    seen_but_never_framed: int = 0
    per_image: list[tuple[int, int]] = field(default_factory=list)   # (n_gt, detected)

    def merge(self, other: "MissBreakdown") -> "MissBreakdown":
        for f in ("n_gt", "detected", "suppressed", "not_emitted", "never_framed",
                  "seen_but_never_framed"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        self.per_image.extend(other.per_image)
        return self

    def to_metrics(self, prefix: str = "") -> dict[str, float]:
        p = f"{prefix}_" if prefix else ""
        n = max(self.n_gt, 1)
        return {f"{p}miss_n_gt": float(self.n_gt),
                f"{p}miss_detected_frac": self.detected / n,
                f"{p}miss_suppressed_frac": self.suppressed / n,
                f"{p}miss_not_emitted_frac": self.not_emitted / n,
                f"{p}miss_never_framed_frac": self.never_framed / n}

    def report(self) -> str:
        n = max(self.n_gt, 1)
        rows = [("detected", self.detected), ("suppressed by NMS", self.suppressed),
                ("framed but not emitted", self.not_emitted), ("never framed", self.never_framed)]
        out = [f"miss diagnostics — {self.n_gt} GT instances over {len(self.per_image)} images"]
        for name, v in rows:
            out.append(f"  {name:<24s} {v:5d}  {v / n:6.1%}")
        if self.never_framed:
            out.append(f"    of the never-framed, {self.seen_but_never_framed} "
                       f"({self.seen_but_never_framed / max(self.never_framed, 1):.0%}) were inside "
                       f"some visited crop but never isolated")
        return "\n".join(out)


@dataclass
class PredBreakdown:
    """Per-*prediction* outcome — the mirror of :class:`MissBreakdown`, pooled over images.

    The miss breakdown answers "which objects did we lose?". With an oracle Extract and an oracle
    Stop that number is already small, and what limits AP is the opposite question: **what are all
    the masks we emit that are not true positives?** Precision, not recall.

    ``matched``
        Claimed a GT instance at IoU >= 0.5 under the same greedy score-ranked matching AP uses.
    ``duplicate``
        Would have matched at IoU >= 0.5, but a higher-scoring detection already claimed that
        instance. A second, redundant detection of an object we already found — the Merge slot's
        job, and the one an overlap rule can miss when the two masks barely overlap each other.
    ``partial``
        Best IoU with any instance is in ``(0, 0.5)``: it lands *on* an object but not well enough
        to count — a fragment, a straddler covering two neighbours, or an over-zoomed piece. No
        suppression rule can repair these; only a union or a better emission point can.
    ``background``
        Overlaps no instance at all. With an oracle extractor this should be ~0; anything else
        means masks are being emitted from regions the ground truth calls empty.
    """

    n_pred: int = 0
    matched: int = 0
    duplicate: int = 0
    partial: int = 0
    background: int = 0

    def merge(self, other: "PredBreakdown") -> "PredBreakdown":
        for f in ("n_pred", "matched", "duplicate", "partial", "background"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        return self

    def to_metrics(self, prefix: str = "") -> dict[str, float]:
        p = f"{prefix}_" if prefix else ""
        n = max(self.n_pred, 1)
        return {f"{p}fp_n_pred": float(self.n_pred),
                f"{p}fp_matched_frac": self.matched / n,
                f"{p}fp_duplicate_frac": self.duplicate / n,
                f"{p}fp_partial_frac": self.partial / n,
                f"{p}fp_background_frac": self.background / n}

    def report(self) -> str:
        n = max(self.n_pred, 1)
        rows = [("matched a GT instance", self.matched), ("duplicate of one", self.duplicate),
                ("partial (0 < IoU < 0.5)", self.partial), ("background (IoU 0)", self.background)]
        out = [f"prediction breakdown — {self.n_pred} detections",
               f"  precision {self.matched / n:.1%}"]
        for name, v in rows:
            out.append(f"  {name:<24s} {v:5d}  {v / n:6.1%}")
        return "\n".join(out)


def classify_predictions(gt_masks, pred_masks, scores, *, iou_match: float = 0.5) -> PredBreakdown:
    """Classify every prediction into matched / duplicate / partial / background.

    Matching is greedy in descending score order against unclaimed GT, which is exactly what
    :func:`experiments.eval._match` does, so ``matched`` is the TP count AP50 sees.
    """
    gt = [np.asarray(m, dtype=bool) for m in gt_masks]
    pred = [np.asarray(m, dtype=bool) for m in pred_masks]
    out = PredBreakdown(n_pred=len(pred))
    taken = [False] * len(gt)

    for i in np.argsort(-np.asarray(scores, dtype=np.float64)) if len(pred) else []:
        ious = [_iou(pred[i], g) for g in gt]
        best = float(max(ious)) if ious else 0.0
        if best <= 0.0:
            out.background += 1
            continue
        free = [j for j, t in enumerate(taken) if not t and ious[j] >= iou_match]
        if free:
            j = max(free, key=lambda j: ious[j])
            taken[j] = True
            out.matched += 1
        elif best >= iou_match:
            out.duplicate += 1                      # good enough, but the instance is already taken
        else:
            out.partial += 1
    return out


def classify_misses(
    gt_masks, pre_masks, post_masks, visited_boxes, *,
    iou_match: float = 0.5, frame_iou: float = 0.5, contain_frac: float = 0.9,
) -> MissBreakdown:
    """Classify every GT instance into detected / suppressed / not_emitted / never_framed.

    ``pre_masks`` are the cascade's emitted leaves *before* NMS, ``post_masks`` the survivors.
    ``visited_boxes`` is every crop box the observer saw. ``frame_iou`` defines "framed": the crop's
    box and the instance's box overlap at least this much — the same notion the oracle Stop rule
    peaks on, so "framed" means "a crop existed that the Stop rule could have emitted from".
    """
    gt = [np.asarray(m, dtype=bool) for m in gt_masks]
    pre = [np.asarray(m, dtype=bool) for m in pre_masks]
    post = [np.asarray(m, dtype=bool) for m in post_masks]
    out = MissBreakdown(n_gt=len(gt))
    detected = 0

    for g in gt:
        if any(_iou(g, p) >= iou_match for p in post):
            out.detected += 1
            detected += 1
            continue
        if any(_iou(g, p) >= iou_match for p in pre):
            out.suppressed += 1                     # emitted, then lost to the score-ranked NMS
            continue
        gbox = _mask_box(g)
        if gbox is not None and any(_box_iou(gbox, b) >= frame_iou for b in visited_boxes):
            out.not_emitted += 1                    # a crop framed it; nothing usable came out
            continue
        out.never_framed += 1
        if gbox is not None and any(_contained_frac(g, b) >= contain_frac for b in visited_boxes):
            out.seen_but_never_framed += 1
    out.per_image.append((len(gt), detected))
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(config: dict, limit: int = 8, target: str = "intra"
        ) -> tuple[MissBreakdown, PredBreakdown]:
    """Run with Merge off, re-apply it, and classify every GT instance AND every prediction.

    Both breakdowns come from the same forward pass, so the recall side (what did we lose?) and the
    precision side (what did we emit that is not a true positive?) describe one run.
    """
    import copy

    from data import DataConfig
    from experiments.datasets import build_datasets, build_support_index, iter_inter_items, \
        iter_intra_items
    from experiments.methods import build_method
    from foveate.config import Config
    from foveate.merge_rule import build_merge_rule
    from foveate.types import Instance

    cfg = copy.deepcopy(config)
    cfg.pop("sweep", None)
    cfg["mlflow"] = {"enabled": False}
    fov = cfg.setdefault("foveate", {})
    merge_rule = build_merge_rule(Config.from_dict(fov))
    fov["merge_rule"] = "none"                      # disable inside the cascade; re-applied below,
                                                    # so both sets come from ONE forward pass
    # A Merge rule reasons about the CROP a mask was found in — ``merge_drop_clipped`` asks whether
    # the mask reaches that crop's border. Ask the method for crop boxes so the rule sees what it
    # would see inside the cascade; nothing here scores boxes, so this cannot affect the breakdown.
    fov["report_boxes"] = "crop"

    method = build_method(cfg)
    intra, inter = build_datasets(DataConfig.from_dict(cfg["data"]))
    if target == "inter":
        items = iter_inter_items(inter, build_support_index(intra),
                                 max_exemplars=cfg.get("eval", {}).get("max_exemplars", 5),
                                 limit=limit)
    else:
        items = iter_intra_items(intra, max_exemplars=cfg.get("eval", {}).get("max_exemplars", 5),
                                 limit=limit)

    total, preds = MissBreakdown(), PredBreakdown()
    for item in items:
        trace: list[dict] = []
        pred = method.predict(item, observer=lambda ev: trace.append(
            {k: v for k, v in ev.items() if k in ("box", "decision")}))
        pre = pred.masks.astype(bool)
        # ``pred.boxes`` is (x0, y0, x1, y1); Instance.box is (y0, y1, x0, x1). Transpose rather
        # than substituting the mask's own tight box — a mask always touches its tight box, so the
        # clipped-fragment rule would delete every detection.
        boxes = pred.boxes if pred.boxes is not None else np.zeros((pre.shape[0], 4))
        leaves = [Instance(m.astype(np.uint8),
                           (int(b[1]), int(b[3]), int(b[0]), int(b[2])), 0, float(s))
                  for m, b, s in zip(pre, boxes, pred.scores)]
        kept, _, _ = merge_rule.merge(leaves)
        post = np.stack([i.mask.astype(bool) for i in kept]) if kept else np.zeros((0, *pre.shape[1:]), bool)

        visited = [ev["box"] for ev in trace if ev.get("box") is not None]
        total.merge(classify_misses(item.gt_masks, pre, post, visited))
        preds.merge(classify_predictions(item.gt_masks, post,
                                         [i.score for i in kept]))
    return total, preds


def main(argv: list[str] | None = None) -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--target", default="intra", choices=("intra", "inter"))
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="dotted override, e.g. foveate.stop_rule=oracle")
    args = ap.parse_args(argv)

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    for ov in args.overrides:
        key, _, raw = ov.partition("=")
        node = config
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)

    misses, preds = run(config, limit=args.limit, target=args.target)
    print(misses.report())
    print()
    print(preds.report())


if __name__ == "__main__":
    main()
