"""Carve the two Track-A tuning venues out of [[LVIS]] — ``general`` and ``dense`` (roadmap A3.0).

Both slices come from **one** dataset on purpose. The paper claims a general in-context segmenter
that *additionally* wins where instances are small and densely packed; to support that, the two
tuning venues must differ in **the regime and nothing else**. A separate dense dataset (aerial, say)
would confound "small and dense" with "different domain, sensor and statistics", and any knob tuned
on it would be un-attributable. Same images, same annotators, same label space — only the filter
changes.

Selection works on **(image, category) pairs**, which is what our protocol actually consumes: prompt
with a few instances of class *c* in image *i*, discover the rest of *c* in *i*.

Two filters matter:

**Exhaustiveness (mandatory).** LVIS is a *federated* dataset: a category is annotated exhaustively
only in images where it is absent from ``not_exhaustive_category_ids``. Elsewhere, real instances are
simply unlabelled — and a discovery method that finds them is punished with false positives it did
not earn. Ignoring this drops **~55 % of the otherwise-eligible dense pairs** into a regime where AP
is systematically depressed and hyperparameters would be ranked against annotation noise.

**Regime (what defines the slice).** ``patches`` = the median instance's side measured in encoder
patches at inference resolution: ``sqrt(area)/max(H, W) * (image_size / patch_size)``. It answers the
only question that matters for a patch-based encoder — *how many patches does one instance get in a
single full-frame pass?* At the default 768/16, LVIS's median instance is ~3 patches and its 10th
percentile is under one, so the dense regime is not exotic: **it is the low tail of an ordinary
benchmark**, and that is exactly why it can be carved out rather than imported.

Output is a COCO-format json per slice, restricted to the selected pairs, with ``file_name`` filled
in from ``coco_url`` so any COCO reader can open it.

    python -m experiments.build_lvis_slices --root datasets/lvis
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import median

#: Slice definitions: (min instances of the class in the image, max median instance size in patches).
#: ``None`` = unbounded. Tuned against the measured LVIS val distribution (see the module docstring).
SLICES = {
    # The parity guard: ordinary multi-instance images, no size restriction at all.
    "general": dict(min_instances=2, max_patches=None),
    # The regime where a fixed patch budget must fail: >= 10 instances, each ~2 patches or less in a
    # single full-frame pass. One instance is then smaller than the encoder's own granularity.
    "dense": dict(min_instances=10, max_patches=2.0),
}


def _pairs(data: dict, image_size: float, patch_size: float):
    """Yield ``(image_id, category_id, n_instances, median_patches)`` for exhaustively-annotated pairs."""
    images = {im["id"]: im for im in data["images"]}
    grouped: dict[tuple[int, int], list] = defaultdict(list)
    for ann in data["annotations"]:
        grouped[(ann["image_id"], ann["category_id"])].append(ann)

    scale = image_size / patch_size
    for (image_id, category_id), anns in grouped.items():
        im = images[image_id]
        if category_id in set(im.get("not_exhaustive_category_ids") or ()):
            continue                       # federated gap — unlabelled instances would score as FPs
        side = max(im["width"], im["height"])
        rel = (median(a["area"] for a in anns) ** 0.5) / side
        yield image_id, category_id, len(anns), rel * scale


def build_slice(data: dict, name: str, spec: dict, *, n_images: int, seed: int,
                image_size: float, patch_size: float) -> dict:
    """Filter ``data`` down to one slice's (image, category) pairs → a COCO-format dict."""
    keep = [
        (i, c) for i, c, n, p in _pairs(data, image_size, patch_size)
        if n >= spec["min_instances"] and (spec["max_patches"] is None or p <= spec["max_patches"])
    ]
    # One category per image keeps the protocol unambiguous (the runner prompts with *a* class), and
    # sampling images rather than pairs keeps the subset's size predictable.
    by_image: dict[int, list[int]] = defaultdict(list)
    for image_id, category_id in keep:
        by_image[image_id].append(category_id)

    rng = random.Random(seed)
    chosen = sorted(by_image)
    rng.shuffle(chosen)
    chosen = sorted(chosen[:n_images])
    pairs = {i: rng.choice(sorted(by_image[i])) for i in chosen}

    images = {im["id"]: im for im in data["images"]}
    out_images = []
    for image_id in chosen:
        im = dict(images[image_id])
        im["file_name"] = im.get("file_name") or (im.get("coco_url") or "").rsplit("/", 1)[-1]
        out_images.append(im)
    out_anns = [a for a in data["annotations"]
                if a["image_id"] in pairs and a["category_id"] == pairs[a["image_id"]]]
    used = {a["category_id"] for a in out_anns}
    out_cats = [c for c in data["categories"] if c["id"] in used]

    counts = [sum(1 for a in out_anns if a["image_id"] == i) for i in chosen]
    print(f"[{name}] {len(keep)} eligible pairs -> {len(out_images)} images, {len(out_anns)} "
          f"instances ({min(counts)}-{max(counts)} per image), {len(out_cats)} classes")
    return {
        "info": {"description": f"LVIS {name} slice (foveate Track-A tuning venue)",
                 "spec": spec, "seed": seed, "n_images": n_images,
                 "image_size": image_size, "patch_size": patch_size},
        "licenses": data.get("licenses", []),
        "images": out_images,
        "annotations": out_anns,
        "categories": out_cats,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="datasets/lvis", type=Path)
    ap.add_argument("--annotations", default="lvis_v1_val.json")
    ap.add_argument("--out", default="slices", help="output dir, relative to --root")
    ap.add_argument("--n-images", type=int, default=50, help="frozen subset size per slice")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-size", type=float, default=768.0)
    ap.add_argument("--patch-size", type=float, default=16.0)
    args = ap.parse_args(argv)

    print(f"loading {args.root / args.annotations} ...")
    data = json.load(open(args.root / args.annotations, encoding="utf-8"))
    out_dir = args.root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, spec in SLICES.items():
        sliced = build_slice(data, name, spec, n_images=args.n_images, seed=args.seed,
                             image_size=args.image_size, patch_size=args.patch_size)
        path = out_dir / f"lvis_{name}.json"
        path.write_text(json.dumps(sliced), encoding="utf-8")
        ids = [im["id"] for im in sliced["images"]]
        (out_dir / f"lvis_{name}.ids.txt").write_text(
            "\n".join(map(str, ids)), encoding="utf-8")
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
