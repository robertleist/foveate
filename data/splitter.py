"""In-memory image selection and known/unknown instance splitting.

These are pure functions of ``(image metadata, ratio, seed)`` — cheap enough to run at
dataset-construction time, so the split never needs to be materialised to a file. Given the
same seed and a stably-ordered image list they are fully reproducible.

Per image, instances are partitioned into two sets:

* ``known``   — the labelled instances. They double as the **exemplar prompts**.
* ``unknown`` — everything else. These are the GT instances the method must *discover*.

A split is ``{image_id: {"known": [...], "unknown": [...]}}`` where the lists hold annotation IDs.
"""

from __future__ import annotations

import math
import random
from typing import Dict, FrozenSet, List, Optional, Tuple

from data.source import ImageMeta


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------

def select_images(
    images: List[ImageMeta],
    selection: str,
    max_images: Optional[int],
    seed: int,
) -> List[ImageMeta]:
    """Select up to ``max_images`` images using the given strategy."""
    if selection == "densest":
        images = sorted(images, key=lambda x: x.instance_count, reverse=True)
    elif selection == "first":
        images = list(images)
    elif selection == "random":
        rng = random.Random(seed)
        images = list(images)
        rng.shuffle(images)
    else:
        raise ValueError(
            f"Unknown selection strategy: {selection!r}. "
            "Choose from: random, first, densest."
        )
    if max_images is not None:
        images = images[:max_images]
    return images


# ---------------------------------------------------------------------------
# Instance partitioning
# ---------------------------------------------------------------------------

def partition_instances(
    image: ImageMeta,
    known_ratio: float,
    stratify_by_class: bool,
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    """Return ``(known_ann_ids, unknown_ann_ids)`` for one image.

    ``known`` are the exemplar prompts (a ``known_ratio`` fraction, with a per-class floor of
    one when ``stratify_by_class`` so every present class can be prompted); ``unknown`` are the
    remaining instances the method must rediscover.
    """
    if stratify_by_class:
        by_class = image.annotations_by_class()
        guaranteed = []
        remaining = []
        for class_anns in by_class.values():
            shuffled = list(class_anns)
            rng.shuffle(shuffled)
            guaranteed.append(shuffled[0])
            remaining.extend(shuffled[1:])

        target_known = max(
            len(guaranteed),
            math.ceil(image.instance_count * known_ratio),
        )
        extra_needed = target_known - len(guaranteed)
        rng.shuffle(remaining)
        extra = remaining[: max(0, extra_needed)]
        unknown_anns = remaining[max(0, extra_needed):]
        known_ids = [a.ann_id for a in guaranteed + extra]
        unknown_ids = [a.ann_id for a in unknown_anns]
    else:
        all_anns = list(image.annotations)
        rng.shuffle(all_anns)
        n_known = max(1, math.ceil(image.instance_count * known_ratio))
        known_ids = [a.ann_id for a in all_anns[:n_known]]
        unknown_ids = [a.ann_id for a in all_anns[n_known:]]

    return known_ids, unknown_ids


# ---------------------------------------------------------------------------
# Top-level split computation
# ---------------------------------------------------------------------------

PUSplit = Dict[str, Dict[str, List[int]]]


def compute_pu_split(
    images: List[ImageMeta],
    known_ratio: float,
    stratify_by_class: bool,
    seed: int,
) -> PUSplit:
    """Compute the per-image known/unknown split for a list of (category-filtered) images."""
    rng = random.Random(seed)
    split: PUSplit = {}
    for image in images:
        known_ids, unknown_ids = partition_instances(
            image, known_ratio, stratify_by_class, rng
        )
        split[str(image.image_id)] = {"known": known_ids, "unknown": unknown_ids}
    return split


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def detect_collisions(inst_splits: List[PUSplit]) -> Tuple[int, int]:
    """Whether instance splits produce distinct known-sets per image.

    Returns ``(n_fully_distinct, n_with_collision)`` over images.
    """
    if len(inst_splits) <= 1:
        return (len(inst_splits[0]) if inst_splits else 0), 0

    image_ids = list(inst_splits[0].keys())
    n_distinct = n_collision = 0
    for image_id in image_ids:
        known_sets: List[FrozenSet[int]] = []
        for split in inst_splits:
            entry = split.get(image_id, {})
            known_sets.append(frozenset(entry.get("known", [])))
        if len(set(known_sets)) == len(known_sets):
            n_distinct += 1
        else:
            n_collision += 1
    return n_distinct, n_collision


def recommended_min_instances(known_ratio: float, n_inst_splits: int) -> int:
    """Smallest N such that C(N, ceil(N * known_ratio)) >= n_inst_splits."""
    for n in range(1, 500):
        k = max(1, math.ceil(n * known_ratio))
        lo = min(k, n - k)
        c = 1
        for x in range(lo):
            c = c * (n - x) // (x + 1)
            if c >= n_inst_splits:
                break
        if c >= n_inst_splits:
            return n
    return math.ceil(n_inst_splits / max(known_ratio, 1e-6))
