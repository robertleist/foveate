"""The CLS-trajectory tracer reconstructs the zoom tree from observer events alone."""

import numpy as np

from foveate import Config
from experiments.trajectories import (
    build_segments,
    summarize_segment,
    summarize_segments,
    trace_discovery,
    trajectory_report,
)


def test_segments_reconstruct_from_events(backbone, two_squares):
    """Every reconstructed segment node links to its parent by exact box match, and a fork
    (>= 2 children) starts a fresh segment per child rather than continuing the chain."""
    img, ex = two_squares
    cfg = Config(min_crop=24, cascade_min_instance_area=4)
    _, _, events = trace_discovery(backbone, img, ex, config=cfg)

    assert events, "observer produced no events"
    segments = build_segments(events)
    assert segments, "no segments reconstructed"

    # Reconstruction is lossless: every observed region lands in exactly one segment.
    n_nodes = sum(len(s) for s in segments)
    assert n_nodes == len(events)

    # Within a segment, every non-terminal node has exactly one child whose box is the next
    # node's box — i.e. no fork is ever hidden inside a segment.
    for seg in segments:
        for parent, child in zip(seg, seg[1:]):
            assert len(parent.children) == 1
            assert parent.children[0] == child.box


def test_single_node_segment_is_classified_single(backbone, two_squares):
    img, ex = two_squares
    _, _, events = trace_discovery(backbone, img, ex, config=Config(min_crop=24,
                                                                     cascade_min_instance_area=4))
    summaries = summarize_segments(events)
    assert summaries
    for s in summaries:
        if len(s.nodes) == 1:
            assert s.kind == "single"


def test_overshoot_and_rising_detection():
    """Peak-before-stop -> overshoot; monotone-rising-into-stop -> rising. Pure unit check on
    synthetic nodes, independent of the backbone."""
    from experiments.trajectories import TraceNode

    def node(depth, score, decision="zoom"):
        box = (0, 100 - depth, 0, 100 - depth)
        return TraceNode(box=box, depth=depth, cls_score=score, decision=decision,
                         n_components=1, children=[])

    rising = [node(0, 0.40), node(1, 0.55), node(2, 0.70, decision="leaf")]
    s = summarize_segment(rising)
    assert s.kind == "rising"
    assert s.unimodal
    assert s.overshoot == 0.0

    overshoot = [node(0, 0.40), node(1, 0.72), node(2, 0.60, decision="leaf")]
    s = summarize_segment(overshoot)
    assert s.kind == "overshoot"
    assert s.peak_index == 1
    assert abs(s.overshoot - 0.12) < 1e-9

    bimodal = [node(0, 0.4), node(1, 0.7), node(2, 0.5), node(3, 0.65, decision="leaf")]
    assert not summarize_segment(bimodal).unimodal


def test_report_runs(backbone, two_squares):
    img, ex = two_squares
    _, _, events = trace_discovery(backbone, img, ex, config=Config(min_crop=24,
                                                                    cascade_min_instance_area=4))
    report = trajectory_report(events)
    assert "segments" in report
