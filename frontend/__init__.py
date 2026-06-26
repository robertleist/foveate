"""Streamlit frontend for visualizing the foveate recursive instance-discovery pipeline.

Two sub-modules:

- :mod:`frontend.viz` -- pure, dependency-light rendering helpers (numpy + matplotlib +
  PIL only, no streamlit import at module top).
- :mod:`frontend.app` -- the Streamlit application (DINOv3 only).

The helpers are importable and testable without streamlit or model weights.
"""
