# foveate

This is the code for the paper *"Foveate: Recursive zooming for training-free in-context
instance segmentation"*. **Work in progress.**

**Training-free, recursive, prototype-guided instance discovery from frozen DINOv3 features.**

A *foveated zoom*: starting from the whole image, the cascade recursively directs the encoder's
fixed patch budget at candidate regions (a connected-components fixed point), re-identifies the
prompted concept (CLS / prototype similarity), and refines each instance's border — discovering
*every* instance of a prompted morphology within a single image, or across images.

> Lineage: inspired by **INSID3** (CVPR 2026), which produces a single cross-image mask in one
> forward pass. foveate adds recursion + individuation to turn that into instance discovery.

## Install

```bash
pip install -e ".[dino,exp,dev]"
```

- core depends only on `torch numpy opencv scipy scikit-image scikit-learn`;
- `[dino]` adds `transformers` for the default `DINOv3Backbone` (weights are HF-gated — run
  `huggingface-cli login` first);
- `[exp]` adds the experiment/metrics deps (`pycocotools mlflow pyyaml pandas …`);
- `[dev]` adds `pytest ruff`.

No GPU/weights needed for tests and demos — use the weightless `MockBackbone`.

## Quickstart

```python
import numpy as np
from foveate import discover_instances, Config, DINOv3Backbone

backbone = DINOv3Backbone()                       # or MockBackbone() for a smoke test
instances, stats = discover_instances(
    backbone, image, exemplar_masks, config=Config(gate_threshold=0.5),
)
for inst in instances:
    inst.mask     # (H, W) uint8 in original image coordinates
    inst.score    # re-identification confidence
```

Every knob lives in one `Config` (`from_dict` / `from_request`), threaded through every stage —
no notebook/service drift. The single-pass (non-recursive) pipeline is also exposed as
`foveate.run(...)` for step-by-step visualization.

## Experiments

Mask-based datasets live in the `data` package (COCO, PanNuke, and an offline `synthetic`
source); `experiments/` wraps discovery with metrics and MLflow tracking.

```bash
# Offline end-to-end smoke test (synthetic data + mock backbone, no downloads/weights):
python -m experiments.run --config configs/mock_smoke.yaml

# Real run on a COCO subset (needs the [dino] extra + DINOv3 weights):
python -m experiments.run --config configs/coco_baseline.yaml --set foveate.gate_threshold=0.6

# Ablation sweep -> one MLflow run per grid point:
python -m experiments.ablations --config configs/ablation_thresholds.yaml
```

**Metrics** (class-agnostic, `experiments/eval.py`): mask **AP / AP50 / AP75**, **Panoptic
Quality** (PQ/SQ/RQ), mean per-instance **IoU**, **count error**, and **exemplar-recovery IoU**.
Params, metrics, per-image cost (embeds, runtime) and the saved masks are logged to MLflow.

## Layout

```
foveate/        core package: cascade (discover_instances), single-pass pipeline (run),
                features, gate, clustering, individuation, merge, border, prototypes,
                thresholding, debias, config, types, backbones/{dinov3,mock}
data/           mask-based datasets: DatasetSource (coco/pannuke/synthetic), InstanceDataset,
                PU instance splitting
experiments/    eval (metrics), datasets (bridge), run (MLflow runner), ablations (sweeps)
configs/        example experiment + ablation configs
tests/          fast mock-backbone / synthetic-data tests (no weights, no downloads)
```

## Tests

```bash
pytest
```
