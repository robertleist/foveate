# foveate

This is the code for the paper *"Foveate: A Training-Free, In-Context, Recursive Method for
Instance Segmentation"*. **Work in progress.**

**Training-free, in-context, recursive instance segmentation from frozen DINOv3 features, guided
by as few as one exemplar.**

A *foveated zoom*: starting from the whole image, the cascade recursively directs the encoder's
fixed patch budget at candidate regions (a connected-components fixed point), re-identifies the
prompted concept against the exemplar bank (crop similarity of CLS tokens), and separates each
instance — discovering *every* instance of a prompted concept within a single image, or across
images.

> Lineage: inspired by **INSID3**, which produces a single cross-image mask in one forward pass.
> Foveate adds the recursive cascade + splitting to turn that into instance discovery.

### The algorithm: Where × Stop × Extract

Three questions, and only the last one is allowed to be expensive:

| Stage | Question | Cost | Config |
|---|---|---|---|
| **Where** | Where is the concept? A *region*, split by connected components; each becomes a child crop. | cheap, per crop | `foreground_extractor` × `instance_extractor: cc` |
| **Stop** | Is this still more than one thing? Descend while the CLS re-identification score `g` rises; emit at its peak. | cheap, per crop | `stop_rule: reid` |
| **Extract** | Once the zoom has bottomed out, *what is in this crop?* | expensive, **per leaf** | `leaf_extractor` |

The descent never asks for an instance decomposition, because it never uses one — it reads the
components' boxes (where to zoom) and `g` (whether to keep zooming). That is also why forcing an
instance-first model into the descent backfires: **SAM 3** and **NTT** predict instances rather
than a similarity region, so they only ever propose to foveate onto what they can already segment,
never onto the ambiguous region a zoom would resolve. Confining them to `leaf_extractor` puts them
where they are strongest — one call, on the best crop the recursion could produce — and drops their
cost from `O(crops visited)` to `O(leaves)` (`Stats.n_leaf_calls`). An empty leaf answer falls back
to the descent's region, so the leaf slot can only refine what the recursion committed to.

Every stage has an **oracle** arm, so the headroom table is one run per knock-out:

```bash
python -m experiments.ablations --config configs/ablation/knob_cost_lvis_dense.yaml \
    --out runs/knob_cost/lvis_dense.csv
```

## Install

**With uv (recommended):**

```bash
uv sync --extra dino --extra exp --extra dev
```

Or install into an existing environment:

```bash
uv pip install -e ".[dino,exp,dev]"
```

**With pip:**

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
from foveate import cascade, Config, DINOv3Backbone

backbone = DINOv3Backbone()  # or MockBackbone() for a smoke test
instances, stats = cascade(
    backbone, image, exemplar_masks, config=Config(gate_threshold=0.5),
)
for inst in instances:
    inst.mask  # (H, W) uint8 in original image coordinates
    inst.score  # re-identification confidence
```

Every knob lives in one `Config` (`from_dict` / `from_request`), threaded through every stage —
no notebook/service drift. The single-pass (non-recursive) pipeline is also exposed as
`foveate.run(...)` for step-by-step visualization.

## Experiments

Mask-based datasets live in the `data` package (COCO, PanNuke, and an offline `synthetic`
source); `experiments/` wraps discovery with metrics and MLflow tracking.

**Protocol.** Per image, a `known_ratio` fraction of instances are `known` (the exemplar
prompts); the rest are `unknown` (the GT to discover). Two evaluation targets (`eval.targets`):

- **intra** (same image) — prompt with an image's `known` instances, discover its `unknown`
  instances in that same image.
- **inter** (cross image) — prompt with `known` instances from a disjoint *support* image and
  discover instances of that class in a novel held-out image (`interval_images`). This is the
  cross-image setting; set `foveate.debias: true` to correct DINOv3's positional bias.

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

## Frontend

A small Streamlit app visualizes the whole pipeline stage by stage on real DINOv3 features —
the **WHERE** (INSID3 foreground extraction: clusters → forward/backward similarity → seed →
aggregated foreground), the recursive **cascade trace**, and the **WHAT** (per-leaf
re-identification table: re-id score g against the exemplar bank).

```bash
uv run --extra frontend --extra dino streamlit run frontend/app.py
```

It needs DINOv3 weights (set `HF_TOKEN` / request hub access first). Load a built-in synthetic
sample or upload a reference image + mask (free-draw canvas, or a mask PNG fallback), tune the
`Config` from the sidebar, and hit **Run**.

## Layout

```
foveate/        core package: cascade (the `cascade` entry point) driving the swappable slots —
                extract (which instances are on this crop: composite = foreground x grouping, or a
                monolithic one; `leaf_extractor` runs a second, expensive one at the leaves only),
                stop (descend/emit/reject), merge_rule (how the leaves combine);
                single-pass pipeline (run), features, gate, clustering, individuation, merge,
                border, prototypes, thresholding, debias, config, types, backbones/{dinov3,mock}
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
