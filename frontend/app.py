"""Streamlit frontend for the foveate recursive instance-discovery pipeline (DINOv3 only).

Visualizes the two questions the reworked pipeline asks:

- **WHERE** is the concept on a crop?  -> the INSID3 foreground-extraction algorithm
  (fine-grained clusters + debiased forward/backward similarity matching).
- **WHAT** does a crop contain?  -> mean cosine similarity of the crop's CLS token to all
  exemplar CLS tokens, thresholded by ``cls_threshold``.

Run with::

    uv run --extra frontend --extra dino streamlit run frontend/app.py

DINOv3 weights are gated -- set ``HF_TOKEN`` and request access on the HF hub first.

This module *does* import streamlit (it is the app entry point) but all heavy / model
work is wrapped in cached resources and ``st.spinner`` blocks so a missing-weights error
surfaces as a clear ``st.error`` rather than a crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Make the repo root importable when launched as `streamlit run frontend/app.py`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from frontend import viz  # noqa: E402

st.set_page_config(page_title="foveate — instance discovery", layout="wide")


# ---------------------------------------------------------------------------
# Cached heavy resources.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_backbone(model_id: str, image_size: int, device: str):
    """Load DINOv3 once (weights are heavy / gated). Cached across reruns."""
    from foveate import DINOv3Backbone

    dev = None if device == "auto" else device
    return DINOv3Backbone(model_id=model_id, image_size=int(image_size), device=dev)


@st.cache_data(show_spinner=False)
def load_synthetic(idx: int, image_size: int, n_instances: int, n_categories: int):
    """Generate one synthetic sample -> (image uint8 HxWx3, list[InstanceMask])."""
    from data.sources.synthetic import SyntheticSource

    src = SyntheticSource(
        n_images=idx + 1, image_size=int(image_size),
        n_instances=int(n_instances), n_categories=int(n_categories),
    )
    img_t = src.load_image(str(idx))                       # (3, H, W) float [0,1]
    image = (img_t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    masks = src.load_masks(str(idx))
    return image, masks


# ---------------------------------------------------------------------------
# Small input helpers (pure-ish, kept here to avoid bloating viz.py).
# ---------------------------------------------------------------------------
def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.5 else arr.clip(0, 255).astype(np.uint8)
    return arr


def _build_config(ui: dict):
    from foveate import Config

    return Config.from_dict({
        "standardize": ui["standardize"],
        "foreground_extractor": ui["extractor"],
        "insid3_tau": ui["insid3_tau"],
        "insid3_aggregate_threshold": ui["insid3_aggregate_threshold"],
        "debias": ui["debias"],
        "debias_subspace_dim": ui["debias_subspace_dim"],
        "cls_threshold": ui["cls_threshold"],
        "gate_threshold": ui["gate_threshold"],
        "max_depth": ui["max_depth"],
    })


def _hf_error_hint(exc: Exception) -> None:
    st.error(
        "Failed to load / run DINOv3. The weights are gated on the Hugging Face hub.\n\n"
        "1. Request access to the model on huggingface.co.\n"
        "2. Set an access token: `setx HF_TOKEN <your-token>` (then restart), or "
        "`huggingface-cli login`.\n\n"
        f"Underlying error: `{type(exc).__name__}: {exc}`"
    )


# ---------------------------------------------------------------------------
# Sidebar.
# ---------------------------------------------------------------------------
st.title("foveate — recursive instance discovery")
st.markdown(
    "Two questions, visualized stage by stage:\n"
    "- **WHERE** is the concept on a crop? → the **INSID3** foreground extractor "
    "(fine-grained clusters + debiased similarity matching).\n"
    "- **WHAT** does a crop contain? → **mean cosine similarity of the crop CLS** to all "
    "exemplar CLS, thresholded by `cls_threshold`."
)

with st.sidebar:
    st.header("Model")
    model_id = st.text_input("model_id", value="facebook/dinov3-vits16-pretrain-lvd1689m")
    image_size = st.number_input("image_size", min_value=64, max_value=2048, value=768, step=16)
    device = st.selectbox("device", ["auto", "cpu", "cuda"], index=0)

    st.header("Foreground (WHERE)")
    extractor = st.selectbox("extractor", ["insid3", "bank"], index=0)
    insid3_tau = st.slider("insid3_tau", 0.0, 1.0, 0.6, 0.01)
    insid3_aggregate_threshold = st.slider("insid3_aggregate_threshold", 0.0, 1.0, 0.2, 0.01)
    gate_threshold = st.slider("gate_threshold (bank)", 0.0, 1.0, 0.55, 0.01)

    st.header("Debias / features")
    standardize = st.checkbox("standardize", value=True)
    debias = st.checkbox("debias", value=False)
    debias_subspace_dim = st.number_input("debias_subspace_dim", 1, 64, 8, 1)

    st.header("Classification (WHAT) / recursion")
    cls_threshold = st.slider("cls_threshold", 0.0, 1.0, 0.5, 0.01)
    max_depth = st.number_input("max_depth", 0, 16, 8, 1)

ui = dict(
    model_id=model_id, image_size=int(image_size), device=device,
    extractor=extractor, insid3_tau=insid3_tau,
    insid3_aggregate_threshold=insid3_aggregate_threshold, gate_threshold=gate_threshold,
    standardize=standardize, debias=debias, debias_subspace_dim=int(debias_subspace_dim),
    cls_threshold=cls_threshold, max_depth=int(max_depth),
)

# ---------------------------------------------------------------------------
# Inputs: synthetic sample OR uploads.
# ---------------------------------------------------------------------------
st.header("1 · Inputs")
src_mode = st.radio("Source", ["Built-in synthetic sample", "Upload"], horizontal=True)

ref_image = None         # uint8 HxWx3, where the exemplar lives
target_image = None      # uint8 HxWx3, where we discover instances
exemplar_masks: list[np.ndarray] = []
image_id = "input"

if src_mode == "Built-in synthetic sample":
    c0, c1, c2, c3 = st.columns(4)
    idx = c0.number_input("sample idx", 0, 64, 0, 1)
    n_instances = c1.number_input("n_instances", 1, 16, 5, 1)
    n_categories = c2.number_input("n_categories", 1, 6, 2, 1)
    syn_size = c3.number_input("synthetic size", 64, 1024, 256, 32)
    image, masks = load_synthetic(int(idx), int(syn_size), int(n_instances), int(n_categories))
    # Pick the most-populated category's masks as exemplars; same image is the target (intra).
    cats = [m.category_id for m in masks]
    if cats:
        chosen = max(set(cats), key=cats.count)
        sel = st.selectbox("exemplar category", sorted(set(cats)),
                           index=sorted(set(cats)).index(chosen))
        exemplar_masks = [np.asarray(m.mask, dtype=bool) for m in masks if m.category_id == sel]
    ref_image = target_image = _to_uint8_rgb(image)
    image_id = f"synthetic-{idx}"
else:
    c0, c1 = st.columns(2)
    ref_up = c0.file_uploader("Reference image", type=["png", "jpg", "jpeg"])
    tgt_up = c1.file_uploader("Target image (default = reference)", type=["png", "jpg", "jpeg"])
    mask_up = c0.file_uploader("Exemplar mask PNG (thresholded > 0)", type=["png"])

    from PIL import Image  # local import: PIL is a frontend extra

    if ref_up is not None:
        ref_image = _to_uint8_rgb(np.array(Image.open(ref_up).convert("RGB")))
        target_image = ref_image
        image_id = "uploaded"
    if tgt_up is not None and ref_image is not None:
        target_image = _to_uint8_rgb(np.array(Image.open(tgt_up).convert("RGB")))

    # Mask input: try the drawable canvas (guarded), else fall back to a mask PNG.
    drew = None
    if ref_image is not None:
        try:
            from streamlit_drawable_canvas import st_canvas

            st.caption("Draw the exemplar mask on the reference image (free-draw).")
            h, w = ref_image.shape[:2]
            scale = min(1.0, 512.0 / max(h, w))
            canvas = st_canvas(
                fill_color="rgba(255,0,0,0.4)", stroke_width=20, stroke_color="#ff0000",
                background_image=Image.fromarray(ref_image),
                height=int(h * scale), width=int(w * scale),
                drawing_mode="freedraw", key="mask_canvas",
            )
            if canvas.image_data is not None:
                alpha = np.asarray(canvas.image_data)[..., 3]
                if alpha.any():
                    drew = viz.upsample_grid(alpha > 0, (h, w)).astype(bool)
        except Exception:
            st.info("`streamlit_drawable_canvas` unavailable — upload a mask PNG instead.")

    if drew is not None and drew.any():
        exemplar_masks = [drew]
    elif mask_up is not None and ref_image is not None:
        m = np.array(Image.open(mask_up).convert("L"))
        m = viz.upsample_grid(m, ref_image.shape[:2])
        exemplar_masks = [m > 0]

run = st.button("Run", type="primary", disabled=(ref_image is None or not exemplar_masks))
if ref_image is None:
    st.info("Load a synthetic sample or upload a reference image + mask to begin.")
elif not exemplar_masks:
    st.warning("No exemplar mask yet — draw one, upload a mask PNG, or pick a synthetic category.")


# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
if run and ref_image is not None and exemplar_masks:
    cfg = _build_config(ui)
    same_image = target_image is None or target_image is ref_image
    if target_image is None:
        target_image = ref_image

    # --- backbone ---
    try:
        with st.spinner("Loading DINOv3 weights…"):
            backbone = load_backbone(ui["model_id"], ui["image_size"], ui["device"])
    except Exception as exc:  # noqa: BLE001 — surface gated-weights errors cleanly
        _hf_error_hint(exc)
        st.stop()

    st.success(f"DINOv3 ready on device `{backbone.device}` — grid {backbone.grid}×{backbone.grid}.")

    from foveate import discover_instances
    from foveate import features as featlib
    from foveate.foreground import build_extractor

    # ---------------- 1 · Reference panel ----------------
    st.header("2 · Reference (the exemplar prompt)")
    ref_overlay = ref_image
    for k, m in enumerate(exemplar_masks):
        ref_overlay = viz.overlay_mask(ref_overlay, m, viz._PALETTE[k % len(viz._PALETTE)])
    rc0, rc1 = st.columns([2, 1])
    rc0.image(ref_overlay, caption=f"{len(exemplar_masks)} exemplar mask(s)", use_column_width=True)

    # Build the foreground extractor; it exposes the per-exemplar CLS bank.
    try:
        with st.spinner("Building INSID3 reference / CLS bank…"):
            extractor_obj = build_extractor(cfg)
            extractor_obj.set_reference(
                backbone, ref_image, exemplar_masks,
                negative_masks=None, cfg=cfg,
            )
    except Exception as exc:  # noqa: BLE001
        _hf_error_hint(exc)
        st.stop()

    cls_bank = getattr(extractor_obj, "cls_bank", None)
    if cls_bank is not None:
        bank_np = cls_bank.detach().cpu().numpy() if hasattr(cls_bank, "detach") else np.asarray(cls_bank)
        rc1.write(f"`cls_bank.shape` = {tuple(bank_np.shape)}")
        if bank_np.ndim == 2 and bank_np.shape[0] >= 1:
            norms = np.linalg.norm(bank_np, axis=1)
            rc1.caption("Per-exemplar CLS norm")
            rc1.bar_chart(norms)
            if bank_np.shape[0] >= 2:
                sim = bank_np @ bank_np.T  # already L2-normalized → cosine
                rc1.caption("Pairwise CLS cosine")
                rc1.dataframe(np.round(sim, 3))
    else:
        rc1.info("Extractor exposed no `cls_bank`.")

    # ---------------- 2 · INSID3 stages on the target ----------------
    st.header("3 · WHERE — foreground extraction on the target")
    try:
        with st.spinner("Embedding target & running the extractor…"):
            target_feat = featlib.embed_image(backbone, target_image, standardize=cfg.standardize)
            gr = extractor_obj.predict(target_feat, return_internals=True)
    except Exception as exc:  # noqa: BLE001
        _hf_error_hint(exc)
        st.stop()

    hw = target_image.shape[:2]
    internals = getattr(gr, "internals", {}) or {}

    def _zoom_overlay(grid_bool, color, alpha=0.45):
        """Upsample a patch-grid bool onto the target image as an overlay."""
        return viz.overlay_mask(target_image, np.asarray(grid_bool, dtype=bool), color, alpha)

    # Raw feature PCA + (optional) debiased feature PCA.
    pca_cols = st.columns(4)
    pca_cols[0].image(
        viz.upsample_grid(viz.feature_pca_rgb(target_feat), hw),
        caption="raw feature PCA", use_column_width=True,
    )
    # Debiased PCA: recompute the basis via foveate.debias and project out.
    if cfg.debias:
        try:
            from foveate.debias import estimate_positional_basis, project_out

            B = estimate_positional_basis(
                backbone, subspace_dim=cfg.debias_subspace_dim,
                standardize=cfg.standardize,
            )
            hp, wp, d = target_feat.shape
            deb = project_out(target_feat.reshape(hp * wp, d), B).reshape(hp, wp, d)
            pca_cols[1].image(
                viz.upsample_grid(viz.feature_pca_rgb(deb), hw),
                caption="debiased feature PCA", use_column_width=True,
            )
        except Exception:  # noqa: BLE001 — degrade gracefully if interface differs
            pca_cols[1].caption("debiased PCA unavailable")
    else:
        pca_cols[1].caption("debias off")

    # Score map + aggregated foreground (always available from GateResult).
    score_map = getattr(gr, "score_map", None)
    if score_map is not None and np.asarray(score_map).size:
        pca_cols[2].image(
            viz.heatmap(np.asarray(score_map), hw),
            caption="score_map (forward/CLS confidence)", use_column_width=True,
        )
    fg = getattr(gr, "foreground", None)
    if fg is not None and np.asarray(fg).size:
        pca_cols[3].image(
            _zoom_overlay(fg, viz._PALETTE[2]),
            caption="aggregated foreground", use_column_width=True,
        )

    if cfg.foreground_extractor == "insid3":
        # INSID3 internal stages — guard every key with .get(...).
        stage_specs = [
            ("clusters", "fine-grained clusters", "labels"),
            ("forward_sim", "forward similarity", "heat"),
            ("backward_candidates", "backward candidates", "mask"),
            ("candidate_mask", "candidate clusters", "mask"),
            ("seed", "seed cluster", "mask"),
            ("foreground", "aggregated foreground", "mask"),
        ]
        cols = st.columns(3)
        rendered = 0
        for key, caption, kind in stage_specs:
            val = internals.get(key)
            if val is None or not np.asarray(val).size:
                continue
            col = cols[rendered % 3]
            arr = np.asarray(val)
            if kind == "labels":
                col.image(viz.upsample_grid(viz.colorize_labels(arr), hw),
                          caption=caption, use_column_width=True)
            elif kind == "heat":
                col.image(viz.heatmap(arr, hw), caption=caption, use_column_width=True)
            else:  # mask overlay
                col.image(_zoom_overlay(arr.astype(bool), viz._PALETTE[rendered % len(viz._PALETTE)]),
                          caption=caption, use_column_width=True)
            rendered += 1
        if rendered == 0:
            st.info("Extractor returned no INSID3 internals (keys absent) — showing score/fg only.")

        scores = internals.get("cluster_scores")
        if isinstance(scores, dict) and scores:
            st.caption("Per-cluster scores (cross / intra / combined)")
            rows = [
                {"cluster": cid, "cross": s.get("cross"), "intra": s.get("intra"),
                 "combined": s.get("combined")}
                for cid, s in scores.items()
            ]
            st.dataframe(rows, use_container_width=True)
    else:
        st.caption("extractor = 'bank' — only the score map + foreground are meaningful (above).")

    # ---------------- 3 · Cascade ----------------
    st.header("4 · Recursive cascade")
    events: list[dict] = []
    try:
        with st.spinner("Running discover_instances (recursive zoom)…"):
            instances, stats = discover_instances(
                backbone, target_image, exemplar_masks,
                negative_masks=None, config=cfg,
                exemplar_image=None if same_image else ref_image,
                observer=events.append,
            )
    except Exception as exc:  # noqa: BLE001
        _hf_error_hint(exc)
        st.stop()

    sc = st.columns(4)
    sc[0].metric("instances", len(instances))
    sc[1].metric("embeds", stats.n_embeds)
    sc[2].metric("max depth", stats.max_depth)
    sc[3].metric("discarded", stats.discarded)

    if events:
        with st.spinner("Rendering cascade trace…"):
            fig = viz.render_trace(target_image, image_id, events)
        st.pyplot(fig)

    # Final instances overlaid with their scores.
    final = target_image
    for k, inst in enumerate(instances):
        final = viz.overlay_mask(final, np.asarray(inst.mask, dtype=bool),
                                 viz._PALETTE[k % len(viz._PALETTE)])
    cap = "  ".join(f"#{k}={inst.score:.2f}" for k, inst in enumerate(instances))
    st.image(final, caption=f"final instances — scores: {cap or '(none)'}", use_column_width=True)

    # ---------------- 4 · Leaf classification (WHAT) ----------------
    st.header("5 · WHAT — leaf classification table")
    converged = [e for e in events if e.get("decision") in {"leaf", "discard", "clump-split"}]
    rows = []
    for e in converged:
        cs = e.get("cls_score")
        rows.append({
            "depth": e.get("depth"),
            "box": str(e.get("box")),
            "decision": e.get("decision"),
            "cls_score": round(cs, 4) if isinstance(cs, (int, float)) else None,
            "accept": bool(cs is not None and cs >= cfg.cls_threshold),
        })
    if rows:
        st.caption(f"accept = `cls_score >= cls_threshold` ({cfg.cls_threshold:.2f})")
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No converged regions to classify (decision in leaf/discard/clump-split).")
