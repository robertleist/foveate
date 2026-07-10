"""Streamlit frontend for the foveate recursive instance-discovery pipeline (DINOv3 only).

Visualizes the two questions the reworked pipeline asks:

- **WHERE** is the concept on a crop?  -> the INSID3 foreground-extraction algorithm
  (fine-grained clusters + debiased forward/backward similarity matching).
- **WHAT** does a crop contain?  -> a converged crop is ALWAYS split k=2 and each sub-crop is
  kept only if it re-identifies the exemplar MORE strongly in CLS cosine (the same rule as zoom);
  if neither sub-crop beats it, the crop itself is emitted. CLS (plus the ``min_crop`` size floor)
  is the only stopping signal; ``crop_sim_floor`` is the class *floor* (reject below it).

Run with::

    uv run --extra frontend --extra dino streamlit run frontend/app.py

DINOv3 weights are gated -- set ``HF_TOKEN`` and request access on the HF hub first.

This module *does* import streamlit (it is the app entry point) but all heavy / model
work is wrapped in cached resources and ``st.spinner`` blocks so a missing-weights error
surfaces as a clear ``st.error`` rather than a crash.
"""

from __future__ import annotations

import sys
import time
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


@st.cache_resource(show_spinner=False)
def load_coco_source(root: str, annotation_file: str, images_dir: str):
    """Build a COCOSource for a local COCO-format folder (cached across reruns)."""
    from data.sources.coco import COCOSource

    src = COCOSource(root=root, annotation_file=annotation_file or None,
                     images_dir=images_dir or None)
    src.ensure_available()                                  # raises a clear FileNotFoundError
    return src


@st.cache_data(show_spinner=False)
def coco_index(root: str, annotation_file: str, images_dir: str):
    """Lightweight, picklable index: per-image category counts + a category-id→name map."""
    src = load_coco_source(root, annotation_file, images_dir)
    cat_names = {cid: src.category_name(cid) for cid in src.category_ids()}
    index = []
    for m in src.list_images():
        counts: dict[int, int] = {}
        for a in m.annotations:
            counts[a.category_id] = counts.get(a.category_id, 0) + 1
        index.append({"image_id": m.image_id, "counts": counts})
    return index, cat_names


@st.cache_data(show_spinner=False)
def coco_load(root: str, annotation_file: str, images_dir: str, image_id: str):
    """One COCO sample → (image uint8 HxWx3, list[(category_id, mask bool HxW)])."""
    src = load_coco_source(root, annotation_file, images_dir)
    img_t = src.load_image(image_id)                       # [3, H, W] float [0,1]
    image = (img_t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    masks = [(int(m.category_id), np.asarray(m.mask, dtype=bool))
             for m in src.load_masks(image_id)]
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
        "insid3_tau_fg": ui["insid3_tau_fg"],
        "insid3_aggt": ui["insid3_aggt"],
        "insid3_top_k_exemplars": ui["insid3_top_k_exemplars"],
        "insid3_dynamic_tau_fg": ui["insid3_dynamic_tau_fg"],
        "insid3_dynamic_aggt": ui["insid3_dynamic_aggt"],
        "insid3_tau_fg_scale": ui["insid3_tau_fg_scale"],
        "insid3_aggt_scale": ui["insid3_aggt_scale"],
        "debias": ui["debias"],
        "debias_subspace_dim": ui["debias_subspace_dim"],
        "crop_sim_floor": ui["crop_sim_floor"],
        "split_mode": ui["split_mode"],
        "cluster_tau": ui["cluster_tau"],
        "zoom_split_retry_eps": ui["zoom_split_retry_eps"],
        "boundary_smooth_sigma": ui["boundary_smooth_sigma"],
        "gate_threshold": ui["gate_threshold"],
        "min_crop": ui["min_crop"],
        "nms_iou": ui["nms_iou"],
        "nms_containment": ui["nms_containment"],
    })


def _take_ratio(masks: list, ratio: float) -> list:
    """The ``ratio`` fraction of ``masks`` with the largest area (>= 1), as bool arrays.

    Largest-first so the exemplars are the most prominent instances — deterministic across reruns.
    """
    if not masks:
        return []
    k = max(1, int(np.ceil(len(masks) * ratio)))
    order = sorted(range(len(masks)), key=lambda i: -int(np.asarray(masks[i]).sum()))
    return [np.asarray(masks[i], dtype=bool) for i in order[:k]]


def _exemplar_thumb(image: np.ndarray, mask: np.ndarray, color, pad_frac: float = 0.08):
    """Overlay ``mask`` on ``image`` cropped to the mask's padded bbox — the *actual* exemplar
    INSID3 embeds (``insid3_crop_reference``), not the whole image. ``pad_frac`` mirrors
    ``cfg.pad_frac`` (the shared bank/target framing) so the thumbnail matches the reference crop 1:1."""
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.where(m)
    if ys.size == 0:
        return viz.overlay_mask(image, m, color)
    h, w = m.shape
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    py, px = int((y1 - y0) * pad_frac), int((x1 - x0) * pad_frac)
    y0, y1 = max(0, y0 - py), min(h, y1 + py)
    x0, x1 = max(0, x0 - px), min(w, x1 + px)
    return viz.overlay_mask(image[y0:y1, x0:x1], m[y0:y1, x0:x1], color)


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
    "- **WHAT** does a crop contain? → a converged crop is **always split k=2**. The split is kept "
    "when its **best** sub-crop beats the parent (isolating a real object raises CLS); once "
    "confirmed, **every** sub-crop above `crop_sim_floor` is kept — so a *novel* instance sharing the "
    "crop with the exemplar is not discarded for scoring below the exemplar-biased parent. If no "
    "sub-crop beats the parent, the crop itself is emitted. CLS + the `min_crop` size floor are the "
    "only stops."
)

with st.sidebar:
    st.header("Model")
    model_id = st.text_input("model_id", value="facebook/dinov3-vits16-pretrain-lvd1689m")
    image_size = st.number_input("image_size", min_value=64, max_value=2048, value=768, step=16)
    device = st.selectbox("device", ["auto", "cpu", "cuda"], index=0)

    st.header("Foreground (WHERE)")
    extractor = st.selectbox("extractor", ["insid3"], index=0)
    insid3_top_k_exemplars = st.number_input(
        "insid3_top_k_exemplars", 1, 32, 1, 1,
        help="Per crop, run INSID3 on the K exemplars most CLS-similar to it (1 = standard "
             "single-exemplar INSID3).")
    insid3_dynamic_tau_fg = st.checkbox(
        "insid3_dynamic_tau_fg", value=False,
        help="Derive tau from the crop↔exemplar CLS sim s (overrides the static insid3_tau_fg "
             "slider): tau = tau_scale·(1-s) — a dissimilar crop is split into many fine "
             "clusters to find the object; a frame-filling match only needs fg/bg.")
    insid3_dynamic_aggt = st.checkbox(
        "insid3_dynamic_aggt", value=False,
        help="Derive aggregate_threshold from the crop↔exemplar CLS sim s (overrides the static "
             "slider): aggregate = aggregate_scale·s — a dissimilar crop's many clusters "
             "re-merge freely.")
    if insid3_dynamic_tau_fg:
        insid3_tau_fg_scale = st.slider("insid3_tau_fg_scale", 0.0, 1.5, 0.3, 0.01,
                                     help="Dynamic tau = tau_scale · (1 - CLS sim).")
        insid3_tau_fg = 0.6
    else:
        insid3_tau_fg_scale = 0.3
        insid3_tau_fg = st.slider("insid3_tau_fg", 0.0, 1.0, 0.6, 0.01)
    if insid3_dynamic_aggt:
        insid3_aggt_scale = st.slider("insid3_aggt_scale", 0.0, 1.0, 0.3, 0.01,
                                           help="Dynamic aggregate_threshold = aggregate_scale · CLS sim.")
        insid3_aggt = 0.2
    else:
        insid3_aggt_scale = 0.3
        insid3_aggt = st.slider("insid3_aggt", 0.0, 1.0, 0.2, 0.01)
        #gate_threshold = st.slider("gate_threshold (bank)", 0.0, 1.0, 0.55, 0.01)
    gate_threshold = 0.55  # This is deprecated
    st.header("Debias / features")
    standardize = st.checkbox("standardize", value=True)
    debias = st.checkbox("debias", value=True)
    debias_subspace_dim = st.number_input("debias_subspace_dim", 1, 64, 8, 1)

    st.header("Acceptance & splitting (WHAT) / recursion")
    st.caption("Stopping is re-identification-only: a converged crop is **always** split k=2. The "
               "split is kept when its best sub-crop's re-id score g beats the parent; then every "
               "sub-crop above `crop_sim_floor` (τ_C) is kept (novel siblings included), else the crop "
               "itself is emitted. The single geometric bound is the `min_crop` size floor ρ.")
    crop_sim_floor = st.slider("crop_sim_floor (τ_C, crop similarity floor)", 0.0, 1.0, 0.3, 0.01)
    split_mode = st.selectbox("split_mode", ["kmeans", "watershed", "agglomerative", "none"], index=0,
                              help="How a converged crop is split: kmeans (k=2 on features — "
                                   "always splits) | watershed (marker-controlled) | agglomerative "
                                   "(cluster the clump's foreground patches at cluster_tau) | none "
                                   "(disable splitting, accept converged crops whole).")
    cluster_tau = st.slider("cluster_tau (distance threshold)", 0.0, 1.0, 0.18, 0.01,
                            disabled=(split_mode not in ("agglomerative", "watershed")),
                            help="Agglomerative cosine-distance threshold. In agglomerative split "
                                 "it directly sets the sub-crop granularity (lower = more, finer "
                                 "splits); in watershed it sizes the marker over-segmentation.")
    zoom_split_retry_eps = st.slider("zoom_split_retry_eps", 0.0, 0.1, 0.01, 0.005,
                                     help="When a zoom peaks by less than this (parent CLS beats the "
                                          "child by a hair), try ONE split of the parent before "
                                          "emitting — it may be a clump. 0 disables.")
    boundary_smooth_sigma = st.slider("boundary_smooth_sigma", 0.0, 3.0, 0.0, 0.1,
                                      disabled=(split_mode != "watershed"),
                                      help="Gaussian (patches) on watershed maps; >0 = fewer fragments.")
    min_crop = st.number_input("min_crop (px size floor)", 8, 512, 64, 8,
                               help="Stop zooming/splitting once a crop side is at or below this. "
                                    "This is the ONLY geometric stop — there is no depth cap.")
    st.caption("Dedup: nested k=2 sub-crops can re-find the same object down two branches. A final "
               "NMS keeps the best detection of each. Set both to 1.0 to disable.")
    nms_iou = st.slider("nms_iou", 0.0, 1.0, 0.5, 0.05,
                        help="Suppress a lower-scored leaf overlapping a kept one above this mask IoU.")
    nms_containment = st.slider("nms_containment", 0.0, 1.0, 0.7, 0.05,
                                help="...or contained in a kept one beyond this fraction of its area "
                                     "— catches the nested duplicate a plain IoU misses.")

ui = dict(
    model_id=model_id, image_size=int(image_size), device=device,
    extractor=extractor, insid3_tau_fg=insid3_tau_fg,
    insid3_aggt=insid3_aggt, gate_threshold=gate_threshold,
    insid3_top_k_exemplars=int(insid3_top_k_exemplars),
    insid3_dynamic_tau_fg=bool(insid3_dynamic_tau_fg),
    insid3_dynamic_aggt=bool(insid3_dynamic_aggt),
    insid3_tau_fg_scale=float(insid3_tau_fg_scale),
    insid3_aggt_scale=float(insid3_aggt_scale),
    standardize=standardize, debias=debias, debias_subspace_dim=int(debias_subspace_dim),
    crop_sim_floor=crop_sim_floor, split_mode=split_mode, cluster_tau=float(cluster_tau),
    zoom_split_retry_eps=float(zoom_split_retry_eps),
    boundary_smooth_sigma=boundary_smooth_sigma, min_crop=int(min_crop),
    nms_iou=float(nms_iou), nms_containment=float(nms_containment),
)

# ---------------------------------------------------------------------------
# Inputs: synthetic sample OR uploads.
# ---------------------------------------------------------------------------
st.header("1 · Inputs")
src_mode = st.radio("Source", ["Built-in synthetic sample", "COCO-style folder", "Upload"],
                    horizontal=True)

ref_image = None         # uint8 HxWx3, where the exemplar lives (single-image / display fallback)
target_image = None      # uint8 HxWx3, where we discover instances
exemplar_masks: list[np.ndarray] = []
gt_masks = None          # list[full-res bool] = ALL instances of the chosen class in the TARGET
exemplar_images = None   # list[uint8 HxWx3] parallel to exemplar_masks (multi-image); else None
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
        gt_masks = [np.asarray(m.mask, dtype=bool) for m in masks if m.category_id == sel]
    ref_image = target_image = _to_uint8_rgb(image)
    image_id = f"synthetic-{idx}"
elif src_mode == "COCO-style folder":
    st.caption("Point at a local COCO-format dataset directory. `root` contains the annotation "
               "JSON and the images folder. Requires `pycocotools`.")
    cc0, cc1, cc2 = st.columns(3)
    coco_root = cc0.text_input("dataset root (path)", value=r"C:\Users\role01-admin\Downloads\corals_coco",
                               placeholder="/data/coco  or  C:\\data\\corals")
    coco_ann = cc1.text_input("annotation_file", value="annotations.json")
    coco_imgs = cc2.text_input("images_dir", value="images")

    if coco_root:
        try:
            with st.spinner("Indexing COCO dataset…"):
                index, cat_names = coco_index(coco_root, coco_ann, coco_imgs)
        except Exception as exc:  # noqa: BLE001 — surface bad paths / missing pycocotools cleanly
            st.error(f"Could not load COCO dataset: `{type(exc).__name__}: {exc}`")
            index = None

        if index:
            # Class first — it defines what we prompt with and pool across images.
            cat_opts = [f"{cid}: {cat_names.get(cid, cid)}" for cid in sorted(cat_names)]
            cat_pick = st.selectbox("category", cat_opts, index=0)
            sel_cid = int(cat_pick.split(":")[0])
            cand = [e for e in index if e["counts"].get(sel_cid, 0) >= 1]      # images with the class

            if not cand:
                st.warning("No images contain this category.")
            else:
                fc0, fc1 = st.columns(2)
                tmode = fc0.radio("Target", ["intra (same image)", "cross (different image)"],
                                  help="intra: prompt and discover in the same image. "
                                       "cross: pool exemplars from N images, discover in another.")
                known_pct = fc1.slider("exemplar ratio (% of instances)", 5, 100, 40, 5,
                                       help="Fraction of the category's instances (per image) used "
                                            "as exemplars/prompts.")
                cand_ids = [e["image_id"] for e in cand]
                cnt = {e["image_id"]: e["counts"].get(sel_cid, 0) for e in cand}

                if tmode.startswith("intra"):
                    tgt_id = st.selectbox("target image", cand_ids,
                                          format_func=lambda i: f"{i} ({cnt[i]} instances)")
                    image, cmasks = coco_load(coco_root, coco_ann, coco_imgs, tgt_id)
                    cat_masks = [m for cid, m in cmasks if cid == sel_cid]
                    exemplar_masks = _take_ratio(cat_masks, known_pct / 100)
                    gt_masks = cat_masks
                    exemplar_images = None
                    ref_image = target_image = _to_uint8_rgb(image)
                    image_id = f"coco-{tgt_id}"
                    st.caption(f"intra · image `{tgt_id}` — {len(exemplar_masks)}/{len(cat_masks)} "
                               f"instances as exemplars, discover the rest in the same image.")
                else:
                    st.info("Cross-image → enable **debias** in the sidebar to correct DINOv3's "
                            "positional bias.")
                    n_max = max(1, len(cand) - 1)
                    ec0, ec1 = st.columns(2)
                    n_ex_imgs = ec0.number_input("exemplar images", 1, n_max, min(3, n_max), 1)
                    tgt_id = ec1.selectbox("target image", cand_ids,
                                           format_func=lambda i: f"{i} ({cnt[i]} instances)")
                    # Exemplar images: the first N candidates that are not the target.
                    ex_ids = [i for i in cand_ids if i != tgt_id][:int(n_ex_imgs)]
                    ex_masks, ex_imgs = [], []
                    for iid in ex_ids:
                        im, cm = coco_load(coco_root, coco_ann, coco_imgs, iid)
                        chosen = _take_ratio([m for cid, m in cm if cid == sel_cid], known_pct / 100)
                        im_u8 = _to_uint8_rgb(im)
                        ex_masks.extend(chosen)
                        ex_imgs.extend([im_u8] * len(chosen))
                    timg, tmasks = coco_load(coco_root, coco_ann, coco_imgs, tgt_id)
                    target_image = _to_uint8_rgb(timg)
                    gt_masks = [m for cid, m in tmasks if cid == sel_cid]
                    exemplar_masks = ex_masks
                    exemplar_images = ex_imgs if ex_masks else None
                    ref_image = ex_imgs[0] if ex_imgs else None
                    image_id = f"coco-{tgt_id}"
                    st.caption(f"cross · {len(ex_masks)} exemplars pooled from {len(ex_ids)} image(s) "
                               f"→ discover the category in target `{tgt_id}`.")
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
    multi = exemplar_images is not None
    same_image = not multi and (target_image is None or target_image is ref_image)
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

    from foveate import foveate_cascade
    from foveate.foreground import build_extractor

    # ---------------- 1 · Reference panel ----------------
    st.header("2 · Exemplar Bank")
    # Each exemplar is shown as the padded-bbox CROP INSID3 actually embeds — not the whole image.
    ref_bases = exemplar_images if multi else [ref_image] * len(exemplar_masks)
    if multi:
        n_imgs = len({id(im) for im in exemplar_images})
        st.caption(f"{len(exemplar_masks)} exemplar instance(s) pooled from {n_imgs} image(s), "
                   f"cropped to each exemplar — cross-image, target is a different image.")
    else:
        st.caption(f"{len(exemplar_masks)} exemplar mask(s), cropped to each exemplar.")
    gal = st.columns(16)
    for k, (im, m) in enumerate(zip(ref_bases, exemplar_masks)):
        thumb = _exemplar_thumb(im, m, viz._PALETTE[k % len(viz._PALETTE)], cfg.pad_frac)
        gal[k % len(gal)].image(thumb, caption=f"exemplar {k}", width="content")
    rc1 = st.container()

    # Build the foreground extractor; it exposes the per-exemplar CLS bank.
    try:
        with st.spinner("Building INSID3 reference / CLS bank…"):
            extractor_obj = build_extractor(cfg)
            extractor_obj.set_reference(
                backbone, (exemplar_images if multi else ref_image), exemplar_masks,
                negative_masks=None, cfg=cfg,
            )
    except Exception as exc:  # noqa: BLE001
        _hf_error_hint(exc)
        st.stop()

    exemplar_cls = getattr(extractor_obj, "exemplar_cls", None)
    if exemplar_cls is not None:
        bank_np = exemplar_cls.detach().cpu().numpy() if hasattr(exemplar_cls, "detach") else np.asarray(exemplar_cls)
        rc1.write(f"`exemplar_cls.shape` = {tuple(bank_np.shape)}")
        if bank_np.ndim == 2 and bank_np.shape[0] >= 1:
            norms = np.linalg.norm(bank_np, axis=1)
            if bank_np.shape[0] >= 2:
                sim = bank_np @ bank_np.T  # already L2-normalized → cosine

                rc1.caption("Pairwise CLS cosine")
                rc1.dataframe(np.round(sim, 3))
    else:
        rc1.info("Extractor exposed no `exemplar_cls`.")

    # ---------------- 3 · Step-by-step cascade ----------------
    st.header("3 · Step-by-step cascade")
    events: list[dict] = []

    try:
        t0 = time.time()
        with st.spinner("Running foveate_cascade (recursive zoom)…", show_time=True):
            instances, stats = foveate_cascade(
                backbone, target_image, exemplar_masks,
                negative_masks=None, config=cfg,
                exemplar_image=None if (same_image or multi) else ref_image,
                exemplar_images=exemplar_images,
                observer=events.append,
            )
        t1 = time.time()
        st.caption(f"Ran in {t1 - t0:.1f}s")
    except Exception as exc:  # noqa: BLE001
        _hf_error_hint(exc)
        st.stop()

    sc = st.columns(5)
    sc[0].metric("instances", len(instances))
    sc[1].metric("embeds", stats.n_embeds)
    sc[2].metric("max depth", stats.max_depth)
    sc[3].metric("discarded", stats.discarded)
    sc[4].metric("NMS-suppressed", stats.suppressed,
                 help="Duplicate detections of one object (e.g. nested k=2 sub-crops) dropped by the "
                      "final NMS.")

    # Producer note per decision — what the node did with its extracted instances.
    _split_name = {"kmeans": "k=2 means", "watershed": "watershed",
                   "agglomerative": "agglomerative"}.get(cfg.split_mode, cfg.split_mode)
    def _producer(decision: str, n_grids: int) -> str:
        return {
            "split": f"Connected components found {n_grids} instances → recurse",
            "clump-split": f"Converged → {_split_name} split into {n_grids} sub-crops; kept if the "
                           f"best beats the parent, then every sub-crop above the class floor "
                           f"(novel siblings included), else this crop is emitted",
            "zoom": "1 component, not converged → zoom in",
            "leaf": "Converged & unsplittable (or splitting off) → accepted as 1 instance",
            "leaf-cap": f"min-crop size floor → emitted {n_grids} instance(s)",
            "reid-stop": "No split/zoom child beat this crop → emit it as the instance",
            "discard": "Below class floor → discarded",
            "empty": "Foreground gate fired on nothing",
        }.get(decision, decision)

    # ---------------- 3 · Result ----------------
    st.header("3 · Result")
    hw = target_image.shape[:2]
    pred_masks = [np.asarray(inst.mask, dtype=bool) for inst in instances]



    if gt_masks:
        from experiments.eval import ImagePrediction, evaluate
        pmasks = np.stack(pred_masks) if pred_masks else np.zeros((0, *hw), dtype=bool)
        pscores = np.array([float(inst.score) for inst in instances], dtype=np.float64)
        # box AP is scored on the final crop box (inst.box is (y0,y1,x0,x1) → [x0,y0,x1,y1]).
        pboxes = np.array(
            [(inst.box[2], inst.box[0], inst.box[3], inst.box[1]) for inst in instances],
            dtype=np.float64,
        ).reshape(-1, 4)
        gt = np.stack([np.asarray(m, dtype=bool) for m in gt_masks])
        with st.spinner("Evaluating predictions", show_time=True):
            metrics = evaluate([ImagePrediction(masks=pmasks, scores=pscores, boxes=pboxes)], [gt])
        mc = st.columns(7)
        mc[0].metric("AP", f"{metrics.ap:.3f}", help="mask AP, mean over IoU 0.50:0.95")
        mc[1].metric("AP@50", f"{metrics.ap50:.3f}", help="mask AP@50")
        mc[2].metric("box AP", f"{metrics.box_ap:.3f}",
                     help="detection AP over box IoU, mean 0.50:0.95")
        mc[3].metric("box AP@50", f"{metrics.box_ap50:.3f}", help="detection AP@50")
        mc[4].metric("mean IoU", f"{metrics.mean_iou:.3f}", help="mean IoU of matched pairs")
        mc[5].metric("PQ", f"{metrics.pq:.3f}", help="panoptic quality")
        mc[6].metric("#pred / #gt", f"{metrics.n_pred} / {metrics.n_gt}")
    ic0, ic1, ic2 = st.columns(3)
    with st.spinner("Loading prediction plot", show_time=True):
        final_pred = target_image
        for k in range(len(instances)):
            final_pred = viz.overlay_mask(final_pred, pred_masks[k], viz._PALETTE[k % len(viz._PALETTE)])
    ic0.image(final_pred,
              caption="prediction", width="content")
    if gt_masks:
        with st.spinner("Loading GT plot", show_time=True):
            final_gt = target_image
            for k in range(len(gt_masks)):
                final_gt = viz.overlay_mask(final_gt, gt_masks[k], viz._PALETTE[k % len(viz._PALETTE)])
        ic1.image(final_gt,
                  caption="annotation", width="content")
    else:
        ic1.caption("No GT available for this source.")

    # ---------------- 4 · Cascade ----------------
    st.header("4 · Cascade")
    st.caption("The recursive zoom — trajectory tree + per-region steps. Skip if you only need "
               "the result above.")
    try:
        from experiments.trajectories import plot_cls_tree

        ic2.pyplot(plot_cls_tree(events, crop_sim_floor=cfg.crop_sim_floor), width="content")
    except Exception as e:  # noqa: BLE001
        ic2.caption(f"trajectory unavailable: {e}")

    steps = viz.region_tree(events)
    if len(steps) > 60:
        st.info(f"Cascade has {len(steps)} regions — showing the first 60.")
        steps = steps[:60]
    with st.status("Loading cascade steps", expanded=False):
        for path_id, ev in steps:
            with st.expander(f"{path_id}  ·  depth {ev.get('depth')}  ·  {ev.get('decision')}",
                             expanded=False):
                y0, y1, x0, x1 = ev["box"]
                crop = target_image[y0:y1, x0:x1]
                cs = ev.get("reid_score")
                cs_txt = f"{cs:.3f}" if isinstance(cs, (int, float)) else "nan"

                if ev.get("decision") in ("reid-worse", "below-floor"):
                    # A dropped child. Two reasons: reid-worse = no child beat this crop's parent, so
                    # the parent was the peak and the cascade stopped there; below-floor = a stronger
                    # sibling confirmed the split, but this crop fell below BOTH its parent and the
                    # class floor, so it is pruned while the sibling keeps zooming.
                    pcls = ev.get("parent_reid")
                    pcls_txt = f"{pcls:.3f}" if isinstance(pcls, (int, float)) else "?"
                    if ev.get("decision") == "below-floor":
                        st.image(_to_uint8_rgb(crop), caption="crop — weak split sibling → pruned",
                                 width="content")
                        st.caption(f"reid_score {cs_txt} ≤ parent {pcls_txt} and below the class "
                                   f"floor {cfg.crop_sim_floor:.3f} → pruned; a stronger sibling beat "
                                   f"the parent and continued the split.")
                    else:
                        st.image(_to_uint8_rgb(crop), caption="crop — CLS worse than parent → dropped",
                                 width="content")
                        st.caption(f"reid_score {cs_txt} ≤ parent {pcls_txt} → dropped; no child beat "
                                   f"this crop, so the cascade stopped zooming here.")
                    continue

                st.caption(f"box {ev.get('box')}  ·  reid_score {cs_txt}")

                intern = ev.get("internals") or {}

                # --- Chosen exemplar(s) ---
                selected = intern.get("selected_exemplars")
                if selected is not None:
                    st.write(f"**Chosen exemplar(s):** {list(selected)}")
                    ex_cols = st.columns(8)
                    for j, i in enumerate(selected):
                        base = exemplar_images[i] if exemplar_images else ref_image
                        try:
                            thumb = _exemplar_thumb(base, exemplar_masks[i],
                                                    viz._PALETTE[i % len(viz._PALETTE)],
                                                    cfg.pad_frac)
                            ex_cols[j % len(ex_cols)].image(thumb, caption=f"exemplar {i}",
                                                            width="content")
                        except Exception:  # noqa: BLE001 — bad index / shape mismatch
                            ex_cols[j % len(ex_cols)].caption(f"exemplar {i} (unavailable)")
                    sel_sim = intern.get("select_sim")
                    sim_txt = f"{sel_sim:.3f}" if isinstance(sel_sim, (int, float)) else "n/a"
                    st.caption(f"CLS sim to selection: {sim_txt}")
                else:
                    st.caption("single exemplar / bank gate — no per-crop selection")

                # --- Foreground + INSID3 params ---
                col1, col2 = st.columns(2)
                with col1:
                    fg = ev.get("fg")
                    tau = intern.get("tau_used", cfg.insid3_tau_fg)
                    agg = intern.get("aggregate_used", cfg.insid3_aggt)
                    if fg is not None and np.asarray(fg).size:
                        st.image(viz.overlay_mask(crop, np.asarray(fg, dtype=bool), (255, 0, 0)),
                                 caption=f"foreground on crop - tau={tau}, aggregate={agg}", width="content")

                with col2:
                    # --- Extracted instances + producer ---
                    grids = ev.get("instance_grids") or []
                    st.image(viz.instances_overlay(viz._as_rgb(crop), grids),
                             caption=_producer(ev.get("decision"), len(grids)), width="content")

                # --- Decision line ---
                st.write(
                    f"**Decision:** {ev.get('decision')} — "
                    f"{viz._DECISIONS.get(ev.get('decision'), '')}  ·  "
                    f"{len(ev.get('children', []))} child crop(s) enqueued"
                )
                with st.expander("INSID3 internals", expanded=False):
                    shown = False
                    clusters = intern.get("clusters")
                    cols = st.columns(4)
                    if clusters is not None and np.asarray(clusters).size:
                        with cols[0]:
                            st.image(
                                viz.upsample_grid(viz.colorize_labels(np.asarray(clusters)),
                                                  crop.shape[:2]),
                                caption="clusters", width="content")
                            shown = True
                    fsim = intern.get("forward_sim")
                    if fsim is not None and np.asarray(fsim).size:
                        with cols[1]:
                            st.image(viz.heatmap(np.asarray(fsim), crop.shape[:2]),
                                     caption="forward_sim", width="content")
                            shown = True
                    cand = intern.get("candidate_mask")
                    if cand is not None and np.asarray(cand).size:
                        with cols[2]:
                            st.image(viz.overlay_mask(crop, np.asarray(cand, dtype=bool),
                                                      viz._PALETTE[1]),
                                     caption="candidate_mask", width="content")
                            shown = True
                    seed = intern.get("seed")
                    if seed is not None and np.asarray(seed).size:
                        with cols[3]:
                            st.image(viz.overlay_mask(crop, np.asarray(seed, dtype=bool),
                                                      viz._PALETTE[2]),
                                     caption="seed", width="content")
                            shown = True
                    if not shown:
                        st.caption("no internals (bank gate)")
