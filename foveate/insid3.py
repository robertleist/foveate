"""The official INSID3 foreground extractor (visinf/INSID3, CVPR 2026).

INSID3 answers "which patches of the target are the prompted class?" without any
training, from frozen DINOv3 features and a handful of reference masks. Our
earlier bank gate scored each target patch in isolation; INSID3 is richer in two
ways that matter for open-world discovery:

* **Discriminative matching.** A target patch is a *candidate* only when its
  nearest neighbour over **all** reference patches (foreground *and* background)
  lands on a foreground patch. This backward test rejects patches that look
  vaguely like the class but match the reference background even better -- the
  failure mode a one-sided prototype cosine cannot see.
* **Cluster-level aggregation.** Candidates are grown into instances at the
  granularity of feature-coherent clusters, not pixels. We over-segment the whole
  target grid, seed from the cluster most similar to the reference prototype, and
  pull in further clusters whose combined cross-class / intra-class agreement
  clears ``aggregate_threshold``. This keeps the region tight while spanning
  whole objects, where a per-patch threshold frays at the edges.

The matching half runs on **positionally debiased** features (the reference and
target are different crops, so raw features share a spurious position signal --
see :mod:`foveate.debias`). The clustering half runs on **raw** features, where
the positional structure actually helps group spatially coherent parts. Every
feature here is L2-normalized, so all cosines are plain dot products.
"""

from __future__ import annotations

import numpy as np
import torch

from foveate import clustering, features as featlib
from foveate.debias import estimate_positional_basis, project_out
from foveate.foreground import GateResult, normalize_reference


def _mask_bbox(mask: np.ndarray, pad_frac: float) -> tuple[int, int, int, int]:
    """Padded bbox of a binary mask -- duplicated from :mod:`foveate.prototypes`."""
    ys, xs = np.where(mask)
    h, w = mask.shape
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    py, px = int((y1 - y0) * pad_frac), int((x1 - x0) * pad_frac)
    return max(0, y0 - py), min(h, y1 + py), max(0, x0 - px), min(w, x1 + px)


class InSID3Extractor:
    """INSID3 foreground extractor behind the :class:`ForegroundExtractor` interface."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.cls_bank: torch.Tensor | None = None
        self._B: torch.Tensor | None = None
        # Per-exemplar reference kept SEPARATE (not pooled): each entry is
        # ``(ref_deb (Rs, D), ref_fg (Rs,) bool, p_ref (D,))``. ``predict`` picks the exemplars
        # most CLS-similar to the crop and builds the matching gallery from just those — pooling
        # every exemplar into one gallery dilutes the match (the failure the redesign fixes).
        self._ex: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # Pooled over ALL exemplars — the fallback used when ``predict`` gets no crop CLS.
        self.ref_deb: torch.Tensor | None = None      # (R, D) debiased reference patches
        self.ref_fg: torch.Tensor | None = None       # (R,) bool reference foreground
        self.p_ref: torch.Tensor | None = None        # (D,) debiased reference prototype

    # ------------------------------------------------------------------ reference
    def set_reference(
        self,
        backbone,
        ref_image: "np.ndarray | list[np.ndarray]",
        ref_masks: list[np.ndarray],
        negative_masks: list[np.ndarray] | None,
        cfg,
    ) -> None:
        """Build the reference state: matching gallery + prototype + per-exemplar CLS bank.

        ``ref_image`` is a single array or a list parallel to ``ref_masks`` (multi-image
        exemplars — each exemplar cropped from its own image; the CLS / prototype / matching
        banks are simply stacked across images).

        With ``cfg.insid3_crop_reference`` (default) the reference is built from a crop around
        *each* exemplar mask padded by ``cfg.pad_frac`` — the SAME fraction the cascade pads its
        child/foreground crops with (``_child_box``) — so the exemplar fills its frame at the exact
        scale the target crops do. This matters for the CLS comparison: DINOv3's CLS token is
        framing-sensitive, so if the bank exemplars carried more padding than the target crops, a
        tightly-zoomed single instance would score *lower* than a looser parent crop purely from the
        framing mismatch (nothing to do with content). Sharing ``pad_frac`` removes that bias.
        Embedding the whole image instead (``insid3_crop_reference=False``) leaves the exemplar tiny
        in a large frame, and that scale mismatch makes INSID3's matching progressively tighter as
        the cascade zooms.
        """
        images, valid = normalize_reference(ref_image, ref_masks)

        self._B = None
        if cfg.debias:
            self._B = estimate_positional_basis(
                backbone, subspace_dim=cfg.debias_subspace_dim, n_noise=cfg.debias_n_noise,
                seed=cfg.debias_seed, standardize=cfg.standardize,
            )

        if cfg.insid3_crop_reference:
            self._set_reference_cropped(backbone, images, valid, cfg)
        else:
            self._set_reference_full(backbone, images, valid, cfg)

    def _set_reference_cropped(self, backbone, images, valid, cfg) -> None:
        """Reference patches + prototype + CLS from a padded crop around each exemplar mask
        (each cropped from its own image in ``images``). Padded by ``cfg.pad_frac`` — the same
        fraction the cascade uses for its crops — so bank and target framing match (see
        :meth:`set_reference`)."""
        boxes = [_mask_bbox(m, cfg.pad_frac) for m in valid]
        crops = [img[y0:y1, x0:x1] for img, (y0, y1, x0, x1) in zip(images, boxes)]
        embedded = featlib.embed_batch(backbone, crops, chunk=8, standardize=cfg.standardize)
        device = embedded[0][0].device

        cls_list = []
        for m, (y0, y1, x0, x1), (feat, cls) in zip(valid, boxes, embedded):
            hp, wp, d = feat.shape
            mask_grid = featlib.resize_mask_to_grid(m[y0:y1, x0:x1], (hp, wp))
            if not mask_grid.any():                              # mask thinner than a patch
                mask_grid = np.ones((hp, wp), dtype=bool)
            rd = project_out(feat.reshape(hp * wp, d), self._B)  # (hp*wp, D)
            rf = torch.from_numpy(mask_grid.reshape(-1)).to(device)
            self._ex.append((rd, rf, featlib.l2_normalize(rd[rf].mean(dim=0), dim=0)))
            cls_list.append(cls)

        self._set_pooled(device)
        self.cls_bank = featlib.l2_normalize(torch.stack(cls_list), dim=1).to(device)  # (S, D)

    def _set_reference_full(self, backbone, images, valid, cfg) -> None:
        """Reference from the whole image(s) (original behaviour); CLS still per-exemplar crop.

        Each distinct reference image is embedded once and its exemplar masks OR-merged into a
        per-image foreground; the per-image patches (and foreground flags) are concatenated, so
        multi-image references simply stack their patch galleries."""
        by_image: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
        order: list[int] = []
        for img, m in zip(images, valid):
            key = id(img)
            if key not in by_image:
                feat = featlib.embed_image(backbone, img, standardize=cfg.standardize)
                by_image[key] = (feat, np.zeros(feat.shape[:2], dtype=bool))
                order.append(key)
            feat, union = by_image[key]
            union |= featlib.resize_mask_to_grid(m, feat.shape[:2])

        device = by_image[order[0]][0].device
        # One per-exemplar entry per distinct image (its patches + OR-merged mask foreground).
        for key in order:
            feat, union = by_image[key]
            hp, wp, d = feat.shape
            rd = project_out(feat.reshape(hp * wp, d), self._B)
            rf = torch.from_numpy(union.reshape(-1)).to(device)
            if not bool(rf.any()):
                continue                                         # no ref patch overlaps this grid
            self._ex.append((rd, rf, featlib.l2_normalize(rd[rf].mean(dim=0), dim=0)))

        if not self._ex:
            raise ValueError("No reference patch overlaps the feature grid; image too small "
                             "or reference mask empty. Increase backbone image_size.")
        self._set_pooled(device)
        self.cls_bank = self._build_cls_bank(backbone, images, valid, cfg).to(device)

    def _set_pooled(self, device) -> None:
        """Concatenate the per-exemplar reference into the all-exemplar pooled fallback."""
        self.ref_deb = torch.cat([e[0] for e in self._ex], dim=0).to(device)
        self.ref_fg = torch.cat([e[1] for e in self._ex], dim=0).to(device)
        self.p_ref = featlib.l2_normalize(self.ref_deb[self.ref_fg].mean(dim=0), dim=0)

    def _build_cls_bank(self, backbone, images, valid, cfg) -> torch.Tensor:
        """Crop each exemplar bbox from its own image, embed it, stack the per-crop CLS. Padded by
        ``cfg.pad_frac`` so the bank CLS is framed like the target crops it will be matched against."""
        boxes = [_mask_bbox(m, cfg.pad_frac) for m in valid]
        crops = [img[y0:y1, x0:x1] for img, (y0, y1, x0, x1) in zip(images, boxes)]
        embedded = featlib.embed_batch(backbone, crops, chunk=8, standardize=cfg.standardize)
        cls_list = [cls for _, cls in embedded]                  # each (D,), L2-normed
        return featlib.l2_normalize(torch.stack(cls_list), dim=1)  # (S, D)

    # ------------------------------------------------------------------- select
    def _select_reference(self, cls: torch.Tensor | None, device):
        """Pick the exemplars to run INSID3 with, and the tau / aggregate to run it at.

        ``cls`` is the crop's CLS token. The top ``insid3_top_k_exemplars`` exemplars by CLS
        cosine build the matching gallery (``k=1`` ⇒ standard single-exemplar INSID3). With
        ``insid3_dynamic_params`` the granularity follows that similarity ``s``:
        ``tau = insid3_tau_scale * s`` (dissimilar crops stay coarse so the cascade zooms first;
        similar crops cluster finely — scaled because raw ``s≈1`` over-segments) and
        ``aggregate_threshold = insid3_aggregate_scale * (1 - s)`` (similar crops aggregate freely).
        The similarity is measured even for a single exemplar; only ``cls=None`` (the single-pass
        ``predict`` with no crop CLS) falls back to the pooled reference and static config.
        """
        cfg = self.cfg
        n = len(self._ex)
        if cls is None:
            sel, sim = list(range(n)), None
        else:
            # Always measure the crop↔exemplar CLS similarity when we have a CLS — a single
            # exemplar still needs it for the dynamic tau / aggregate. Only the top-k *selection*
            # is trivial when n == 1 (sims is (1,) → sel == [0]).
            sims = (self.cls_bank.to(cls.device, cls.dtype) @ cls).detach().cpu().numpy()  # (S,)
            k = max(1, min(int(cfg.insid3_top_k_exemplars), n))
            sel = sorted(int(i) for i in np.argsort(-sims)[:k])
            sim = float(np.mean(sims[sel]))

        ref_deb = torch.cat([self._ex[i][0] for i in sel], dim=0).to(device)
        ref_fg = torch.cat([self._ex[i][1] for i in sel], dim=0).to(device)
        p_ref = (featlib.l2_normalize(ref_deb[ref_fg].mean(dim=0), dim=0)
                 if bool(ref_fg.any()) else featlib.l2_normalize(ref_deb.mean(dim=0), dim=0))

        if cfg.insid3_dynamic_params and sim is not None:
            tau = float(np.clip(cfg.insid3_tau_scale * sim, 0.05, 0.95))
            aggregate = float(np.clip(cfg.insid3_aggregate_scale * (1.0 - sim), 0.0, 0.95))
        else:
            tau, aggregate = float(cfg.insid3_tau), float(cfg.insid3_aggregate_threshold)
        return ref_deb, ref_fg, p_ref, tau, aggregate, sel, sim

    def _selection_internals(self, sel_ex, sim, tau, aggregate) -> dict:
        """Small viz payload describing which exemplars ran and at what granularity."""
        return {"selected_exemplars": list(sel_ex), "select_sim": sim,
                "tau_used": tau, "aggregate_used": aggregate}

    # ------------------------------------------------------------------- predict
    def predict(
        self, target_feat: torch.Tensor, *, cls: torch.Tensor | None = None,
        return_internals: bool = False,
    ) -> GateResult:
        """Run INSID3 on an ``(Hp, Wp, D)`` L2-normalized target grid.

        ``cls`` is the crop's CLS token; when given, only the ``insid3_top_k_exemplars`` exemplars
        most similar to it build the matching gallery (and, if ``insid3_dynamic_params``, set the
        granularity) — see :meth:`_select_reference`.

        Mirrors the official ``_locate_candidates`` + ``_seed_and_aggregate``: locate candidate
        patches (forward prototype prior ∧ backward nearest-neighbour vote), over-segment the
        whole grid in feature space, pick the seed cluster by cross-image similarity, then keep
        clusters whose ``cross · intra · area_weight`` clears the aggregation threshold. The
        **area weight** (fraction of a cluster's patches that are candidates) is what stops
        background clusters — which can still score a non-trivial cross/intra similarity on
        DINOv3 — from being swept into the foreground.
        """
        hp, wp, d = target_feat.shape
        device = target_feat.device
        flat = target_feat.reshape(hp * wp, d)                   # (P, D) raw
        t_deb = project_out(flat, self._B)                       # (P, D) debiased

        ref_deb, ref_fg, p_ref, tau, aggregate, sel_ex, sim = self._select_reference(cls, device)

        # Forward prior: per-patch cosine to the reference prototype.
        fwd = t_deb @ p_ref                                      # (P,)
        forward_mask = fwd > 0

        # Backward matching: keep a patch when its nearest reference patch is foreground.
        sims = t_deb @ ref_deb.T                                 # (P, R)
        nn = sims.argmax(dim=1)                                  # (P,)
        backward = ref_fg[nn]                                    # (P,) bool

        candidate = (forward_mask & backward).cpu().numpy()      # (P,) candidate patches
        fwd_np = fwd.cpu().numpy()
        fwd_grid = fwd_np.reshape(hp, wp)
        score_map = np.clip(_minmax(fwd_grid), 0.0, 1.0)

        # Fine-grained clustering of ALL patches in feature space (no spatial graph).
        clusters = clustering.cluster_all(
            target_feat, distance_threshold=1.0 - tau,
            linkage=self.cfg.insid3_linkage,
        )                                                        # (Hp, Wp) int, labels 0..K-1
        labels = clusters.reshape(-1)
        K = int(labels.max()) + 1 if labels.size else 0

        sel_info = self._selection_internals(sel_ex, sim, tau, aggregate)
        matched_ids = np.unique(labels[candidate]) if candidate.any() else np.empty(0, int)
        if matched_ids.size == 0:                                # nothing matched → raw candidates
            foreground = candidate.reshape(hp, wp)
            return self._make_result(foreground, score_map, clusters, candidate.reshape(hp, wp),
                                     backward.reshape(hp, wp).cpu().numpy(), -1, {},
                                     return_internals, sel_info)

        # Area weight: fraction of each cluster's patches that are candidates (seed forced to 1).
        total_area = np.bincount(labels, minlength=K).astype(np.float64)
        matched_area = np.bincount(labels[candidate], minlength=K).astype(np.float64)
        area_weights = np.divide(matched_area, total_area,
                                 out=np.zeros(K), where=total_area > 0)

        # Per-cluster cross-image similarity (mean debiased→ref-prototype sim over the cluster)
        # and L2-normalized raw / debiased prototypes (for intra sim and seed selection).
        cross_sim = np.zeros(K)
        proto_raw = torch.zeros(K, d, device=device)
        proto_deb = torch.zeros(K, d, device=device)
        for k in range(K):
            sel = torch.from_numpy(labels == k).to(device)
            cross_sim[k] = float(fwd[sel].mean())
            proto_raw[k] = featlib.l2_normalize(flat[sel].mean(dim=0), dim=0)
            proto_deb[k] = featlib.l2_normalize(t_deb[sel].mean(dim=0), dim=0)

        # Seed: matched cluster with the highest cross-image prototype similarity.
        cross_proto = (proto_deb @ p_ref).cpu().numpy()          # (K,)
        seed_id = int(matched_ids[int(np.argmax(cross_proto[matched_ids]))])

        # Intra-image similarity of every cluster to the seed (raw feature space).
        intra_sim = (proto_raw @ proto_raw[seed_id]).cpu().numpy()  # (K,)

        area_weights[seed_id] = 1.0
        combined = cross_sim * intra_sim * area_weights          # (K,)

        keep = combined > aggregate
        keep[seed_id] = True
        foreground = keep[labels].reshape(hp, wp)

        cluster_scores = {
            int(k): {"cross": float(cross_sim[k]), "intra": float(intra_sim[k]),
                     "area": float(area_weights[k]), "combined": float(combined[k])}
            for k in range(K)
        }
        return self._make_result(foreground, score_map, clusters, candidate.reshape(hp, wp),
                                 backward.reshape(hp, wp).cpu().numpy(), seed_id, cluster_scores,
                                 return_internals, sel_info)

    # ------------------------------------------------------------------- helpers
    def _make_result(self, foreground, score_map, clusters, candidate_grid, backward_grid,
                     seed_id, cluster_scores, return_internals, sel_info=None) -> GateResult:
        internals: dict = {}
        if return_internals:
            internals = {
                "clusters": clusters.astype(np.int32),
                "forward_sim": score_map,
                "backward_candidates": backward_grid,
                "candidate_mask": candidate_grid,
                "seed": (clusters == seed_id) if seed_id >= 0 else np.zeros_like(foreground),
                "foreground": foreground,
                "cluster_scores": cluster_scores,
                **(sel_info or {}),
            }
        return GateResult(
            foreground=foreground,
            score_map=score_map,
            cls_bank=self.cls_bank.cpu().numpy(),
            internals=internals,
        )


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
