"""Vendored standalone DINOv3 backbone (the default :class:`~foveate.types.Backbone`).

Lifted from ``iquana_toolbox.ai.backbones.dinov3`` with the iquana dependency removed — it
needs only ``torch`` + ``transformers``. The service can construct its own instance and pass
it to :func:`foveate.discover_instances` to share weights.

DINOv3 (and DINOv2) resize any input to a fixed square ``image_size`` and patchify with
``patch_size``, so the patch grid is ``image_size // patch_size`` per side regardless of the
input aspect ratio.
"""

from __future__ import annotations

import numpy as np
import torch

# DINOv3 ViT-S/16 on the HF hub. Override via ``model_id``.
_DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv3Backbone:
    """Frozen DINOv3 dense feature extractor.

    Parameters
    ----------
    model_id:
        HF model id. DINOv3 weights are gated — set ``HF_TOKEN`` / log in with
        ``huggingface-cli login`` first.
    image_size:
        Square side every input is resized to. Must be a multiple of ``patch_size``.
    device, dtype:
        Where/how to run. Defaults to CUDA if available, else CPU; float32.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        image_size: int = 768,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        from transformers import AutoModel

        self.model_id = model_id
        self.image_size = int(image_size)
        
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            # Find an available GPU by checking memory usage
            best_device = "cpu"
            best_free_memory = -1
            for gpu_id in range(torch.cuda.device_count()):
                try:
                    free_memory = torch.cuda.mem_get_info(gpu_id)[0]
                    if free_memory > best_free_memory:
                        best_free_memory = free_memory
                        best_device = f"cuda:{gpu_id}"
                except RuntimeError:
                    continue
            self.device = torch.device(best_device)
        else:
            self.device = torch.device("cpu")
        
        self.dtype = dtype

        self.model = AutoModel.from_pretrained(model_id).to(self.device, dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} must be a multiple of patch_size={self.patch_size}."
            )
        self.grid = self.image_size // self.patch_size

        mean = torch.tensor(_IMAGENET_MEAN, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=dtype).view(1, 3, 1, 1)
        self._mean = mean.to(self.device)
        self._std = std.to(self.device)

    # ------------------------------------------------------------------
    def preprocess(self, images: np.ndarray | list[np.ndarray]) -> torch.Tensor:
        """Resize + normalize one image or a list to ``(B, 3, S, S)``.

        Accepts HxWx3 uint8/float arrays (RGB). Values are scaled to [0, 1] if they look
        like uint8 (max > 1).
        """
        import torch.nn.functional as F

        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]

        tensors = []
        for img in images:
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            t = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device).float()
            if t.max() > 1.5:
                t = t / 255.0
            t = t.permute(2, 0, 1)[:3]                          # (3, H, W)
            t = F.interpolate(
                t.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
            tensors.append(t)
        batch = torch.cat(tensors, dim=0).to(self.dtype)        # (B, 3, S, S)
        return (batch - self._mean) / self._std

    @torch.inference_mode()
    def __call__(
        self, pixel_values: torch.Tensor, return_cls: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``(B, 3, S, S)`` -> patch grid ``(B, C, Hp, Wp)`` (+ CLS ``(B, C)``)."""
        out = self.model(pixel_values=pixel_values.to(self.device, self.dtype))
        hidden = out.last_hidden_state                          # (B, 1 + R + P, C)
        b, n, c = hidden.shape
        n_patches = self.grid * self.grid
        n_special = n - n_patches                               # CLS + register tokens
        cls = hidden[:, 0]                                      # (B, C)
        patches = hidden[:, n_special:]                         # (B, P, C)
        grid = patches.transpose(1, 2).reshape(b, c, self.grid, self.grid)
        if return_cls:
            return grid, cls
        return grid
