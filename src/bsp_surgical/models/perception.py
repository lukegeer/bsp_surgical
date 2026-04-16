"""Frozen pretrained vision backbones for feature extraction.

The default is DINOv2-base (facebook/dinov2-base, 86M params, 768-d CLS).
DINOv2 uses patch-14, so inputs must be resized to 224x224 (our SurRoL
frames are 128x128 and get upsampled).

Features are deterministic — no training, no gradient — so downstream
models train on cached features rather than re-running the backbone
every epoch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ImageNet normalization stats that DINOv2 expects
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class BackboneSpec:
    hf_name: str
    feature_dim: int
    input_size: int


_BACKBONES: dict[str, BackboneSpec] = {
    "dinov2-base": BackboneSpec("facebook/dinov2-base", 768, 224),
    "dinov2-base-448": BackboneSpec("facebook/dinov2-base", 768, 448),
    "dinov2-small": BackboneSpec("facebook/dinov2-small", 384, 224),
}


def preprocess_rgb_batch(
    images: torch.Tensor,
    target_size: int,
) -> torch.Tensor:
    """Resize float images in [0, 1] of shape (B, 3, H, W) to
    (B, 3, target_size, target_size) and apply ImageNet normalization."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W); got {tuple(images.shape)}")
    if images.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"expected float tensor in [0,1]; got dtype={images.dtype}")

    resized = F.interpolate(
        images, size=(target_size, target_size),
        mode="bilinear", align_corners=False, antialias=True,
    )

    mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (resized - mean) / std


class PretrainedEncoder:
    """Thin wrapper over a frozen HuggingFace vision model.

    Call it like a function: encoder(images_in_01) -> (B, feature_dim).
    """

    def __init__(self, name: str = "dinov2-base", device: str | torch.device = "cpu"):
        if name not in _BACKBONES:
            raise ValueError(f"unknown backbone '{name}'; choose from {sorted(_BACKBONES)}")
        self.name = name
        self.spec = _BACKBONES[name]
        self.device = torch.device(device)

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(self.spec.hf_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        return self.spec.feature_dim

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) float in [0, 1] — returns (B, feature_dim).

        Uses mean of patch tokens (skipping CLS) rather than CLS alone.
        Patch-mean preserves spatial layout information that CLS collapses
        into a single global summary — critical for sub-mm precision tasks
        like surgical grasping where "gripper is 0.5mm left of needle"
        must be distinguishable from "gripper is aligned for grasp"."""
        images = images.to(self.device)
        preprocessed = preprocess_rgb_batch(images, self.spec.input_size)
        out = self.model(preprocessed)
        return out.last_hidden_state[:, 1:].mean(dim=1)  # avg patch tokens (skip CLS)

    def encode_numpy_frames(self, frames, batch_size: int = 32) -> torch.Tensor:
        """Convenience: take a numpy array of shape (N, H, W, 3) uint8 and
        encode it in batches. Returns (N, feature_dim) cpu tensor."""
        import numpy as np

        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(np.asarray(frames))
        frames = frames.permute(0, 3, 1, 2).float() / 255.0

        chunks: list[torch.Tensor] = []
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            chunks.append(self(batch).cpu())
        return torch.cat(chunks, dim=0)
