"""Encoder that combines frozen DINOv2 CLS features (from RGB) with a
small trainable CNN over segmentation + depth channels.

Rationale: DINOv2 alone at 128x128 can't resolve sub-mm surgical detail
(patch size > needle size). PyBullet segmentation + depth give
pixel-precise object positions at zero inference cost. Concatenating the
two feature streams gives the downstream MLP both semantic and precise
spatial information.

Output: 1024-d feature (768 DINOv2 CLS + 256 seg/depth CNN)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bsp_surgical.models.perception import PretrainedEncoder, preprocess_rgb_batch


class _SegDepthCNN(nn.Module):
    """Small CNN over one-hot segmentation channels + depth, 128x128 input."""

    def __init__(self, num_seg_channels: int = 8, out_dim: int = 256):
        super().__init__()
        in_ch = num_seg_channels + 1  # +1 for depth
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),  # 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),     # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),    # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),   # 8x8
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),                   # 4x4
            nn.Flatten(),                                    # 128*16 = 2048
            nn.Linear(128 * 16, out_dim),
        )

    def forward(self, seg_onehot: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # seg_onehot: (B, num_seg_channels, H, W); depth: (B, 1, H, W)
        x = torch.cat([seg_onehot, depth], dim=1)
        return self.net(x)


def seg_to_onehot(seg_map: np.ndarray, num_channels: int = 8) -> np.ndarray:
    """Dense-rank segmentation IDs and one-hot to at most num_channels.

    seg_map: (H, W) int32 from PyBullet.
    Returns: (num_channels, H, W) float32.

    PyBullet encodes body_id in the lower 24 bits of each pixel. IDs
    are dense-ranked by frequency so the most common surfaces get
    stable channel assignments across frames (channel 0 is most common,
    usually background). Extra IDs beyond num_channels are dropped.
    """
    body_ids = (seg_map & 0xFFFFFF).astype(np.int32)
    ids, counts = np.unique(body_ids, return_counts=True)
    order = np.argsort(-counts)
    top_ids = ids[order][:num_channels]
    out = np.zeros((num_channels, *seg_map.shape), dtype=np.float32)
    for ch, bid in enumerate(top_ids):
        out[ch] = (body_ids == bid).astype(np.float32)
    return out


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """PyBullet depth is in [0, 1] already (near/far normalized). Return
    shape (1, H, W) float32."""
    d = depth_map.astype(np.float32)
    return d[None, :, :]


class RGBSegDepthEncoder:
    """Wraps frozen DINOv2 + trainable seg/depth CNN. Call site looks
    like PretrainedEncoder for consistency."""

    def __init__(
        self,
        dinov2_name: str = "dinov2-base",
        num_seg_channels: int = 8,
        aux_dim: int = 256,
        device: str | torch.device = "mps",
    ):
        self.device = torch.device(device)
        self.dinov2 = PretrainedEncoder(dinov2_name, device)
        self.cnn = _SegDepthCNN(num_seg_channels=num_seg_channels, out_dim=aux_dim).to(self.device)
        self.num_seg_channels = num_seg_channels
        self.aux_dim = aux_dim

    @property
    def feature_dim(self) -> int:
        return self.dinov2.feature_dim + self.aux_dim  # e.g. 768 + 256 = 1024

    def encode(
        self,
        rgb: torch.Tensor,
        seg_onehot: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """rgb: (B, 3, H, W) float [0,1].
        seg_onehot: (B, K, H, W) float.
        depth: (B, 1, H, W) float."""
        rgb = rgb.to(self.device)
        seg_onehot = seg_onehot.to(self.device)
        depth = depth.to(self.device)
        dino_feat = self.dinov2(rgb)  # (B, 768)
        aux_feat = self.cnn(seg_onehot, depth)  # (B, 256)
        return torch.cat([dino_feat, aux_feat], dim=-1)
