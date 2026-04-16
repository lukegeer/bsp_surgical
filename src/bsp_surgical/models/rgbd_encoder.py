"""Small CNN over (RGB + seg one-hot + depth) that preserves spatial
information. No DINOv2 — the segmentation masks already provide
pixel-perfect semantic localization, and depth gives 3D structure."""
import numpy as np
import torch
import torch.nn as nn


def seg_to_onehot(seg_map: np.ndarray, num_channels: int = 8) -> np.ndarray:
    """Dense-rank segmentation IDs by frequency and one-hot encode to
    at most num_channels. seg_map: (H, W) int32 from PyBullet (lower 24
    bits = body id). Most common surface gets channel 0 (usually
    background). Extra IDs beyond num_channels are dropped."""
    body_ids = (seg_map & 0xFFFFFF).astype(np.int32)
    ids, counts = np.unique(body_ids, return_counts=True)
    order = np.argsort(-counts)
    top_ids = ids[order][:num_channels]
    out = np.zeros((num_channels, *seg_map.shape), dtype=np.float32)
    for ch, bid in enumerate(top_ids):
        out[ch] = (body_ids == bid).astype(np.float32)
    return out


class RGBDSegEncoder(nn.Module):
    """Input: (B, 3+num_seg+1, H, W). Output: (B, feature_dim) flat feature."""

    def __init__(
        self,
        num_seg_channels: int = 8,
        resolution: int = 128,
        feature_dim: int = 1024,
    ):
        super().__init__()
        in_ch = 3 + num_seg_channels + 1  # RGB + seg one-hot + depth
        # Strided-conv backbone, keeps 1/16 spatial at the end.
        # 128 -> 64 -> 32 -> 16 -> 8
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        spatial = resolution // 16  # 8 for 128
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * spatial * spatial, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.feature_dim = feature_dim
        self.num_seg_channels = num_seg_channels

    def forward(self, rgb: torch.Tensor, seg_onehot: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, seg_onehot, depth], dim=1)
        return self.head(self.backbone(x))
