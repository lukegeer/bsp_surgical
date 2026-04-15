"""Behavioral cloning baseline: single-image -> action policy.

This is the "no planning, no world model" baseline — directly map the
current observation to the oracle's action. Useful as a lower-bound
comparison against any planner that leverages a goal image."""
import math

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(self, resolution: int = 128, action_dim: int = 5):
        super().__init__()
        if resolution not in (64, 128):
            raise ValueError(f"resolution must be 64 or 128, got {resolution}")
        depth = int(math.log2(resolution)) - 2
        channels = [3, 32, 64, 128, 256, 256][: depth + 1]
        layers: list[nn.Module] = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1), nn.ReLU(inplace=True)]
        self.backbone = nn.Sequential(*layers)
        flat = channels[-1] * 4 * 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
