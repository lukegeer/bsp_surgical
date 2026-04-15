"""Small decoder that maps a frozen-backbone feature vector back to a
128x128 RGB image. Purely for visualization: lets us 'see' subgoals
by decoding them, and debug the planner's trajectory through latent
space. Not used in the inference path.

Trained with pixel MSE against the original frames."""
import torch
import torch.nn as nn


def _up_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class FeatureDecoder(nn.Module):
    """feature (B, feature_dim) -> image (B, 3, 128, 128)."""

    def __init__(self, feature_dim: int = 768):
        super().__init__()
        # project to 512x4x4
        self.fc = nn.Linear(feature_dim, 512 * 4 * 4)
        self.up = nn.Sequential(
            _up_block(512, 256),  # 4 -> 8
            _up_block(256, 128),  # 8 -> 16
            _up_block(128, 64),   # 16 -> 32
            _up_block(64, 32),    # 32 -> 64
            _up_block(32, 16),    # 64 -> 128
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 512, 4, 4)
        return torch.sigmoid(self.up(h))
