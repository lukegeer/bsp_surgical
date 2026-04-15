import torch
import torch.nn as nn


class SubgoalGenerator(nn.Module):
    """h(z_now, z_target) -> z_midpoint.

    Recursive bisection: apply this module once to get sg_1 between
    z_now and z_goal, twice (h(z_now, sg_1)) to get sg_2. Two inputs
    are concatenated, so order matters (h(a,b) != h(b,a))."""

    def __init__(self, latent_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_now: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_now, z_target], dim=-1))
