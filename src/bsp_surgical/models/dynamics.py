import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 3) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
    for _ in range(depth - 2):
        layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class ForwardDynamics(nn.Module):
    """f(z_t, a_t) -> delta_z. Residual prediction: z_{t+1} = z_t + delta_z."""

    def __init__(self, latent_dim: int = 128, action_dim: int = 5, hidden: int = 256):
        super().__init__()
        self.net = _mlp(latent_dim + action_dim, hidden, latent_dim, depth=3)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))

    def predict_next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.forward(z, a)


class InverseDynamics(nn.Module):
    """g(z_t, z_{t+1}) -> a_t."""

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 5,
        hidden: int = 256,
        jaw_dim: int = 1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.jaw_dim = jaw_dim
        self.net = _mlp(2 * latent_dim, hidden, action_dim, depth=3)

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, z_next], dim=-1))
