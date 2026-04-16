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
    """g(z_t, z_target) -> a_t, or with chunk_size > 1 a chunk of K actions.

    When chunk_size=1 (default), the model predicts a single action — the
    classic inverse dynamics. When chunk_size=K>1, the model predicts K
    actions in one forward pass (Seer-style action chunking). Chunked
    output shape is (B, K, action_dim); single-step shape is (B, action_dim)
    for backward compatibility.

    Action chunking helps on tasks with discrete phase transitions
    (grasp, insertion) because the model can schedule the whole sequence
    as a unit instead of deciding action-by-action from a memoryless state.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 5,
        hidden: int = 256,
        jaw_dim: int = 1,
        chunk_size: int = 1,
    ):
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.action_dim = action_dim
        self.jaw_dim = jaw_dim
        self.chunk_size = chunk_size
        self.net = _mlp(2 * latent_dim, hidden, chunk_size * action_dim, depth=3)

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([z_t, z_next], dim=-1))
        if self.chunk_size == 1:
            return out
        return out.view(-1, self.chunk_size, self.action_dim)
