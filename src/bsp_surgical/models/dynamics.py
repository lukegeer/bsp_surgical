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
    """g(z_t, z_target[, p_t]) -> a_t or chunk of K actions.

    Optional proprioception concat: if proprio_dim > 0, the model also
    takes a (B, proprio_dim) state vector and concatenates it with the
    latent features. This is how we give the model the same
    privileged-ish signal (EE pose, joint angles) that SurRoL's
    state-based SOTA methods use — without abandoning the visual
    pipeline.

    Chunked output shape is (B, K, action_dim); single-step is
    (B, action_dim) for backward compatibility.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 5,
        hidden: int = 256,
        jaw_dim: int = 1,
        chunk_size: int = 1,
        proprio_dim: int = 0,
    ):
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if proprio_dim < 0:
            raise ValueError(f"proprio_dim must be >= 0, got {proprio_dim}")
        self.action_dim = action_dim
        self.jaw_dim = jaw_dim
        self.chunk_size = chunk_size
        self.proprio_dim = proprio_dim
        in_dim = 2 * latent_dim + proprio_dim
        self.net = _mlp(in_dim, hidden, chunk_size * action_dim, depth=3)

    def forward(
        self,
        z_t: torch.Tensor,
        z_next: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [z_t, z_next]
        if self.proprio_dim > 0:
            if proprio is None:
                raise ValueError(
                    f"InverseDynamics was built with proprio_dim={self.proprio_dim} "
                    "but no proprio tensor was passed."
                )
            parts.append(proprio)
        out = self.net(torch.cat(parts, dim=-1))
        if self.chunk_size == 1:
            return out
        return out.view(-1, self.chunk_size, self.action_dim)
