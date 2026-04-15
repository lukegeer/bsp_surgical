from dataclasses import dataclass

import torch

from bsp_surgical.models.vae import VAE
from bsp_surgical.models.dynamics import ForwardDynamics, InverseDynamics
from bsp_surgical.models.losses import (
    kl_divergence_loss,
    reconstruction_loss,
    forward_dynamics_loss,
    inverse_dynamics_loss,
)


@dataclass
class Phase2Models:
    vae: VAE
    forward: ForwardDynamics
    inverse: InverseDynamics

    def to(self, device: torch.device) -> "Phase2Models":
        self.vae.to(device)
        self.forward.to(device)
        self.inverse.to(device)
        return self

    def train(self) -> None:
        self.vae.train()
        self.forward.train()
        self.inverse.train()

    def eval(self) -> None:
        self.vae.eval()
        self.forward.eval()
        self.inverse.eval()


def compute_phase2_losses(
    models: Phase2Models,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    kl_weight: float = 0.5,
    jaw_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    img_t, action, img_next = batch

    recon_t, mu_t, logvar_t = models.vae(img_t)
    mu_next, logvar_next = models.vae.encode(img_next)
    z_t = models.vae.reparameterize(mu_t, logvar_t)
    z_next = models.vae.reparameterize(mu_next, logvar_next)

    recon = reconstruction_loss(recon_t, img_t)
    kl = kl_divergence_loss(mu_t, logvar_t)

    delta_z_pred = models.forward(z_t, action)
    z_next_pred = z_t + delta_z_pred
    forward = forward_dynamics_loss(z_next_pred, z_next)

    a_pred = models.inverse(z_t, z_next)
    inverse = inverse_dynamics_loss(a_pred, action, jaw_weight=jaw_weight)

    total = recon + kl_weight * kl + forward + inverse
    return {"recon": recon, "kl": kl, "forward": forward, "inverse": inverse, "total": total}


def train_step(
    models: Phase2Models,
    optimizer: torch.optim.Optimizer,
    batch,
    *,
    kl_weight: float = 0.5,
    jaw_weight: float = 0.01,
) -> dict[str, float]:
    models.train()
    optimizer.zero_grad(set_to_none=True)
    losses = compute_phase2_losses(models, batch, kl_weight=kl_weight, jaw_weight=jaw_weight)
    losses["total"].backward()
    optimizer.step()
    return {k: v.item() for k, v in losses.items()}
