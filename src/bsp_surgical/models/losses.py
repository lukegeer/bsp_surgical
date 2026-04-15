import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, I)), averaged over batch, summed over latent dims."""
    kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
    return kl_per_sample.mean()


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target)


def forward_dynamics_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(z_pred, z_target.detach())


def inverse_dynamics_loss(
    a_pred: torch.Tensor,
    a_target: torch.Tensor,
    jaw_weight: float = 0.01,
) -> torch.Tensor:
    """Smooth-L1 on the first (action_dim - 1) continuous dims;
    BCE-with-logits on the final "jaw" dim."""
    continuous_pred, jaw_pred = a_pred[..., :-1], a_pred[..., -1]
    continuous_target, jaw_target = a_target[..., :-1], a_target[..., -1]

    continuous_loss = F.smooth_l1_loss(continuous_pred, continuous_target)
    jaw_loss = F.binary_cross_entropy_with_logits(jaw_pred, jaw_target)
    return continuous_loss + jaw_weight * jaw_loss


def subgoal_dual_supervision_loss(
    subgoal_mlp: nn.Module,
    *,
    z_start: torch.Tensor,
    z_quarter: torch.Tensor,
    z_mid: torch.Tensor,
    z_end: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """L = L_gt + L_pred.

    L_gt:   MSE( h(z_start, z_end_gt),       z_mid_gt )
    L_pred: MSE( h(z_start, h(z_start, z_end_gt)), z_quarter_gt )

    The second term feeds the model's own sg_1 prediction back in as
    z_target to bridge the train-test gap (at inference only
    predictions are available)."""
    sg1_from_gt = subgoal_mlp(z_start, z_end)
    loss_gt = F.mse_loss(sg1_from_gt, z_mid)

    sg1_pred = subgoal_mlp(z_start, z_end)
    sg2_from_pred = subgoal_mlp(z_start, sg1_pred)
    loss_pred = F.mse_loss(sg2_from_pred, z_quarter)

    total = loss_gt + loss_pred
    return total, {"gt": loss_gt, "pred": loss_pred}
