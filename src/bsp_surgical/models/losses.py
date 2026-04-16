import torch
import torch.nn.functional as F


def forward_dynamics_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(z_pred, z_target.detach())


def inverse_dynamics_loss(
    a_pred: torch.Tensor,
    a_target: torch.Tensor,
    jaw_weight: float = 0.01,
    jaw_is_binary: bool = False,
) -> torch.Tensor:
    """Smooth-L1 on all action dims by default.

    If jaw_is_binary=True, apply BCE-with-logits on the last dim instead
    (only valid when that dim is actually a probability / {0,1} target).
    SurRoL oracles for NeedlePick/NeedleRegrasp output the jaw as a
    continuous value in [-0.5, 0.5], NOT a probability — feeding that
    into BCE-with-logits gives negative divergent loss and destroys
    the inverse model's weights. Default stays Smooth-L1 on all dims."""
    if not jaw_is_binary:
        return F.smooth_l1_loss(a_pred, a_target)

    # .contiguous() is required for MPS; slice views can have non-standard strides.
    continuous_pred = a_pred[..., :-1].contiguous()
    jaw_pred = a_pred[..., -1].contiguous()
    continuous_target = a_target[..., :-1].contiguous()
    jaw_target = a_target[..., -1].contiguous()

    continuous_loss = F.smooth_l1_loss(continuous_pred, continuous_target)
    jaw_loss = F.binary_cross_entropy_with_logits(jaw_pred, jaw_target)
    return continuous_loss + jaw_weight * jaw_loss


