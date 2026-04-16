import torch

from bsp_surgical.models.losses import (
    forward_dynamics_loss,
    inverse_dynamics_loss,
)


def test_forward_dynamics_loss_stops_gradient_on_target():
    z_pred = torch.randn(4, 16, requires_grad=True)
    z_target = torch.randn(4, 16, requires_grad=True)

    loss = forward_dynamics_loss(z_pred, z_target)
    loss.backward()

    assert z_pred.grad is not None
    assert z_target.grad is None  # stop-gradient on target


def test_inverse_dynamics_loss_is_zero_when_all_dims_match():
    a_pred = torch.zeros(4, 5)
    a_target = torch.zeros(4, 5)
    loss = inverse_dynamics_loss(a_pred, a_target, jaw_weight=0.01)
    assert loss.item() < 1e-6


def test_inverse_dynamics_loss_is_positive_for_mismatch():
    a_pred = torch.zeros(4, 5)
    a_target = torch.ones(4, 5)
    loss = inverse_dynamics_loss(a_pred, a_target)
    assert loss.item() > 0


def test_binary_jaw_path_uses_bce():
    """jaw_is_binary=True uses BCE-with-logits on last dim for tasks
    where the jaw label is a {0,1} probability."""
    a_pred = torch.zeros(4, 5)
    a_pred[:, -1] = 1e6  # sigmoid(1e6) ~= 1
    a_target = torch.zeros(4, 5)
    a_target[:, -1] = 1.0

    loss = inverse_dynamics_loss(a_pred, a_target, jaw_weight=0.01, jaw_is_binary=True)
    assert loss.item() < 1e-3
