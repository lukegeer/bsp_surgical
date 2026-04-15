import math

import torch

from bsp_surgical.models.losses import (
    kl_divergence_loss,
    reconstruction_loss,
    forward_dynamics_loss,
    inverse_dynamics_loss,
)


def test_kl_is_zero_for_standard_normal():
    mu = torch.zeros(4, 16)
    logvar = torch.zeros(4, 16)  # variance = 1

    loss = kl_divergence_loss(mu, logvar)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_kl_is_positive_for_offset_mean():
    mu = torch.ones(4, 16) * 2.0
    logvar = torch.zeros(4, 16)

    loss = kl_divergence_loss(mu, logvar)

    assert loss.item() > 0


def test_reconstruction_loss_zero_for_identical_images():
    x = torch.rand(4, 3, 32, 32)
    loss = reconstruction_loss(x, x)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_reconstruction_loss_positive_for_different_images():
    x = torch.rand(4, 3, 32, 32)
    x_hat = x + 0.5
    loss = reconstruction_loss(x_hat, x)
    assert loss.item() > 0


def test_forward_dynamics_loss_stops_gradient_on_target():
    z_pred = torch.randn(4, 16, requires_grad=True)
    z_target = torch.randn(4, 16, requires_grad=True)

    loss = forward_dynamics_loss(z_pred, z_target)
    loss.backward()

    assert z_pred.grad is not None
    assert z_target.grad is None  # stop-gradient on target


def test_inverse_dynamics_loss_splits_continuous_and_jaw():
    a_pred = torch.randn(4, 5)
    a_target = torch.randn(4, 5)
    # force jaw target into {0, 1}
    a_target_with_jaw = a_target.clone()
    a_target_with_jaw[:, -1] = (a_target[:, -1] > 0).float()

    loss = inverse_dynamics_loss(a_pred, a_target_with_jaw, jaw_weight=0.01)

    assert loss.ndim == 0  # scalar
    assert loss.item() >= 0


def test_inverse_dynamics_loss_is_zero_when_continuous_match_and_jaw_is_trivial():
    a_pred = torch.zeros(4, 5)
    a_pred[:, -1] = 1e6  # sigmoid(1e6) ~= 1
    a_target = torch.zeros(4, 5)
    a_target[:, -1] = 1.0

    loss = inverse_dynamics_loss(a_pred, a_target, jaw_weight=0.01)

    # Smooth-L1 on zeros is 0; BCE logits with huge +logit and target=1 -> ~0
    assert loss.item() < 1e-3
