import torch

from bsp_surgical.models.dynamics import ForwardDynamics, InverseDynamics


def test_forward_dynamics_shapes():
    fwd = ForwardDynamics(latent_dim=128, action_dim=5)
    z = torch.randn(4, 128)
    a = torch.randn(4, 5)

    delta_z = fwd(z, a)

    assert delta_z.shape == (4, 128)


def test_forward_dynamics_predict_next_latent():
    fwd = ForwardDynamics(latent_dim=128, action_dim=5)
    z = torch.randn(2, 128)
    a = torch.randn(2, 5)

    z_next = fwd.predict_next(z, a)

    # residual: z_{t+1} = z_t + delta_z
    assert z_next.shape == (2, 128)
    assert torch.allclose(z_next, z + fwd(z, a))


def test_inverse_dynamics_shapes():
    inv = InverseDynamics(latent_dim=128, action_dim=5)
    z_t = torch.randn(4, 128)
    z_next = torch.randn(4, 128)

    action = inv(z_t, z_next)

    assert action.shape == (4, 5)


def test_inverse_dynamics_jaw_dim_is_separable():
    """Last action dim is gripper jaw (binary); model should expose it."""
    inv = InverseDynamics(latent_dim=128, action_dim=5, jaw_dim=1)
    z_t = torch.randn(4, 128)
    z_next = torch.randn(4, 128)

    action = inv(z_t, z_next)
    continuous, jaw = action[:, :4], action[:, 4:]

    assert continuous.shape == (4, 4)
    assert jaw.shape == (4, 1)
