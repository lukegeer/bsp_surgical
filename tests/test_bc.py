import torch

from bsp_surgical.models.bc import BCPolicy


def test_bc_policy_output_shape():
    policy = BCPolicy(resolution=64, action_dim=5)
    x = torch.randn(4, 3, 64, 64)

    a = policy(x)

    assert a.shape == (4, 5)


def test_bc_policy_supports_128_resolution():
    policy = BCPolicy(resolution=128, action_dim=5)
    x = torch.randn(2, 3, 128, 128)

    a = policy(x)

    assert a.shape == (2, 5)
