import torch

from bsp_surgical.models.vae import VAE
from bsp_surgical.models.dynamics import ForwardDynamics, InverseDynamics
from bsp_surgical.training.phase2 import Phase2Models, compute_phase2_losses, train_step


def _make_models(resolution=64, latent_dim=16, action_dim=5):
    return Phase2Models(
        vae=VAE(latent_dim=latent_dim, resolution=resolution),
        forward=ForwardDynamics(latent_dim=latent_dim, action_dim=action_dim),
        inverse=InverseDynamics(latent_dim=latent_dim, action_dim=action_dim),
    )


def _fake_batch(batch_size=4, resolution=64, action_dim=5):
    img_t = torch.rand(batch_size, 3, resolution, resolution)
    action = torch.randn(batch_size, action_dim).clamp_(-1, 1)
    # make jaw binary for BCE
    action[:, -1] = (action[:, -1] > 0).float()
    img_next = torch.rand(batch_size, 3, resolution, resolution)
    return img_t, action, img_next


def test_compute_phase2_losses_returns_all_four_scalar_losses():
    models = _make_models()
    batch = _fake_batch()

    losses = compute_phase2_losses(models, batch, kl_weight=0.5)

    assert set(losses.keys()) >= {"recon", "kl", "forward", "inverse", "total"}
    for k, v in losses.items():
        assert v.ndim == 0, k
        assert torch.isfinite(v), k


def test_train_step_reduces_total_loss_on_learnable_batch():
    """Trivial dynamics: img stays the same, zero action. All four losses
    should collapse substantially with a handful of steps — this is a
    backprop-wiring sanity check, not a convergence benchmark."""
    torch.manual_seed(0)
    models = _make_models()
    params = (
        list(models.vae.parameters())
        + list(models.forward.parameters())
        + list(models.inverse.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    img = torch.rand(4, 3, 64, 64)
    action = torch.zeros(4, 5)
    batch = (img, action, img)

    initial = compute_phase2_losses(models, batch, kl_weight=0.5)["total"].item()
    for _ in range(100):
        train_step(models, optimizer, batch, kl_weight=0.5)
    final = compute_phase2_losses(models, batch, kl_weight=0.5)["total"].item()

    # 25% is a backprop-wiring sanity threshold, not a convergence benchmark.
    assert final < initial * 0.75, f"expected >25% drop; got {initial:.3f} -> {final:.3f}"


def test_train_step_leaves_all_model_params_with_grads():
    models = _make_models()
    params = (
        list(models.vae.parameters())
        + list(models.forward.parameters())
        + list(models.inverse.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    batch = _fake_batch()

    train_step(models, optimizer, batch, kl_weight=0.5)

    for p in params:
        assert p.grad is not None
