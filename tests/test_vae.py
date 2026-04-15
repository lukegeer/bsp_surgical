import torch

from bsp_surgical.models.vae import VAE


def test_vae_forward_returns_recon_mu_logvar_with_correct_shapes():
    batch = 4
    vae = VAE(latent_dim=128, resolution=128)
    x = torch.randn(batch, 3, 128, 128)

    recon, mu, logvar = vae(x)

    assert recon.shape == (batch, 3, 128, 128)
    assert mu.shape == (batch, 128)
    assert logvar.shape == (batch, 128)


def test_vae_encode_is_deterministic():
    vae = VAE(latent_dim=128, resolution=128).eval()
    x = torch.randn(2, 3, 128, 128)

    mu1, logvar1 = vae.encode(x)
    mu2, logvar2 = vae.encode(x)

    assert torch.allclose(mu1, mu2)
    assert torch.allclose(logvar1, logvar2)


def test_vae_reparameterize_returns_mu_in_eval_mode():
    vae = VAE(latent_dim=128, resolution=128).eval()
    mu = torch.randn(3, 128)
    logvar = torch.randn(3, 128)

    z = vae.reparameterize(mu, logvar)

    assert torch.allclose(z, mu)


def test_vae_reparameterize_is_stochastic_in_train_mode():
    torch.manual_seed(0)
    vae = VAE(latent_dim=128, resolution=128).train()
    mu = torch.zeros(3, 128)
    logvar = torch.zeros(3, 128)  # sigma=1

    z1 = vae.reparameterize(mu, logvar)
    z2 = vae.reparameterize(mu, logvar)

    assert not torch.allclose(z1, z2)


def test_vae_supports_64x64_resolution():
    vae = VAE(latent_dim=16, resolution=64)
    x = torch.randn(2, 3, 64, 64)

    recon, mu, logvar = vae(x)

    assert recon.shape == (2, 3, 64, 64)
    assert mu.shape == (2, 16)
    assert logvar.shape == (2, 16)
