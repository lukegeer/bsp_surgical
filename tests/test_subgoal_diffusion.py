import torch

from bsp_surgical.models.subgoal_diffusion import SubgoalDiffusion
from bsp_surgical.models.rgbd_encoder import RGBDSegEncoder


def test_rgbd_encoder_output_shape():
    enc = RGBDSegEncoder(num_seg_channels=8, resolution=128, feature_dim=1024)
    rgb = torch.rand(4, 3, 128, 128)
    seg = torch.rand(4, 8, 128, 128)
    depth = torch.rand(4, 1, 128, 128)
    out = enc(rgb, seg, depth)
    assert out.shape == (4, 1024)


def test_subgoal_diffusion_training_loss_scalar():
    sd = SubgoalDiffusion(latent_dim=64, hidden=128, num_timesteps=100)
    z_mid = torch.randn(4, 64)
    z_start = torch.randn(4, 64)
    z_end = torch.randn(4, 64)
    loss = sd.training_loss(z_mid, z_start, z_end)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_subgoal_diffusion_sample_shape():
    sd = SubgoalDiffusion(latent_dim=64, hidden=128, num_timesteps=100).eval()
    z_start = torch.randn(3, 64)
    z_end = torch.randn(3, 64)
    sg = sd.sample(z_start, z_end, num_inference_steps=10)
    assert sg.shape == (3, 64)


def test_backward_bisect_returns_near_to_far_waypoints():
    sd = SubgoalDiffusion(latent_dim=64, hidden=128, num_timesteps=100).eval()
    z_now = torch.randn(1, 64)
    z_goal = torch.randn(1, 64)
    waypoints = sd.backward_bisect(z_now, z_goal, num_subgoals=2, num_inference_steps=10)
    assert len(waypoints) == 3
    assert waypoints[-1] is z_goal


def test_training_step_actually_updates_weights():
    torch.manual_seed(0)
    sd = SubgoalDiffusion(latent_dim=32, hidden=128, num_timesteps=50)
    opt = torch.optim.AdamW(sd.parameters(), lr=1e-3)
    z = torch.randn(8, 32)
    initial = [p.detach().clone() for p in sd.denoiser.net.parameters()]
    for _ in range(20):
        loss = sd.training_loss(z, torch.randn(8, 32), torch.randn(8, 32))
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = list(sd.denoiser.net.parameters())
    # At least one param should have moved meaningfully
    assert any((f - i).norm().item() > 0.1 for i, f in zip(initial, final))
