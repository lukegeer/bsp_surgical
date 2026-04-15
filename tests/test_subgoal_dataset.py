import numpy as np
import torch

from bsp_surgical.data.trajectory import Trajectory
from bsp_surgical.data.io import save_trajectory
from bsp_surgical.models.vae import VAE
from bsp_surgical.training.phase3 import SubgoalDataset


def _write_traj(path, *, episode_id, num_transitions, resolution=64):
    rng = np.random.default_rng(episode_id)
    images = rng.integers(0, 256, size=(num_transitions + 1, resolution, resolution, 3), dtype=np.uint8)
    actions = rng.standard_normal((num_transitions, 5)).astype(np.float32)
    save_trajectory(
        Trajectory(images=images, actions=actions, success=True,
                   task_name="NeedleReach", episode_id=episode_id),
        path,
    )


def test_subgoal_dataset_yields_four_latents_per_episode(tmp_path):
    _write_traj(tmp_path / "ep_0.npz", episode_id=0, num_transitions=8)
    _write_traj(tmp_path / "ep_1.npz", episode_id=1, num_transitions=12)

    vae = VAE(latent_dim=16, resolution=64).eval()
    ds = SubgoalDataset(tmp_path, vae=vae, device="cpu")

    assert len(ds) == 2
    z_start, z_quarter, z_mid, z_end = ds[0]
    for z in (z_start, z_quarter, z_mid, z_end):
        assert z.shape == (16,)
        assert z.dtype == torch.float32


def test_subgoal_dataset_skips_episodes_too_short_for_quarter_point(tmp_path):
    """With T=3, quarter=0 and mid=1 and end=3; marginal but usable.
    With T=2, quarter=0, mid=1, end=2 — still OK. Skip only T<3."""
    _write_traj(tmp_path / "ep_short.npz", episode_id=0, num_transitions=2)
    _write_traj(tmp_path / "ep_ok.npz", episode_id=1, num_transitions=10)

    vae = VAE(latent_dim=16, resolution=64).eval()
    ds = SubgoalDataset(tmp_path, vae=vae, device="cpu", min_transitions=3)

    assert len(ds) == 1


def test_subgoal_dataset_picks_evenly_spaced_indices(tmp_path):
    """For T=12: start=0, quarter=3, mid=6, end=12 (or close)."""
    _write_traj(tmp_path / "ep_0.npz", episode_id=0, num_transitions=12, resolution=64)

    vae = VAE(latent_dim=16, resolution=64).eval()
    ds = SubgoalDataset(tmp_path, vae=vae, device="cpu")

    indices = ds.episode_indices[0]
    assert indices == (0, 3, 6, 12)
