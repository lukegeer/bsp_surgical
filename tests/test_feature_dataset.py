import numpy as np
import torch

from bsp_surgical.data.trajectory import Trajectory
from bsp_surgical.data.io import save_trajectory
from bsp_surgical.training.feature_dataset import (
    FeatureTransitionDataset,
    FeatureSubgoalDataset,
)


def _write_pair(tmp_path, episode_id, num_transitions, feature_dim=8):
    rng = np.random.default_rng(episode_id)
    images = rng.integers(0, 256, size=(num_transitions + 1, 32, 32, 3), dtype=np.uint8)
    actions = rng.standard_normal((num_transitions, 5)).astype(np.float32)
    traj = Trajectory(
        images=images, actions=actions, success=True,
        task_name="NeedleReach", episode_id=episode_id,
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    save_trajectory(traj, raw_dir / f"ep_{episode_id:05d}.npz")

    feat_dir = tmp_path / "feat"
    feat_dir.mkdir(exist_ok=True)
    features = rng.standard_normal((num_transitions + 1, feature_dim)).astype(np.float32)
    np.savez_compressed(feat_dir / f"ep_{episode_id:05d}.npz", features=features)

    return raw_dir, feat_dir, features, actions


def test_feature_transition_dataset_length_and_shapes(tmp_path):
    raw, feat, _, _ = _write_pair(tmp_path, 0, num_transitions=4, feature_dim=8)
    _write_pair(tmp_path, 1, num_transitions=6, feature_dim=8)

    ds = FeatureTransitionDataset(raw, feat)

    assert len(ds) == 10
    z_t, a, z_next = ds[0]
    assert z_t.shape == (8,)
    assert a.shape == (5,)
    assert z_next.shape == (8,)
    assert z_t.dtype == torch.float32


def test_feature_transition_dataset_preserves_transition_identity(tmp_path):
    raw, feat, features, actions = _write_pair(tmp_path, 3, num_transitions=5, feature_dim=8)

    ds = FeatureTransitionDataset(raw, feat)
    z_t, a, z_next = ds[2]

    assert np.allclose(z_t.numpy(), features[2])
    assert np.allclose(z_next.numpy(), features[3])
    assert np.allclose(a.numpy(), actions[2])


def test_feature_transition_dataset_chunk_size_returns_k_actions(tmp_path):
    raw, feat, _features, actions = _write_pair(tmp_path, 0, num_transitions=6, feature_dim=8)

    ds = FeatureTransitionDataset(raw, feat, chunk_size=3)
    z_t, chunk, z_next = ds[1]

    assert chunk.shape == (3, 5)
    # actions[1], actions[2], actions[3] should be the chunk
    import numpy as np
    assert np.allclose(chunk.numpy(), actions[1:4])


def test_feature_transition_dataset_chunk_pads_with_final_action_at_episode_end(tmp_path):
    raw, feat, _, actions = _write_pair(tmp_path, 0, num_transitions=3, feature_dim=8)

    ds = FeatureTransitionDataset(raw, feat, chunk_size=5)
    # Last possible offset=2; chunk of 5 from there needs 3 pads of actions[-1]
    z_t, chunk, _ = ds[2]

    import numpy as np
    assert chunk.shape == (5, 5)
    assert np.allclose(chunk[0].numpy(), actions[2])
    # padded tail should all be the final action
    assert np.allclose(chunk[1].numpy(), actions[-1])
    assert np.allclose(chunk[-1].numpy(), actions[-1])


def test_feature_transition_dataset_errors_if_features_missing(tmp_path):
    rng = np.random.default_rng(0)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    save_trajectory(
        Trajectory(
            images=rng.integers(0, 256, (3, 32, 32, 3), dtype=np.uint8),
            actions=rng.standard_normal((2, 5)).astype(np.float32),
            success=True, task_name="NeedleReach", episode_id=0,
        ),
        raw_dir / "ep_00000.npz",
    )
    empty_feat_dir = tmp_path / "feat"
    empty_feat_dir.mkdir()

    try:
        FeatureTransitionDataset(raw_dir, empty_feat_dir)
        raise AssertionError("should have raised")
    except FileNotFoundError:
        pass


def test_feature_subgoal_dataset_yields_four_latents(tmp_path):
    raw, feat, features, _ = _write_pair(tmp_path, 0, num_transitions=8, feature_dim=8)
    ds = FeatureSubgoalDataset(raw, feat, min_transitions=3)

    assert len(ds) == 1
    z_start, z_quarter, z_mid, z_end = ds[0]
    # indices should be 0, 2, 4, 8 for T=8
    assert np.allclose(z_start.numpy(), features[0])
    assert np.allclose(z_quarter.numpy(), features[2])
    assert np.allclose(z_mid.numpy(), features[4])
    assert np.allclose(z_end.numpy(), features[8])


def test_feature_subgoal_dataset_skips_short_episodes(tmp_path):
    _write_pair(tmp_path, 0, num_transitions=2, feature_dim=8)
    _write_pair(tmp_path, 1, num_transitions=10, feature_dim=8)
    raw = tmp_path / "raw"
    feat = tmp_path / "feat"

    ds = FeatureSubgoalDataset(raw, feat, min_transitions=3)

    assert len(ds) == 1
    assert ds.episode_indices[0] == (0, 2, 5, 10)
