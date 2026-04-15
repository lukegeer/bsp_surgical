import numpy as np
import torch

from bsp_surgical.data.trajectory import Trajectory
from bsp_surgical.data.io import save_trajectory
from bsp_surgical.training.dataset import TransitionDataset


def _write_fake_trajectory(path, *, episode_id, num_transitions):
    rng = np.random.default_rng(episode_id)
    images = rng.integers(0, 256, size=(num_transitions + 1, 128, 128, 3), dtype=np.uint8)
    actions = rng.standard_normal((num_transitions, 5)).astype(np.float32)
    traj = Trajectory(
        images=images, actions=actions,
        success=True, task_name="NeedleReach", episode_id=episode_id,
    )
    save_trajectory(traj, path)
    return images, actions


def test_dataset_length_equals_total_transitions(tmp_path):
    _write_fake_trajectory(tmp_path / "ep_0.npz", episode_id=0, num_transitions=5)
    _write_fake_trajectory(tmp_path / "ep_1.npz", episode_id=1, num_transitions=7)

    ds = TransitionDataset(tmp_path)

    assert len(ds) == 5 + 7


def test_dataset_item_returns_img_action_next_img_as_tensors(tmp_path):
    _write_fake_trajectory(tmp_path / "ep_0.npz", episode_id=0, num_transitions=3)

    ds = TransitionDataset(tmp_path)
    img_t, action, img_next = ds[0]

    assert isinstance(img_t, torch.Tensor)
    assert img_t.shape == (3, 128, 128)
    assert img_t.dtype == torch.float32
    assert img_t.min() >= 0.0 and img_t.max() <= 1.0

    assert action.shape == (5,)
    assert action.dtype == torch.float32

    assert img_next.shape == (3, 128, 128)


def test_dataset_preserves_transition_identity(tmp_path):
    images, actions = _write_fake_trajectory(
        tmp_path / "ep_0.npz", episode_id=0, num_transitions=4
    )

    ds = TransitionDataset(tmp_path)
    img_t, action, img_next = ds[2]

    expected_img_t = torch.from_numpy(images[2]).permute(2, 0, 1).float() / 255.0
    expected_img_next = torch.from_numpy(images[3]).permute(2, 0, 1).float() / 255.0

    assert torch.allclose(img_t, expected_img_t)
    assert torch.allclose(img_next, expected_img_next)
    assert torch.allclose(action, torch.from_numpy(actions[2]))


def test_dataset_crosses_episode_boundaries(tmp_path):
    _write_fake_trajectory(tmp_path / "ep_0.npz", episode_id=0, num_transitions=3)
    _write_fake_trajectory(tmp_path / "ep_1.npz", episode_id=1, num_transitions=2)

    ds = TransitionDataset(tmp_path)

    # index 3 should be the first transition of episode 1, not out of bounds
    img_t, action, img_next = ds[3]
    assert img_t.shape == (3, 128, 128)

    # index 4 is the last transition of episode 1
    ds[4]
    # index 5 is out of range
    try:
        ds[5]
        raise AssertionError("expected IndexError")
    except IndexError:
        pass
