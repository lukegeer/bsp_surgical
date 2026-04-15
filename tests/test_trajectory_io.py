import numpy as np

from bsp_surgical.data.trajectory import Trajectory
from bsp_surgical.data.io import save_trajectory, load_trajectory


def _make_trajectory(episode_id: int = 0, success: bool = True) -> Trajectory:
    rng = np.random.default_rng(episode_id)
    images = rng.integers(0, 256, size=(4, 128, 128, 3), dtype=np.uint8)
    actions = rng.standard_normal((3, 5)).astype(np.float32)
    return Trajectory(
        images=images,
        actions=actions,
        success=success,
        task_name="NeedleReach",
        episode_id=episode_id,
    )


def test_trajectory_round_trip_preserves_all_fields(tmp_path):
    traj = _make_trajectory(episode_id=7, success=True)
    path = tmp_path / "ep_0007.npz"

    save_trajectory(traj, path)
    loaded = load_trajectory(path)

    assert np.array_equal(loaded.images, traj.images)
    assert np.array_equal(loaded.actions, traj.actions)
    assert loaded.success is True
    assert loaded.task_name == "NeedleReach"
    assert loaded.episode_id == 7


def test_trajectory_round_trip_preserves_failure_flag(tmp_path):
    traj = _make_trajectory(episode_id=1, success=False)
    path = tmp_path / "ep_0001.npz"

    save_trajectory(traj, path)
    loaded = load_trajectory(path)

    assert loaded.success is False


def test_save_creates_parent_directories(tmp_path):
    traj = _make_trajectory()
    path = tmp_path / "nested" / "subdir" / "ep.npz"

    save_trajectory(traj, path)

    assert path.exists()
