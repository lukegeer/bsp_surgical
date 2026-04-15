import numpy as np
import pytest

from bsp_surgical.data.trajectory import Trajectory


def _make_trajectory(num_transitions: int = 3, resolution: int = 128, action_dim: int = 5) -> Trajectory:
    images = np.zeros((num_transitions + 1, resolution, resolution, 3), dtype=np.uint8)
    images[-1] = 255  # distinctive goal image
    actions = np.zeros((num_transitions, action_dim), dtype=np.float32)
    return Trajectory(
        images=images,
        actions=actions,
        success=True,
        task_name="NeedleReach",
        episode_id=0,
    )


def test_trajectory_exposes_goal_image_and_transition_count():
    traj = _make_trajectory(num_transitions=3)

    assert traj.num_transitions == 3
    assert traj.goal_image.shape == (128, 128, 3)
    assert np.all(traj.goal_image == 255)


def test_trajectory_rejects_mismatched_image_and_action_lengths():
    images = np.zeros((4, 128, 128, 3), dtype=np.uint8)
    actions = np.zeros((5, 5), dtype=np.float32)  # mismatched: needs 3, not 5

    with pytest.raises(ValueError, match="images"):
        Trajectory(
            images=images,
            actions=actions,
            success=True,
            task_name="NeedleReach",
            episode_id=0,
        )


def test_trajectory_rejects_wrong_image_dtype():
    images = np.zeros((4, 128, 128, 3), dtype=np.float32)  # must be uint8
    actions = np.zeros((3, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="uint8"):
        Trajectory(
            images=images,
            actions=actions,
            success=True,
            task_name="NeedleReach",
            episode_id=0,
        )


def test_trajectory_rejects_empty_episode():
    images = np.zeros((1, 128, 128, 3), dtype=np.uint8)
    actions = np.zeros((0, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one transition"):
        Trajectory(
            images=images,
            actions=actions,
            success=True,
            task_name="NeedleReach",
            episode_id=0,
        )
