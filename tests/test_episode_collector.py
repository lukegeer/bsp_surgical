import numpy as np

from bsp_surgical.data.collector import collect_episode


class FakeEnv:
    """Minimal gym-like env whose render output is deterministic per step."""

    def __init__(self, max_steps_until_done: int = 5, render_size: int = 64):
        self._t = 0
        self._max = max_steps_until_done
        self._render_size = render_size

    def reset(self):
        self._t = 0
        return {"observation": np.zeros(3, dtype=np.float32)}

    def step(self, action):
        self._t += 1
        obs = {"observation": np.full(3, self._t, dtype=np.float32)}
        reward = 0.0
        done = self._t >= self._max
        info = {"is_success": done}
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        frame = np.full(
            (self._render_size, self._render_size, 3), self._t % 256, dtype=np.uint8
        )
        return frame


def _constant_oracle(obs):
    return np.array([0.1, 0.2, 0.3, 0.0, 0.0], dtype=np.float32)


def test_collector_produces_T_plus_1_images_and_T_actions():
    env = FakeEnv(max_steps_until_done=5)

    traj = collect_episode(
        env,
        _constant_oracle,
        max_steps=20,
        resolution=128,
        task_name="NeedleReach",
        episode_id=3,
    )

    assert traj.num_transitions == 5
    assert traj.images.shape == (6, 128, 128, 3)
    assert traj.images.dtype == np.uint8
    assert traj.actions.shape == (5, 5)
    assert traj.actions.dtype == np.float32
    assert traj.episode_id == 3
    assert traj.task_name == "NeedleReach"
    assert traj.success is True


def test_collector_respects_max_steps_cap():
    env = FakeEnv(max_steps_until_done=100)

    traj = collect_episode(
        env,
        _constant_oracle,
        max_steps=4,
        resolution=64,
        task_name="NeedleReach",
        episode_id=0,
    )

    assert traj.num_transitions == 4
    assert traj.success is False


class SuccessWithoutDoneEnv:
    """Goal-conditioned env that reports is_success in info but never sets done."""

    def __init__(self, success_at_step: int):
        self._t = 0
        self._success_at = success_at_step

    def reset(self):
        self._t = 0
        return {"observation": np.zeros(3, dtype=np.float32)}

    def step(self, action):
        self._t += 1
        obs = {"observation": np.full(3, self._t, dtype=np.float32)}
        return obs, 0.0, False, {"is_success": self._t >= self._success_at}

    def render(self, mode="rgb_array"):
        return np.zeros((32, 32, 3), dtype=np.uint8)


def test_collector_stops_on_is_success_even_without_done():
    env = SuccessWithoutDoneEnv(success_at_step=3)

    traj = collect_episode(
        env,
        _constant_oracle,
        max_steps=20,
        resolution=32,
        task_name="NeedleReach",
        episode_id=0,
    )

    assert traj.num_transitions == 3
    assert traj.success is True


def test_collector_preserves_oracle_actions():
    env = FakeEnv(max_steps_until_done=3)
    captured = []

    def recording_oracle(obs):
        a = np.array([len(captured) * 0.1] * 5, dtype=np.float32)
        captured.append(a)
        return a

    traj = collect_episode(
        env,
        recording_oracle,
        max_steps=20,
        resolution=64,
        task_name="NeedleReach",
        episode_id=0,
    )

    assert np.allclose(traj.actions, np.stack(captured))
