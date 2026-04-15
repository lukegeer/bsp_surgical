from typing import Callable

import cv2
import numpy as np

from bsp_surgical.data.trajectory import Trajectory


def _resize_rgb(frame: np.ndarray, resolution: int) -> np.ndarray:
    if frame.shape[0] == resolution and frame.shape[1] == resolution:
        return frame.astype(np.uint8, copy=False)
    resized = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8, copy=False)


def collect_episode(
    env,
    get_oracle_action: Callable,
    *,
    max_steps: int,
    resolution: int,
    task_name: str,
    episode_id: int,
) -> Trajectory:
    obs = env.reset()

    images: list[np.ndarray] = [_resize_rgb(env.render("rgb_array"), resolution)]
    actions: list[np.ndarray] = []
    success = False

    for _ in range(max_steps):
        action = np.asarray(get_oracle_action(obs), dtype=np.float32)
        obs, _reward, done, info = env.step(action)
        actions.append(action)
        images.append(_resize_rgb(env.render("rgb_array"), resolution))
        is_success = bool(info.get("is_success", False))
        if is_success or done:
            success = is_success
            break

    return Trajectory(
        images=np.stack(images),
        actions=np.stack(actions),
        success=success,
        task_name=task_name,
        episode_id=episode_id,
    )
