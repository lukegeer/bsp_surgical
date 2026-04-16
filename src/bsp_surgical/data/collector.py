from typing import Callable

import numpy as np

from bsp_surgical.data.trajectory import Trajectory


def _crop_and_resize(
    frame: np.ndarray,
    resolution: int,
    crop_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Optional crop then resize. crop_box = (y1, y2, x1, x2) in raw pixel coords."""
    if crop_box is not None:
        y1, y2, x1, x2 = crop_box
        frame = frame[y1:y2, x1:x2]
    if frame.shape[0] == resolution and frame.shape[1] == resolution:
        return frame.astype(np.uint8, copy=False)
    import cv2

    resized = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8, copy=False)


def _proprio_from_obs(obs) -> np.ndarray | None:
    """Extract a 1-D state vector from a SurRoL goal-env obs dict.
    Returns None if obs is not a dict or lacks 'observation'."""
    if isinstance(obs, dict) and "observation" in obs:
        return np.asarray(obs["observation"], dtype=np.float32)
    return None


def collect_episode(
    env,
    get_oracle_action: Callable,
    *,
    max_steps: int,
    resolution: int,
    task_name: str,
    episode_id: int,
    crop_box: tuple[int, int, int, int] | None = None,
    record_proprioception: bool = True,
) -> Trajectory:
    obs = env.reset()

    images: list[np.ndarray] = [_crop_and_resize(env.render("rgb_array"), resolution, crop_box)]
    proprios: list[np.ndarray] = []
    if record_proprioception:
        p0 = _proprio_from_obs(obs)
        if p0 is not None:
            proprios.append(p0)
        else:
            record_proprioception = False  # env doesn't support it; skip
    actions: list[np.ndarray] = []
    success = False

    for _ in range(max_steps):
        action = np.asarray(get_oracle_action(obs), dtype=np.float32)
        obs, _reward, done, info = env.step(action)
        actions.append(action)
        images.append(_crop_and_resize(env.render("rgb_array"), resolution, crop_box))
        if record_proprioception:
            proprios.append(_proprio_from_obs(obs))
        is_success = bool(info.get("is_success", False))
        if is_success or done:
            success = is_success
            break

    proprio_arr = np.stack(proprios) if record_proprioception and proprios else None
    return Trajectory(
        images=np.stack(images),
        actions=np.stack(actions),
        success=success,
        task_name=task_name,
        episode_id=episode_id,
        proprioception=proprio_arr,
    )
