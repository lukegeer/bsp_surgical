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


def _crop_and_resize_int(
    frame: np.ndarray,
    resolution: int,
    crop_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Nearest-neighbor resize for integer maps (segmentation)."""
    if crop_box is not None:
        y1, y2, x1, x2 = crop_box
        frame = frame[y1:y2, x1:x2]
    if frame.shape[0] == resolution and frame.shape[1] == resolution:
        return frame.astype(np.int32, copy=False)
    import cv2

    resized = cv2.resize(frame.astype(np.int32), (resolution, resolution),
                         interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.int32, copy=False)


def _crop_and_resize_float(
    frame: np.ndarray,
    resolution: int,
    crop_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Bilinear resize for float maps (depth)."""
    if crop_box is not None:
        y1, y2, x1, x2 = crop_box
        frame = frame[y1:y2, x1:x2]
    if frame.shape[0] == resolution and frame.shape[1] == resolution:
        return frame.astype(np.float32, copy=False)
    import cv2

    resized = cv2.resize(frame.astype(np.float32), (resolution, resolution),
                         interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32, copy=False)


def _render_full(env, record_segdepth: bool):
    """Return (rgb, seg or None, depth or None) from the env's camera."""
    rgb = env.render("rgb_array")
    if not record_segdepth:
        return rgb, None, None
    import pybullet as p

    view = env._view_matrix
    proj = env._proj_matrix
    h, w = rgb.shape[:2]
    _, _, _, depth, seg = p.getCameraImage(
        width=w, height=h,
        viewMatrix=view, projectionMatrix=proj,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        renderer=p.ER_TINY_RENDERER,
    )
    depth_np = np.asarray(depth).reshape(h, w).astype(np.float32)
    seg_np = np.asarray(seg).reshape(h, w).astype(np.int32)
    return rgb, seg_np, depth_np


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
    record_segdepth: bool = False,
) -> Trajectory:
    obs = env.reset()

    rgb0, seg0, dep0 = _render_full(env, record_segdepth)
    images: list[np.ndarray] = [_crop_and_resize(rgb0, resolution, crop_box)]
    segs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    if record_segdepth and seg0 is not None:
        segs.append(_crop_and_resize_int(seg0, resolution, crop_box))
        depths.append(_crop_and_resize_float(dep0, resolution, crop_box))

    proprios: list[np.ndarray] = []
    if record_proprioception:
        p0 = _proprio_from_obs(obs)
        if p0 is not None:
            proprios.append(p0)
        else:
            record_proprioception = False
    actions: list[np.ndarray] = []
    success = False

    for _ in range(max_steps):
        action = np.asarray(get_oracle_action(obs), dtype=np.float32)
        obs, _reward, done, info = env.step(action)
        actions.append(action)
        rgb_t, seg_t, dep_t = _render_full(env, record_segdepth)
        images.append(_crop_and_resize(rgb_t, resolution, crop_box))
        if record_segdepth and seg_t is not None:
            segs.append(_crop_and_resize_int(seg_t, resolution, crop_box))
            depths.append(_crop_and_resize_float(dep_t, resolution, crop_box))
        if record_proprioception:
            proprios.append(_proprio_from_obs(obs))
        is_success = bool(info.get("is_success", False))
        if is_success or done:
            success = is_success
            break

    proprio_arr = np.stack(proprios) if record_proprioception and proprios else None
    seg_arr = np.stack(segs) if segs else None
    depth_arr = np.stack(depths) if depths else None
    return Trajectory(
        images=np.stack(images),
        actions=np.stack(actions),
        success=success,
        task_name=task_name,
        episode_id=episode_id,
        proprioception=proprio_arr,
        segmentation=seg_arr,
        depth=depth_arr,
    )
