from pathlib import Path

import numpy as np

from bsp_surgical.data.trajectory import Trajectory


def save_trajectory(trajectory: Trajectory, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        images=trajectory.images,
        actions=trajectory.actions,
        success=np.array(trajectory.success),
        task_name=np.array(trajectory.task_name),
        episode_id=np.array(trajectory.episode_id),
    )
    if trajectory.proprioception is not None:
        payload["proprioception"] = trajectory.proprioception
    if trajectory.segmentation is not None:
        payload["segmentation"] = trajectory.segmentation
    if trajectory.depth is not None:
        payload["depth"] = trajectory.depth
    np.savez_compressed(path, **payload)


def load_trajectory(path: Path) -> Trajectory:
    with np.load(Path(path), allow_pickle=False) as data:
        proprio = data["proprioception"] if "proprioception" in data.files else None
        seg = data["segmentation"] if "segmentation" in data.files else None
        dep = data["depth"] if "depth" in data.files else None
        return Trajectory(
            images=data["images"],
            actions=data["actions"],
            success=bool(data["success"]),
            task_name=str(data["task_name"]),
            episode_id=int(data["episode_id"]),
            proprioception=proprio,
            segmentation=seg,
            depth=dep,
        )
