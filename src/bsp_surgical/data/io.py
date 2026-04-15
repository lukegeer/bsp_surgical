from pathlib import Path

import numpy as np

from bsp_surgical.data.trajectory import Trajectory


def save_trajectory(trajectory: Trajectory, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=trajectory.images,
        actions=trajectory.actions,
        success=np.array(trajectory.success),
        task_name=np.array(trajectory.task_name),
        episode_id=np.array(trajectory.episode_id),
    )


def load_trajectory(path: Path) -> Trajectory:
    with np.load(Path(path), allow_pickle=False) as data:
        return Trajectory(
            images=data["images"],
            actions=data["actions"],
            success=bool(data["success"]),
            task_name=str(data["task_name"]),
            episode_id=int(data["episode_id"]),
        )
