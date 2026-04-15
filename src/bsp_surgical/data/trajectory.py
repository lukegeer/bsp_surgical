from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    images: np.ndarray
    actions: np.ndarray
    success: bool
    task_name: str
    episode_id: int

    def __post_init__(self) -> None:
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError(f"images must be shape (T+1, H, W, 3); got {self.images.shape}")
        if self.images.dtype != np.uint8:
            raise ValueError(f"images must be uint8; got {self.images.dtype}")
        if self.actions.ndim != 2:
            raise ValueError(f"actions must be shape (T, action_dim); got {self.actions.shape}")
        if len(self.actions) < 1:
            raise ValueError("trajectory must have at least one transition")
        if len(self.images) != len(self.actions) + 1:
            raise ValueError(
                f"images ({len(self.images)}) must be one longer than actions ({len(self.actions)})"
            )

    @property
    def num_transitions(self) -> int:
        return len(self.actions)

    @property
    def goal_image(self) -> np.ndarray:
        return self.images[-1]
