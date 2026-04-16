from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    images: np.ndarray
    actions: np.ndarray
    success: bool
    task_name: str
    episode_id: int
    # Optional per-step state vector from the env (e.g. SurRoL's
    # obs['observation'], typically 7-d EE pose + joint angles). Shape
    # (T+1, state_dim) when present. Giving this to the inverse model
    # closes the gap to privileged-state methods without fine-tuning
    # the frozen visual backbone.
    proprioception: Optional[np.ndarray] = None

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
        if self.proprioception is not None:
            if self.proprioception.ndim != 2:
                raise ValueError(f"proprioception must be 2D; got {self.proprioception.shape}")
            if len(self.proprioception) != len(self.images):
                raise ValueError(
                    f"proprioception ({len(self.proprioception)}) must match images ({len(self.images)})"
                )

    @property
    def num_transitions(self) -> int:
        return len(self.actions)

    @property
    def goal_image(self) -> np.ndarray:
        return self.images[-1]
