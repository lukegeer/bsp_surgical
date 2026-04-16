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
    # Optional (T+1, H, W) int32 segmentation map (body/link IDs from
    # PyBullet) and (T+1, H, W) float32 depth buffer. Privileged
    # perception signals — valid to use in a simulator study where
    # the perception stack would come from a real segmentation model
    # on deployment.
    segmentation: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None

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
        if self.segmentation is not None:
            if self.segmentation.ndim != 3:
                raise ValueError(f"segmentation must be (T+1, H, W); got {self.segmentation.shape}")
            if len(self.segmentation) != len(self.images):
                raise ValueError(
                    f"segmentation ({len(self.segmentation)}) must match images ({len(self.images)})"
                )
        if self.depth is not None:
            if self.depth.ndim != 3:
                raise ValueError(f"depth must be (T+1, H, W); got {self.depth.shape}")
            if len(self.depth) != len(self.images):
                raise ValueError(
                    f"depth ({len(self.depth)}) must match images ({len(self.images)})"
                )

    @property
    def num_transitions(self) -> int:
        return len(self.actions)

    @property
    def goal_image(self) -> np.ndarray:
        return self.images[-1]
