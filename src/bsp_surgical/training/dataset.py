from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bsp_surgical.data.io import load_trajectory


class TransitionDataset(Dataset):
    """Yields (img_t, action_t, img_{t+1}) triples across all episodes in a directory.

    Episodes are discovered by glob (`ep_*.npz`). Images are returned as
    float32 tensors in [0, 1] with shape (C, H, W); actions as float32 (action_dim,).
    """

    def __init__(self, root: Path):
        root = Path(root)
        self._episode_paths = sorted(root.glob("ep_*.npz"))
        if not self._episode_paths:
            raise FileNotFoundError(f"no episodes found under {root}")

        # build flat index: for each global i, which episode and offset
        self._episode_starts: list[int] = []
        self._episode_trajectories: list = [None] * len(self._episode_paths)
        total = 0
        for path in self._episode_paths:
            traj = load_trajectory(path)
            self._episode_starts.append(total)
            total += traj.num_transitions
            self._episode_trajectories[self._episode_paths.index(path)] = traj
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        # binary search for episode
        ep_idx = _find_episode(self._episode_starts, idx)
        offset = idx - self._episode_starts[ep_idx]
        traj = self._episode_trajectories[ep_idx]

        img_t = _to_tensor(traj.images[offset])
        img_next = _to_tensor(traj.images[offset + 1])
        action = torch.from_numpy(traj.actions[offset])
        return img_t, action, img_next


def _to_tensor(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0


def _find_episode(starts: list[int], idx: int) -> int:
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= idx:
            lo = mid
        else:
            hi = mid - 1
    return lo
