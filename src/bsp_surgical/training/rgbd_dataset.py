"""Dataset that returns raw (RGB, seg_onehot, depth, action_chunk,
next_RGB, next_seg_onehot, next_depth) per transition. Features are
computed on-the-fly by the trainable encoder — no precompute cache."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bsp_surgical.data.io import load_trajectory
from bsp_surgical.models.segdepth_encoder import seg_to_onehot


class RGBDSegDataset(Dataset):
    """Yields (rgb_t, seg_t, depth_t, action_chunk, rgb_next, seg_next, depth_next).

    RGB normalized to [0,1]; seg one-hot (K channels); depth as-is in [0,1]."""

    def __init__(
        self,
        raw_dir: Path,
        chunk_size: int = 5,
        num_seg_channels: int = 8,
    ):
        raw_dir = Path(raw_dir)
        self.chunk_size = chunk_size
        self.num_seg_channels = num_seg_channels
        raw_paths = sorted(raw_dir.glob("ep_*.npz"))
        if not raw_paths:
            raise FileNotFoundError(f"no episodes at {raw_dir}")

        self._trajs = []
        self._episode_starts: list[int] = []
        total = 0
        for p in raw_paths:
            t = load_trajectory(p)
            if t.segmentation is None or t.depth is None:
                raise ValueError(f"{p} has no segmentation/depth; recollect with --record-segdepth")
            self._trajs.append(t)
            self._episode_starts.append(total)
            total += t.num_transitions
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        ep_idx = _find_ep(self._episode_starts, idx)
        offset = idx - self._episode_starts[ep_idx]
        traj = self._trajs[ep_idx]
        T = traj.num_transitions

        # action chunk with padding
        real_end = min(offset + self.chunk_size, T)
        chunk = traj.actions[offset:real_end]
        if len(chunk) < self.chunk_size:
            pad = np.tile(chunk[-1:], (self.chunk_size - len(chunk), 1))
            chunk = np.concatenate([chunk, pad], axis=0)
        k = real_end - offset

        rgb_t = _rgb_to_tensor(traj.images[offset])
        rgb_next = _rgb_to_tensor(traj.images[offset + k])
        seg_t = torch.from_numpy(seg_to_onehot(traj.segmentation[offset], self.num_seg_channels))
        seg_next = torch.from_numpy(seg_to_onehot(traj.segmentation[offset + k], self.num_seg_channels))
        depth_t = torch.from_numpy(traj.depth[offset]).unsqueeze(0)      # (1, H, W)
        depth_next = torch.from_numpy(traj.depth[offset + k]).unsqueeze(0)

        return (
            rgb_t, seg_t, depth_t,
            torch.from_numpy(chunk.astype(np.float32)),
            rgb_next, seg_next, depth_next,
        )


class RGBDSegSubgoalDataset(Dataset):
    """For subgoal diffusion: yields random (z_start, z_mid, z_end) triples
    per episode via random-window sampling. Encoding happens at train
    time against a frozen encoder checkpoint."""

    def __init__(
        self,
        raw_dir: Path,
        num_seg_channels: int = 8,
        min_span: int = 4,
    ):
        raw_dir = Path(raw_dir)
        self.num_seg_channels = num_seg_channels
        self.min_span = min_span
        paths = sorted(raw_dir.glob("ep_*.npz"))
        if not paths:
            raise FileNotFoundError(f"no episodes at {raw_dir}")
        self._trajs = []
        for p in paths:
            t = load_trajectory(p)
            if t.num_transitions < min_span:
                continue
            if t.segmentation is None or t.depth is None:
                raise ValueError(f"{p} missing segdepth")
            self._trajs.append(t)

    def __len__(self) -> int:
        return len(self._trajs)

    def __getitem__(self, idx: int):
        traj = self._trajs[idx]
        T = traj.num_transitions
        span = int(np.random.randint(self.min_span, T + 1))
        a = int(np.random.randint(0, T - span + 1))
        c = a + span
        m = (a + c) // 2

        return (
            _frame_bundle(traj, a, self.num_seg_channels),
            _frame_bundle(traj, m, self.num_seg_channels),
            _frame_bundle(traj, c, self.num_seg_channels),
        )


def _frame_bundle(traj, idx: int, num_seg_channels: int):
    rgb = _rgb_to_tensor(traj.images[idx])
    seg = torch.from_numpy(seg_to_onehot(traj.segmentation[idx], num_seg_channels))
    depth = torch.from_numpy(traj.depth[idx]).unsqueeze(0)
    return rgb, seg, depth


def _rgb_to_tensor(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0


def _find_ep(starts: list[int], idx: int) -> int:
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        m = (lo + hi + 1) // 2
        if starts[m] <= idx:
            lo = m
        else:
            hi = m - 1
    return lo
