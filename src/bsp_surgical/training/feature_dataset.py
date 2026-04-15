"""Datasets that read pre-encoded backbone features instead of raw images.

Two variants, mirroring the raw-image counterparts:
  - FeatureTransitionDataset: yields (z_t, action_t, z_{t+1}) triples for
    forward/inverse dynamics training.
  - FeatureSubgoalDataset:   yields (z_start, z_quarter, z_mid, z_end) per
    episode for subgoal-MLP training.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bsp_surgical.data.io import load_trajectory


def _load_features(feature_path: Path) -> np.ndarray:
    with np.load(feature_path) as data:
        return data["features"]  # (T+1, feature_dim) float32


class FeatureTransitionDataset(Dataset):
    """For each transition t, yields (z_t, a_t, z_{t+1})."""

    def __init__(self, raw_dir: Path, feature_dir: Path):
        raw_dir = Path(raw_dir)
        feature_dir = Path(feature_dir)
        raw_paths = sorted(raw_dir.glob("ep_*.npz"))
        if not raw_paths:
            raise FileNotFoundError(f"no episodes at {raw_dir}")

        self._features: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._episode_starts: list[int] = []
        total = 0
        for raw_path in raw_paths:
            feat_path = feature_dir / raw_path.name
            if not feat_path.exists():
                raise FileNotFoundError(
                    f"features missing for {raw_path.name}: run precompute_features.py first"
                )
            feats = _load_features(feat_path)
            traj = load_trajectory(raw_path)
            if len(feats) != traj.num_transitions + 1:
                raise ValueError(
                    f"{raw_path.name}: {len(feats)} features vs {traj.num_transitions+1} frames"
                )
            self._features.append(feats)
            self._actions.append(traj.actions)
            self._episode_starts.append(total)
            total += traj.num_transitions
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        ep_idx = _find_episode(self._episode_starts, idx)
        offset = idx - self._episode_starts[ep_idx]
        feats = self._features[ep_idx]
        actions = self._actions[ep_idx]
        return (
            torch.from_numpy(feats[offset]),
            torch.from_numpy(actions[offset]),
            torch.from_numpy(feats[offset + 1]),
        )


class FeatureSubgoalDataset(Dataset):
    """For each episode, yields (z_start, z_quarter, z_mid, z_end)."""

    def __init__(
        self,
        raw_dir: Path,
        feature_dir: Path,
        min_transitions: int = 3,
    ):
        raw_dir = Path(raw_dir)
        feature_dir = Path(feature_dir)
        raw_paths = sorted(raw_dir.glob("ep_*.npz"))
        if not raw_paths:
            raise FileNotFoundError(f"no episodes at {raw_dir}")

        self.quadruples: list[torch.Tensor] = []
        self.episode_indices: list[tuple[int, int, int, int]] = []
        for raw_path in raw_paths:
            feat_path = feature_dir / raw_path.name
            if not feat_path.exists():
                raise FileNotFoundError(
                    f"features missing for {raw_path.name}: run precompute_features.py first"
                )
            traj = load_trajectory(raw_path)
            T = traj.num_transitions
            if T < min_transitions:
                continue
            idx = (0, T // 4, T // 2, T)
            feats = _load_features(feat_path)
            self.quadruples.append(torch.from_numpy(feats[list(idx)]))
            self.episode_indices.append(idx)

    def __len__(self) -> int:
        return len(self.quadruples)

    def __getitem__(self, idx: int):
        q = self.quadruples[idx]
        return q[0], q[1], q[2], q[3]


def _find_episode(starts: list[int], idx: int) -> int:
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= idx:
            lo = mid
        else:
            hi = mid - 1
    return lo
