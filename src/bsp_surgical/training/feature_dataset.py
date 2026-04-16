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
    """For each transition t, yields (z_t, actions, z_{t+k}) where:
      - k is 1 by default or randomly sampled in [1, max_step_jump]
      - actions is a chunk of `chunk_size` consecutive actions starting
        at t: [a_t, a_{t+1}, ..., a_{t+chunk_size-1}]. If chunk_size=1
        (default) the returned shape is (action_dim,) for backwards
        compatibility; otherwise (chunk_size, action_dim).

    Chunked actions let the inverse model learn a multi-step motion
    primitive (Seer-style). Essential on tasks with discrete phase
    transitions (grasp) where a memoryless single-step policy averages
    pre/post-transition actions and never commits to either."""

    def __init__(
        self,
        raw_dir: Path,
        feature_dir: Path,
        max_step_jump: int = 1,
        chunk_size: int = 1,
        with_proprio: bool = False,
    ):
        if max_step_jump < 1:
            raise ValueError(f"max_step_jump must be >= 1, got {max_step_jump}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.max_step_jump = max_step_jump
        self.chunk_size = chunk_size
        self.with_proprio = with_proprio
        self._proprios: list[np.ndarray] = []
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
            if self.with_proprio:
                if traj.proprioception is None:
                    raise ValueError(
                        f"{raw_path.name} has no proprioception, but with_proprio=True"
                    )
                self._proprios.append(traj.proprioception.astype(np.float32))
            self._episode_starts.append(total)
            total += traj.num_transitions
        self._total = total
        self.proprio_dim = self._proprios[0].shape[-1] if self._proprios else 0

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        ep_idx = _find_episode(self._episode_starts, idx)
        offset = idx - self._episode_starts[ep_idx]
        feats = self._features[ep_idx]
        actions = self._actions[ep_idx]
        T = len(actions)

        if self.chunk_size == 1:
            # single-step classic inverse: (z_t, a_t, z_{t+k})
            action_out = torch.from_numpy(actions[offset])
            if self.max_step_jump == 1 or T - offset <= 1:
                k = 1
            else:
                k = int(np.random.randint(1, min(self.max_step_jump, T - offset) + 1))
        else:
            # Chunked inverse: target is the frame after the full chunk so
            # the model sees (z_t, z_{t+K}) and learns to emit the K
            # actions that bridge them. Final-episode chunks are padded
            # with the last action; k reflects how many real actions
            # exist so target stays self-consistent.
            real_end = min(offset + self.chunk_size, T)
            chunk = actions[offset:real_end]
            if len(chunk) < self.chunk_size:
                pad = np.tile(chunk[-1:], (self.chunk_size - len(chunk), 1))
                chunk = np.concatenate([chunk, pad], axis=0)
            action_out = torch.from_numpy(chunk)
            k = real_end - offset
        items = [
            torch.from_numpy(feats[offset]),
            action_out,
            torch.from_numpy(feats[offset + k]),
        ]
        if self.with_proprio:
            items.append(torch.from_numpy(self._proprios[ep_idx][offset]))
        return tuple(items)


class FeatureSubgoalDataset(Dataset):
    """For each episode, yields (z_start, z_quarter, z_mid, z_end).

    Two modes:
      - random_windows=False (default): fixed quadruple at indices
        (0, T/4, T/2, T). Same as before — one training example per episode.
      - random_windows=True: at each __getitem__ call, sample a random
        window (a, c) with c-a >= min_span, then midpoint m = (a+c)//2,
        quarter q = (a+m)//2. This gives many more distinct training
        triples, which the fixed-index version lacks and the MLP needs
        to actually learn a generalizable midpoint function."""

    def __init__(
        self,
        raw_dir: Path,
        feature_dir: Path,
        min_transitions: int = 3,
        random_windows: bool = False,
        min_span: int = 4,
    ):
        raw_dir = Path(raw_dir)
        feature_dir = Path(feature_dir)
        raw_paths = sorted(raw_dir.glob("ep_*.npz"))
        if not raw_paths:
            raise FileNotFoundError(f"no episodes at {raw_dir}")

        self.random_windows = random_windows
        self.min_span = min_span
        self.quadruples: list[torch.Tensor] = []
        self.episode_indices: list[tuple[int, int, int, int]] = []
        self._full_features: list[np.ndarray] = []
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
            feats = _load_features(feat_path)
            if random_windows:
                # need at least one valid window of size >= min_span
                if T < min_span:
                    continue
                self._full_features.append(feats)
                self.episode_indices.append((0, 0, 0, 0))  # placeholder
            else:
                idx = (0, T // 4, T // 2, T)
                self.quadruples.append(torch.from_numpy(feats[list(idx)]))
                self.episode_indices.append(idx)

    def __len__(self) -> int:
        if self.random_windows:
            return len(self._full_features)
        return len(self.quadruples)

    def __getitem__(self, idx: int):
        if self.random_windows:
            feats = self._full_features[idx]
            T = len(feats) - 1  # num transitions
            span = int(np.random.randint(self.min_span, T + 1))
            a = int(np.random.randint(0, T - span + 1))
            c = a + span
            m = (a + c) // 2
            q = (a + m) // 2
            return (
                torch.from_numpy(feats[a]),
                torch.from_numpy(feats[q]),
                torch.from_numpy(feats[m]),
                torch.from_numpy(feats[c]),
            )
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
