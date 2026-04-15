from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bsp_surgical.data.io import load_trajectory
from bsp_surgical.models.vae import VAE
from bsp_surgical.models.subgoal import SubgoalGenerator
from bsp_surgical.models.losses import subgoal_dual_supervision_loss


class SubgoalDataset(Dataset):
    """Per episode, yields (z_start, z_quarter, z_mid, z_end) — four
    evenly-spaced latents produced by encoding the 0th, T/4th, T/2nd,
    and Tth frames of each trajectory with a frozen VAE."""

    def __init__(
        self,
        root: Path,
        vae: VAE,
        device: str = "cpu",
        min_transitions: int = 3,
    ):
        root = Path(root)
        paths = sorted(root.glob("ep_*.npz"))
        if not paths:
            raise FileNotFoundError(f"no episodes under {root}")

        vae = vae.eval().to(device)
        self.latents: list[torch.Tensor] = []
        self.episode_indices: list[tuple[int, int, int, int]] = []

        with torch.no_grad():
            for path in paths:
                traj = load_trajectory(path)
                T = traj.num_transitions
                if T < min_transitions:
                    continue
                indices = (0, T // 4, T // 2, T)

                frames = np.stack([traj.images[i] for i in indices])
                frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
                frames_t = frames_t.to(device)
                mu, _logvar = vae.encode(frames_t)
                self.latents.append(mu.detach().cpu())
                self.episode_indices.append(indices)

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int):
        z = self.latents[idx]
        return z[0], z[1], z[2], z[3]


def phase3_train_step(
    subgoal_mlp: SubgoalGenerator,
    optimizer: torch.optim.Optimizer,
    batch,
) -> dict[str, float]:
    z_start, z_quarter, z_mid, z_end = batch
    subgoal_mlp.train()
    optimizer.zero_grad(set_to_none=True)
    total, parts = subgoal_dual_supervision_loss(
        subgoal_mlp,
        z_start=z_start, z_quarter=z_quarter, z_mid=z_mid, z_end=z_end,
    )
    total.backward()
    optimizer.step()
    return {"total": total.item(), "gt": parts["gt"].item(), "pred": parts["pred"].item()}
