"""Phase 2 training: joint VAE + forward + inverse dynamics.

Example:
    python scripts/train_phase2.py --data-dir data/needle_reach \\
        --out-dir checkpoints/phase2 --epochs 20 --batch-size 64
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bsp_surgical.models.vae import VAE
from bsp_surgical.models.dynamics import ForwardDynamics, InverseDynamics
from bsp_surgical.training.dataset import TransitionDataset
from bsp_surgical.training.phase2 import Phase2Models, train_step, compute_phase2_losses


def _resolve_device(request: str) -> torch.device:
    if request == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(request)


def _batch_to_device(batch, device):
    return tuple(t.to(device, non_blocking=True) for t in batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--backbone", choices=["simple", "resnet18"], default="simple")
    parser.add_argument("--pretrained", action="store_true", help="ImageNet weights for resnet18")
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--jaw-weight", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    print(f"device: {device}")

    dataset = TransitionDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    print(f"dataset: {len(dataset)} transitions across "
          f"{len(dataset._episode_paths)} episodes")

    models = Phase2Models(
        vae=VAE(
            latent_dim=args.latent_dim, resolution=args.resolution,
            backbone=args.backbone, pretrained=args.pretrained,
        ),
        forward=ForwardDynamics(latent_dim=args.latent_dim, action_dim=5),
        inverse=InverseDynamics(latent_dim=args.latent_dim, action_dim=5),
    ).to(device)

    params = (list(models.vae.parameters())
              + list(models.forward.parameters())
              + list(models.inverse.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    history: list[dict] = []
    step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        epoch_losses = {"recon": 0.0, "kl": 0.0, "forward": 0.0, "inverse": 0.0, "total": 0.0}
        n_batches = 0
        for batch in loader:
            batch = _batch_to_device(batch, device)
            losses = train_step(
                models, optimizer, batch,
                kl_weight=args.kl_weight, jaw_weight=args.jaw_weight,
            )
            clean = all(not (math.isnan(v) or math.isinf(v)) for v in losses.values())
            if clean:
                for k, v in losses.items():
                    epoch_losses[k] += v
                n_batches += 1
            step += 1
            if step % args.log_every == 0:
                dt = time.time() - t_start
                print(f"step {step:>6d}  "
                      f"recon={losses['recon']:.4f}  kl={losses['kl']:.3f}  "
                      f"fwd={losses['forward']:.4f}  inv={losses['inverse']:.4f}  "
                      f"total={losses['total']:.4f}  ({step/dt:.1f} step/s)")

        epoch_mean = {k: v / n_batches for k, v in epoch_losses.items()}
        history.append({"epoch": epoch, **epoch_mean})
        print(f"== epoch {epoch}: recon={epoch_mean['recon']:.4f}  "
              f"kl={epoch_mean['kl']:.3f}  fwd={epoch_mean['forward']:.4f}  "
              f"inv={epoch_mean['inverse']:.4f}  total={epoch_mean['total']:.4f}")

    ckpt_path = args.out_dir / "phase2.pt"
    torch.save({
        "vae": models.vae.state_dict(),
        "forward": models.forward.state_dict(),
        "inverse": models.inverse.state_dict(),
        "args": vars(args) | {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir)},
    }, ckpt_path)
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nsaved {ckpt_path}")


if __name__ == "__main__":
    main()
