"""Train forward + inverse dynamics on cached backbone features.

No VAE — just two MLPs operating on frozen-perception feature vectors.
Much faster than the old phase-2 script because no encoder forward
pass is needed at training time.

Example:
    python scripts/train_dynamics.py \\
        --data-dir data/needle_reach \\
        --feature-dir data/needle_reach/features/dinov2-base \\
        --out-dir checkpoints/dynamics --epochs 50 --hidden 1024
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bsp_surgical.models.dynamics import ForwardDynamics, InverseDynamics
from bsp_surgical.models.losses import forward_dynamics_loss, inverse_dynamics_loss
from bsp_surgical.training.feature_dataset import FeatureTransitionDataset


def _resolve_device(request: str) -> torch.device:
    if request == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(request)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="Inverse predicts this many consecutive actions per call. "
                             "Seer uses 3, ACT uses 100. Essential on multi-phase tasks.")
    parser.add_argument("--max-step-jump", type=int, default=1,
                        help="Train inverse on (z_t, z_{t+k}) with k randomly in [1, K]. "
                             "K=1 is pure consecutive-frame training; larger K helps the "
                             "inverse model handle eval-time long-jump targets.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--jaw-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    print(f"device: {device}")

    dataset = FeatureTransitionDataset(
        args.data_dir, args.feature_dir,
        max_step_jump=args.max_step_jump,
        chunk_size=args.chunk_size,
    )
    # Probe feature dim from one sample
    z0, _, _ = dataset[0]
    feature_dim = z0.shape[0]
    print(f"dataset: {len(dataset)} transitions, feature_dim={feature_dim}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    fwd = ForwardDynamics(latent_dim=feature_dim, action_dim=5, hidden=args.hidden).to(device)
    inv = InverseDynamics(
        latent_dim=feature_dim, action_dim=5, hidden=args.hidden,
        chunk_size=args.chunk_size,
    ).to(device)

    params = list(fwd.parameters()) + list(inv.parameters())
    print(f"trainable params: {sum(p.numel() for p in params):,}")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict] = []
    step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        fwd.train()
        inv.train()
        ep_fwd = ep_inv = 0.0
        n = 0
        for batch in loader:
            z_t, a, z_next = (t.to(device) for t in batch)
            # forward dynamics always operates on the first action of the chunk
            a_first = a if a.dim() == 2 else a[:, 0]
            delta_z_pred = fwd(z_t, a_first)
            z_next_pred = z_t + delta_z_pred
            loss_fwd = forward_dynamics_loss(z_next_pred, z_next)
            a_pred = inv(z_t, z_next)
            loss_inv = inverse_dynamics_loss(a_pred, a, jaw_weight=args.jaw_weight)
            loss = loss_fwd + loss_inv

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            if not (math.isnan(loss.item()) or math.isinf(loss.item())):
                ep_fwd += loss_fwd.item()
                ep_inv += loss_inv.item()
                n += 1
            step += 1
            if step % args.log_every == 0:
                dt = time.time() - t_start
                print(f"step {step:>6d}  fwd={loss_fwd.item():.4f}  "
                      f"inv={loss_inv.item():.4f}  ({step/dt:.1f} step/s)")

        history.append({
            "epoch": epoch,
            "forward": ep_fwd / max(n, 1),
            "inverse": ep_inv / max(n, 1),
        })
        print(f"== epoch {epoch}: fwd={ep_fwd/max(n,1):.4f}  inv={ep_inv/max(n,1):.4f}")

    ckpt_path = args.out_dir / "dynamics.pt"
    torch.save({
        "forward": fwd.state_dict(),
        "inverse": inv.state_dict(),
        "feature_dim": feature_dim,
        "hidden": args.hidden,
        "chunk_size": args.chunk_size,
        "args": vars(args) | {
            "data_dir": str(args.data_dir),
            "feature_dir": str(args.feature_dir),
            "out_dir": str(args.out_dir),
        },
    }, ckpt_path)
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nsaved {ckpt_path}")


if __name__ == "__main__":
    main()
