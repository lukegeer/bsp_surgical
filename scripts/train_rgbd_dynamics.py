"""End-to-end train the RGBDSegEncoder + InverseDynamics jointly on
segmentation+depth trajectories. No DINOv2, no precomputed features —
encoder is trained.

Example:
    python scripts/train_rgbd_dynamics.py \\
        --data-dir data/needle_pick_sd \\
        --out-dir checkpoints/pick_sd/dynamics \\
        --epochs 30 --batch-size 32 --chunk-size 5
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bsp_surgical.models.rgbd_encoder import RGBDSegEncoder
from bsp_surgical.models.dynamics import InverseDynamics
from bsp_surgical.models.losses import inverse_dynamics_loss
from bsp_surgical.training.rgbd_dataset import RGBDSegDataset


def _device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device(req)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--num-seg-channels", type=int, default=8)
    parser.add_argument("--inverse-hidden", type=int, default=1024)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    print(f"device: {device}")

    dataset = RGBDSegDataset(args.data_dir, chunk_size=args.chunk_size,
                             num_seg_channels=args.num_seg_channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
    print(f"dataset: {len(dataset)} transitions")

    encoder = RGBDSegEncoder(
        num_seg_channels=args.num_seg_channels,
        resolution=128,
        feature_dim=args.feature_dim,
    ).to(device)
    inverse = InverseDynamics(
        latent_dim=args.feature_dim, action_dim=5,
        hidden=args.inverse_hidden, chunk_size=args.chunk_size,
    ).to(device)

    params = list(encoder.parameters()) + list(inverse.parameters())
    print(f"trainable params: {sum(p.numel() for p in params):,}")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        encoder.train()
        inverse.train()
        ep_loss = 0.0
        n = 0
        for batch in loader:
            rgb, seg, depth, a, rgb_n, seg_n, depth_n = (x.to(device) for x in batch)
            z_t = encoder(rgb, seg, depth)
            z_next = encoder(rgb_n, seg_n, depth_n)
            a_pred = inverse(z_t, z_next)
            loss = inverse_dynamics_loss(a_pred, a)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if math.isfinite(loss.item()):
                ep_loss += loss.item()
                n += 1
            step += 1
            if step % args.log_every == 0:
                print(f"step {step:>6d}  inv={loss.item():.4f}  ({step/(time.time()-t0):.1f} step/s)")

        history.append({"epoch": epoch, "inv": ep_loss / max(n, 1)})
        print(f"== epoch {epoch}: inv={ep_loss/max(n,1):.4f}")

    torch.save({
        "encoder": encoder.state_dict(),
        "inverse": inverse.state_dict(),
        "feature_dim": args.feature_dim,
        "num_seg_channels": args.num_seg_channels,
        "chunk_size": args.chunk_size,
        "inverse_hidden": args.inverse_hidden,
        "args": vars(args) | {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir)},
    }, args.out_dir / "rgbd_dynamics.pt")
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
