"""Train latent subgoal diffusion on (z_start, z_mid, z_end) triples
encoded by a frozen RGBDSegEncoder."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bsp_surgical.models.rgbd_encoder import RGBDSegEncoder
from bsp_surgical.models.subgoal_diffusion import SubgoalDiffusion
from bsp_surgical.training.rgbd_dataset import RGBDSegSubgoalDataset


def _device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device(req)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--encoder-ckpt", type=Path, required=True,
                        help="Path to rgbd_dynamics.pt checkpoint")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--num-timesteps", type=int, default=200)
    parser.add_argument("--min-span", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)

    # Load frozen encoder
    ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    encoder = RGBDSegEncoder(
        num_seg_channels=ckpt["num_seg_channels"],
        resolution=128,
        feature_dim=ckpt["feature_dim"],
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    feature_dim = ckpt["feature_dim"]
    print(f"loaded encoder feature_dim={feature_dim}")

    dataset = RGBDSegSubgoalDataset(
        args.data_dir, num_seg_channels=ckpt["num_seg_channels"], min_span=args.min_span,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"dataset: {len(dataset)} episodes")

    diffusion = SubgoalDiffusion(
        latent_dim=feature_dim, hidden=args.hidden,
        num_timesteps=args.num_timesteps,
    ).to(device)
    print(f"diffusion params: {sum(p.numel() for p in diffusion.parameters()):,}")
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_loss = 0.0
        n = 0
        for batch in loader:
            start_bundle, mid_bundle, end_bundle = batch
            with torch.no_grad():
                z_start = encoder(*[x.to(device) for x in start_bundle])
                z_mid = encoder(*[x.to(device) for x in mid_bundle])
                z_end = encoder(*[x.to(device) for x in end_bundle])
            loss = diffusion.training_loss(z_mid, z_start, z_end)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n += 1
            step += 1
            if step % args.log_every == 0:
                print(f"step {step:>5d}  loss={loss.item():.4f}  ({step/(time.time()-t0):.1f} step/s)")

        history.append({"epoch": epoch, "loss": ep_loss / n})
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"== epoch {epoch}: loss={ep_loss/n:.4f}")

    torch.save({
        "diffusion": diffusion.state_dict(),
        "feature_dim": feature_dim,
        "hidden": args.hidden,
        "num_timesteps": args.num_timesteps,
        "args": vars(args) | {"data_dir": str(args.data_dir),
                               "encoder_ckpt": str(args.encoder_ckpt),
                               "out_dir": str(args.out_dir)},
    }, args.out_dir / "subgoal_diffusion.pt")
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
