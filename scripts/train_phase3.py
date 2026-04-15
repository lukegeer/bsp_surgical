"""Phase 3: train the backward subgoal generator on latents from a frozen
Phase-2 VAE.

Example:
    python scripts/train_phase3.py --data-dir data/needle_reach \\
        --phase2-ckpt checkpoints/phase2/phase2.pt \\
        --out-dir checkpoints/phase3 --epochs 30 --batch-size 256
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bsp_surgical.models.vae import VAE
from bsp_surgical.models.subgoal import SubgoalGenerator
from bsp_surgical.training.phase3 import SubgoalDataset, phase3_train_step


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
    parser.add_argument("--phase2-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--min-transitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    print(f"device: {device}")

    # Load frozen Phase 2 VAE
    ckpt = torch.load(args.phase2_ckpt, map_location=device, weights_only=False)
    vae_args = ckpt["args"]
    vae = VAE(latent_dim=vae_args["latent_dim"], resolution=vae_args["resolution"])
    vae.load_state_dict(ckpt["vae"])
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    latent_dim = vae_args["latent_dim"]
    print(f"loaded VAE (latent_dim={latent_dim}, resolution={vae_args['resolution']})")

    # Encode all episodes → precomputed (z_start, z_quarter, z_mid, z_end)
    t0 = time.time()
    dataset = SubgoalDataset(
        args.data_dir, vae=vae, device=device,
        min_transitions=args.min_transitions,
    )
    print(f"encoded {len(dataset)} episodes in {time.time()-t0:.1f}s")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    subgoal = SubgoalGenerator(latent_dim=latent_dim, hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(subgoal.parameters(), lr=args.lr)

    history: list[dict] = []
    step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        ep_total = ep_gt = ep_pred = 0.0
        n = 0
        for batch in loader:
            batch = tuple(t.to(device) for t in batch)
            losses = phase3_train_step(subgoal, optimizer, batch)
            ep_total += losses["total"]
            ep_gt += losses["gt"]
            ep_pred += losses["pred"]
            n += 1
            step += 1
            if step % args.log_every == 0:
                print(f"step {step:>5d}  gt={losses['gt']:.4f}  "
                      f"pred={losses['pred']:.4f}  total={losses['total']:.4f}")

        history.append({"epoch": epoch,
                        "total": ep_total / n, "gt": ep_gt / n, "pred": ep_pred / n})
        print(f"== epoch {epoch}: gt={ep_gt/n:.4f}  pred={ep_pred/n:.4f}  total={ep_total/n:.4f}")

    ckpt_path = args.out_dir / "phase3.pt"
    torch.save({
        "subgoal": subgoal.state_dict(),
        "latent_dim": latent_dim,
        "hidden": args.hidden,
        "args": vars(args) | {"data_dir": str(args.data_dir),
                               "phase2_ckpt": str(args.phase2_ckpt),
                               "out_dir": str(args.out_dir)},
    }, ckpt_path)
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nsaved {ckpt_path}")


if __name__ == "__main__":
    main()
