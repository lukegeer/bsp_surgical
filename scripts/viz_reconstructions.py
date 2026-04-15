"""Save a side-by-side grid of (original, reconstructed) frames.

Diagnostic for Phase 2: if the VAE collapses to mean colour or loses the
needle tip, reconstructions will look blurry/featureless and no amount
of downstream training will recover task-relevant info.

Example:
    python scripts/viz_reconstructions.py \\
        --phase2-ckpt checkpoints/phase2/phase2.pt \\
        --data-dir data/needle_reach \\
        --out checkpoints/phase2/recons.png --num-samples 8
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from bsp_surgical.data.io import load_trajectory
from bsp_surgical.models.vae import VAE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-ckpt", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() and args.device == "auto" else (
        "cuda" if torch.cuda.is_available() and args.device == "auto" else
        args.device if args.device != "auto" else "cpu"
    )

    ckpt = torch.load(args.phase2_ckpt, map_location=device, weights_only=False)
    vae_args = ckpt["args"]
    vae = VAE(latent_dim=vae_args["latent_dim"], resolution=vae_args["resolution"]).to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()
    resolution = vae_args["resolution"]

    # Sample frames from the last few episodes (held-out-ish — not used for training pref)
    paths = sorted(args.data_dir.glob("ep_*.npz"))[-20:]
    frames = []
    rng = np.random.default_rng(0)
    while len(frames) < args.num_samples and paths:
        path = paths[rng.integers(len(paths))]
        traj = load_trajectory(path)
        t = int(rng.integers(traj.num_transitions + 1))
        frames.append(traj.images[t])

    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().to(device) / 255.0

    with torch.no_grad():
        recon, mu, logvar = vae(x)

    originals = x.cpu().numpy().transpose(0, 2, 3, 1)
    recons = recon.cpu().numpy().transpose(0, 2, 3, 1)
    recons = np.clip(recons, 0, 1)

    fig, axes = plt.subplots(2, args.num_samples, figsize=(2 * args.num_samples, 4))
    for i in range(args.num_samples):
        axes[0, i].imshow(originals[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i])
        axes[1, i].axis("off")
    axes[0, 0].set_title("original", loc="left")
    axes[1, 0].set_title("reconstructed", loc="left")
    fig.suptitle(f"VAE recon @ resolution={resolution}, latent={vae_args['latent_dim']}",
                 fontsize=10)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")
    # also print latent stats — collapse indicator
    print(f"mu   mean={mu.mean().item():.3f}  std={mu.std().item():.3f}")
    print(f"sigma (from logvar) mean={torch.exp(0.5 * logvar).mean().item():.3f}")


if __name__ == "__main__":
    main()
