"""Encode selected frames from training trajectories into an anchor
bank for compositional subgoal reranking.

For each trajectory we take anchors at canonical phase-transition
moments: start, T/4, T/2, 3T/4, end. Over 1000 episodes this gives
~5000 anchor features spanning the training-distribution of
achievable states. At test time we rerank diffusion-sampled subgoal
candidates by cosine similarity to this bank, picking the most
in-distribution candidate.

Example:
    python scripts/build_anchor_bank.py \\
        --data-dirs data/needle_reach_sd data/needle_pick_sd \\
        --encoder-ckpt checkpoints/reach_sd/dynamics/rgbd_dynamics.pt \\
        --out checkpoints/anchors_reach_pick.npz
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from bsp_surgical.data.io import load_trajectory
from bsp_surgical.models.rgbd_encoder import RGBDSegEncoder, seg_to_onehot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", type=Path, nargs="+", required=True,
                        help="One or more training data directories.")
    parser.add_argument("--encoder-ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="Fractions of trajectory length to sample anchors at")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else ("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    )

    ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    num_seg = ckpt["num_seg_channels"]
    encoder = RGBDSegEncoder(
        num_seg_channels=num_seg, resolution=128, feature_dim=ckpt["feature_dim"],
    ).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)

    anchors: list[np.ndarray] = []
    task_tags: list[str] = []
    frac_tags: list[float] = []

    for data_dir in args.data_dirs:
        paths = sorted(data_dir.glob("ep_*.npz"))
        print(f"{data_dir}: {len(paths)} episodes")
        for path in paths:
            traj = load_trajectory(path)
            if traj.segmentation is None or traj.depth is None:
                continue
            T = traj.num_transitions
            for frac in args.fractions:
                idx = int(round(frac * T))
                idx = min(idx, T)
                rgb = torch.from_numpy(traj.images[idx]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                seg = torch.from_numpy(seg_to_onehot(traj.segmentation[idx], num_seg)).unsqueeze(0).to(device)
                depth = torch.from_numpy(traj.depth[idx]).unsqueeze(0).unsqueeze(0).to(device)
                with torch.inference_mode():
                    z = encoder(rgb, seg, depth)
                anchors.append(z.squeeze(0).cpu().numpy())
                task_tags.append(traj.task_name)
                frac_tags.append(frac)

    anchors_arr = np.stack(anchors).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        anchors=anchors_arr,
        tasks=np.array(task_tags),
        fractions=np.array(frac_tags, dtype=np.float32),
    )
    print(f"saved {args.out} with {len(anchors_arr)} anchors ({anchors_arr.shape[1]}-d)")


if __name__ == "__main__":
    main()
