"""Cache pretrained backbone features for all collected episodes.

Loads raw episodes from --data-dir, encodes every frame with a frozen
backbone (default DINOv2-base), and writes (T+1, feature_dim) float32
arrays to --out-dir/<backbone>/ep_NNNNN.npz alongside the original
trajectory data.

Downstream training reads these cached features instead of re-running
the backbone every epoch.

Example:
    python scripts/precompute_features.py \\
        --data-dir data/needle_reach \\
        --out-dir data/needle_reach/features \\
        --backbone dinov2-base --batch-size 32
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from bsp_surgical.data.io import load_trajectory
from bsp_surgical.models.perception import PretrainedEncoder


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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backbone", default="dinov2-base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--amp", action="store_true",
                        help="Use bf16 autocast for encoder forward (MPS-safe, ~2x faster).")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"device: {device}")
    out_dir = args.out_dir / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = PretrainedEncoder(name=args.backbone, device=device)
    print(f"loaded backbone '{args.backbone}' (feature_dim={encoder.feature_dim})")

    paths = sorted(args.data_dir.glob("ep_*.npz"))
    print(f"found {len(paths)} episodes")

    total_frames = 0
    t_start = time.time()
    for i, path in enumerate(paths):
        out_path = out_dir / path.name
        if out_path.exists() and not args.overwrite:
            continue
        traj = load_trajectory(path)
        if args.amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                features = encoder.encode_numpy_frames(traj.images, batch_size=args.batch_size)
        else:
            features = encoder.encode_numpy_frames(traj.images, batch_size=args.batch_size)
        np.savez_compressed(out_path, features=features.float().numpy().astype(np.float32))
        total_frames += len(traj.images)

        if (i + 1) % 25 == 0 or i == len(paths) - 1:
            dt = time.time() - t_start
            print(f"[{i+1:>5d}/{len(paths)}] frames={total_frames} "
                  f"({total_frames/dt:.1f} frame/s, {(i+1)/dt:.1f} ep/s)")

    print(f"\nDone. {total_frames} frames encoded to {out_dir} in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
