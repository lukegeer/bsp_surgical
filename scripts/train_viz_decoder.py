"""Train a small decoder that maps frozen backbone features back to
128x128 RGB images — for visualizing subgoals and debugging. Not used
in the inference path.

Example:
    python scripts/train_viz_decoder.py \\
        --data-dir data/needle_reach \\
        --feature-dir data/needle_reach/features/dinov2-base \\
        --out-dir checkpoints/viz_decoder --epochs 20
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from bsp_surgical.data.io import load_trajectory
from bsp_surgical.models.viz_decoder import FeatureDecoder


class FeatureImagePairs(Dataset):
    """Yields (feature, image_01) pairs across all frames of all episodes."""

    def __init__(self, data_dir: Path, feature_dir: Path):
        data_dir = Path(data_dir)
        feature_dir = Path(feature_dir)
        raw_paths = sorted(data_dir.glob("ep_*.npz"))
        if not raw_paths:
            raise FileNotFoundError(f"no episodes at {data_dir}")
        self._features: list[np.ndarray] = []
        self._images: list[np.ndarray] = []
        self._starts: list[int] = []
        total = 0
        for rp in raw_paths:
            fp = feature_dir / rp.name
            if not fp.exists():
                raise FileNotFoundError(f"missing features for {rp.name}")
            with np.load(fp) as d:
                feats = d["features"]
            traj = load_trajectory(rp)
            assert len(feats) == len(traj.images)
            self._features.append(feats)
            self._images.append(traj.images)
            self._starts.append(total)
            total += len(feats)
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        ep = _find_ep(self._starts, idx)
        off = idx - self._starts[ep]
        feat = torch.from_numpy(self._features[ep][off])
        img = torch.from_numpy(self._images[ep][off]).permute(2, 0, 1).float() / 255.0
        return feat, img


def _find_ep(starts, idx):
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        m = (lo + hi + 1) // 2
        if starts[m] <= idx:
            lo = m
        else:
            hi = m - 1
    return lo


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true",
                        help="bf16 autocast for forward (keeps grads in fp32)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    print(f"device: {device}")

    dataset = FeatureImagePairs(args.data_dir, args.feature_dir)
    feat0, _ = dataset[0]
    feature_dim = feat0.shape[0]
    print(f"dataset: {len(dataset)} frames, feature_dim={feature_dim}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    decoder = FeatureDecoder(feature_dim=feature_dim).to(device)
    print(f"trainable params: {sum(p.numel() for p in decoder.parameters()):,}")
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        decoder.train()
        ep_loss = 0.0
        n = 0
        for feat, img in loader:
            feat = feat.to(device)
            img = img.to(device)
            ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.amp \
                else torch.autocast(device_type=device.type, enabled=False)
            with ctx:
                pred = decoder(feat)
                loss = F.mse_loss(pred, img)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n += 1
            step += 1
            if step % args.log_every == 0:
                print(f"step {step:>6d}  recon={loss.item():.5f}  "
                      f"({step/(time.time()-t0):.1f} step/s)")

        history.append({"epoch": epoch, "recon": ep_loss / n})
        print(f"== epoch {epoch}: recon={ep_loss/n:.5f}")

    torch.save({
        "decoder": decoder.state_dict(),
        "feature_dim": feature_dim,
        "args": vars(args) | {
            "data_dir": str(args.data_dir),
            "feature_dir": str(args.feature_dir),
            "out_dir": str(args.out_dir),
        },
    }, args.out_dir / "viz_decoder.pt")
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nsaved in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
