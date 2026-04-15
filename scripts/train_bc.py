"""Behavioral cloning baseline: (img_t) -> action_t, supervised by the
oracle action from the collected trajectories."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bsp_surgical.models.bc import BCPolicy
from bsp_surgical.training.dataset import TransitionDataset


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    print(f"device: {device}")

    dataset = TransitionDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"dataset: {len(dataset)} transitions")

    policy = BCPolicy(resolution=args.resolution, action_dim=5).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_loss = 0.0
        n = 0
        for img_t, action, _img_next in loader:
            img_t = img_t.to(device)
            action = action.to(device)
            a_pred = policy(img_t)
            # Smooth-L1 on continuous dims + BCE on jaw, matching the inverse model
            cont_loss = F.smooth_l1_loss(a_pred[:, :-1], action[:, :-1])
            jaw_loss = F.binary_cross_entropy_with_logits(a_pred[:, -1], action[:, -1])
            loss = cont_loss + 0.01 * jaw_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n += 1
            step += 1
            if step % args.log_every == 0:
                print(f"step {step:>6d}  loss={loss.item():.4f}  "
                      f"({step/(time.time()-t0):.1f} step/s)")

        history.append({"epoch": epoch, "loss": ep_loss / n})
        print(f"== epoch {epoch}: loss={ep_loss/n:.4f}")

    torch.save({"bc": policy.state_dict(), "args": vars(args) | {
        "data_dir": str(args.data_dir), "out_dir": str(args.out_dir),
    }}, args.out_dir / "bc.pt")
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved {args.out_dir / 'bc.pt'}")


if __name__ == "__main__":
    main()
