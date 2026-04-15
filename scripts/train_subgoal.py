"""Train the backward subgoal generator on cached backbone features.

Loads (z_start, z_quarter, z_mid, z_end) per episode from a feature
cache, then trains an MLP h(z_now, z_target) -> z_mid with
dual-supervision loss: one term uses ground-truth midpoints,
the other feeds the model's own sg_1 prediction back in.

Example:
    python scripts/train_subgoal.py \\
        --data-dir data/needle_reach \\
        --feature-dir data/needle_reach/features/dinov2-base \\
        --out-dir checkpoints/subgoal --epochs 100 --hidden 1024
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bsp_surgical.models.subgoal import SubgoalGenerator
from bsp_surgical.models.losses import subgoal_dual_supervision_loss
from bsp_surgical.training.feature_dataset import FeatureSubgoalDataset


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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--random-windows", action="store_true",
                        help="Sample random windows (a,c) per episode per step instead of the "
                             "fixed (0, T/4, T/2, T) quadruple — gives the MLP many more triples.")
    parser.add_argument("--min-span", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-transitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    print(f"device: {device}")

    dataset = FeatureSubgoalDataset(
        args.data_dir, args.feature_dir,
        min_transitions=args.min_transitions,
        random_windows=args.random_windows,
        min_span=args.min_span,
    )
    feature_dim = dataset[0][0].shape[-1]  # sample any item; all feature tensors are (feature_dim,)
    print(f"dataset: {len(dataset)} episodes, feature_dim={feature_dim}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    mlp = SubgoalGenerator(latent_dim=feature_dim, hidden=args.hidden).to(device)
    print(f"trainable params: {sum(p.numel() for p in mlp.parameters()):,}")
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict] = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        mlp.train()
        ep_total = ep_gt = ep_pred = 0.0
        n = 0
        for batch in loader:
            z_start, z_quarter, z_mid, z_end = (t.to(device) for t in batch)
            total, parts = subgoal_dual_supervision_loss(
                mlp, z_start=z_start, z_quarter=z_quarter,
                z_mid=z_mid, z_end=z_end,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()
            ep_total += total.item()
            ep_gt += parts["gt"].item()
            ep_pred += parts["pred"].item()
            n += 1
            step += 1
            if step % args.log_every == 0:
                print(f"step {step:>5d}  gt={parts['gt'].item():.4f}  "
                      f"pred={parts['pred'].item():.4f}  total={total.item():.4f}")

        history.append({
            "epoch": epoch,
            "total": ep_total / n, "gt": ep_gt / n, "pred": ep_pred / n,
        })
        print(f"== epoch {epoch}: gt={ep_gt/n:.4f} pred={ep_pred/n:.4f} total={ep_total/n:.4f}")

    ckpt_path = args.out_dir / "subgoal.pt"
    torch.save({
        "subgoal": mlp.state_dict(),
        "feature_dim": feature_dim,
        "hidden": args.hidden,
        "args": vars(args) | {
            "data_dir": str(args.data_dir),
            "feature_dir": str(args.feature_dir),
            "out_dir": str(args.out_dir),
        },
    }, ckpt_path)
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nsaved {ckpt_path} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
