"""Visualize what DINOv2 attends to on surgical frames.

Computes similarity between the CLS token and every patch token, then
overlays as a heatmap. Shows whether DINOv2 is focusing on the needle,
the gripper, the tray, or the background.

Example:
    python scripts/viz_dinov2_attention.py --task NeedlePick --out checkpoints/attention.png
"""
import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from bsp_surgical.models.perception import _BACKBONES, preprocess_rgb_batch


TASK_REGISTRY = {
    "NeedleReach": ("surrol.tasks.needle_reach_RL", "NeedleReach"),
    "NeedlePick": ("surrol.tasks.needle_pick_RL_2", "NeedlePickRL"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_REGISTRY), default="NeedlePick")
    parser.add_argument("--backbone", default="dinov2-base")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=4,
                        help="How many points along the oracle trajectory to visualize")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoModel
    spec = _BACKBONES[args.backbone]
    model = AutoModel.from_pretrained(spec.hf_name).to("mps").eval()

    mod, cls = TASK_REGISTRY[args.task]
    env = getattr(importlib.import_module(mod), cls)(render_mode=None)
    np.random.seed(args.seed)
    obs = env.reset()

    # Collect frames along the oracle trajectory
    frames = []
    frames.append(env.render("rgb_array"))
    n_steps = 100 if args.task == "NeedlePick" else 40
    for t in range(n_steps):
        a = env.get_oracle_action(obs)
        obs, _, done, info = env.step(a)
        frames.append(env.render("rgb_array"))
        if info.get("is_success") or done:
            break
    # Pick evenly-spaced frames
    total = len(frames)
    indices = np.linspace(0, total - 1, args.num_steps).astype(int)

    fig, axes = plt.subplots(2, args.num_steps, figsize=(4 * args.num_steps, 6))
    if args.num_steps == 1:
        axes = axes.reshape(2, 1)

    with torch.inference_mode():
        for col, idx in enumerate(indices):
            frame = frames[idx]
            H, W = frame.shape[:2]
            x = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to("mps") / 255.0
            x = preprocess_rgb_batch(x, spec.input_size)
            out = model(x)
            tokens = out.last_hidden_state  # (1, 1+P, dim)
            cls = tokens[:, 0]              # (1, dim)
            patches = tokens[:, 1:]          # (1, P, dim)
            # cosine similarity between CLS and each patch
            sim = F.cosine_similarity(cls.unsqueeze(1), patches, dim=-1).squeeze(0)  # (P,)
            grid_side = int(np.sqrt(sim.shape[0]))
            heat = sim.reshape(grid_side, grid_side).cpu().numpy()
            # upsample to frame size for overlay
            heat_up = np.kron(heat, np.ones((H // grid_side + 1, W // grid_side + 1)))[:H, :W]

            axes[0, col].imshow(frame)
            axes[0, col].set_title(f"frame t={idx}")
            axes[0, col].axis("off")
            axes[1, col].imshow(frame)
            axes[1, col].imshow(heat_up, cmap="jet", alpha=0.5)
            axes[1, col].set_title(f"CLS-patch sim")
            axes[1, col].axis("off")

    fig.suptitle(f"DINOv2 ({args.backbone}) attention on {args.task}", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
