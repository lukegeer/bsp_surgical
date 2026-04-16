"""Visualize what DINOv2 actually attends to on surgical frames, at
fine spatial resolution.

At 224×224 input: 16×16 patch grid → each patch spans ~40 rendered
pixels, larger than a needle. Useless. At 896×896 input: 64×64 patch
grid → ~10 pixels per patch, needle-scale. We extract the last-block
attention weights (CLS queries every patch) and upsample with bicubic
interpolation for a readable heatmap.

Example:
    python scripts/viz_dinov2_attention.py --task NeedlePick \\
        --input-size 896 --out viz/pick_attention_hires.png
"""
import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

from bsp_surgical.models.perception import preprocess_rgb_batch


TASK_REGISTRY = {
    "NeedleReach": ("surrol.tasks.needle_reach_RL", "NeedleReach"),
    "NeedlePick": ("surrol.tasks.needle_pick_RL_2", "NeedlePickRL"),
    "GauzeRetrieve": ("surrol.tasks.gauze_retrieve_RL", "GauzeRetrieve"),
}


def _get_attention(model, x: torch.Tensor) -> torch.Tensor:
    """Return (num_patches,) attention weights from CLS to all patches
    in the last self-attention block. Uses the average across heads."""
    # Grab the last encoder layer's attention module
    last_layer = model.encoder.layer[-1]
    last_attn = last_layer.attention.attention  # (query, key, value)

    attn_weights = {}

    def hook(module, inputs, outputs):
        # Re-run q, k with the hook's inputs to get attention weights
        hidden = inputs[0]  # (B, seq_len, dim)
        mixed_q = last_attn.query(hidden)
        mixed_k = last_attn.key(hidden)
        q = last_attn.transpose_for_scores(mixed_q)  # (B, heads, seq, head_dim)
        k = last_attn.transpose_for_scores(mixed_k)
        scale = 1.0 / (q.shape[-1] ** 0.5)
        attn = (q @ k.transpose(-1, -2)) * scale
        attn = attn.softmax(dim=-1)  # (B, heads, seq, seq)
        attn_weights["w"] = attn

    handle = last_attn.register_forward_hook(hook)
    with torch.inference_mode():
        model(x)
    handle.remove()

    attn = attn_weights["w"]  # (1, heads, seq, seq)
    # CLS is token index 0; attn[:,:,0,1:] = CLS -> patches
    cls_to_patches = attn[0, :, 0, 1:]  # (heads, P)
    return cls_to_patches.mean(dim=0)     # avg over heads -> (P,)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_REGISTRY), default="NeedlePick")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=896,
                        help="Resize raw render to this size before DINOv2 (must be multiple of 14). "
                             "896 gives a 64x64 patch grid.")
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert args.input_size % 14 == 0, "must be multiple of 14"
    grid_side = args.input_size // 14
    print(f"DINOv2 input {args.input_size} -> patch grid {grid_side}x{grid_side}")

    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-base").to("mps").eval()

    mod, cls = TASK_REGISTRY[args.task]
    env = getattr(importlib.import_module(mod), cls)(render_mode=None)
    np.random.seed(args.seed)
    obs = env.reset()

    frames = [env.render("rgb_array")]
    n_steps = 100 if args.task in ("NeedlePick", "GauzeRetrieve") else 40
    for _ in range(n_steps):
        a = env.get_oracle_action(obs)
        obs, _, done, info = env.step(a)
        frames.append(env.render("rgb_array"))
        if info.get("is_success") or done:
            break
    total = len(frames)
    indices = np.linspace(0, total - 1, args.num_steps).astype(int)

    fig, axes = plt.subplots(2, args.num_steps, figsize=(5 * args.num_steps, 7))
    if args.num_steps == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(indices):
        frame = frames[idx]
        H, W = frame.shape[:2]
        x = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to("mps") / 255.0
        x = preprocess_rgb_batch(x, args.input_size)

        attn = _get_attention(model, x)        # (P,)
        heat = attn.reshape(grid_side, grid_side).cpu().float().numpy()

        # Normalize heatmap for display (per-image, so different frames are comparable within-image)
        heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

        # Bicubic upsample to frame size (smooth, not blocky)
        heat_up = cv2.resize(heat_norm, (W, H), interpolation=cv2.INTER_CUBIC)

        axes[0, col].imshow(frame)
        axes[0, col].set_title(f"frame t={idx}", fontsize=11)
        axes[0, col].axis("off")

        axes[1, col].imshow(frame)
        axes[1, col].imshow(heat_up, cmap="jet", alpha=0.55)
        axes[1, col].set_title(f"last-block CLS attention", fontsize=11)
        axes[1, col].axis("off")

    fig.suptitle(
        f"DINOv2-base true attention on {args.task} (input {args.input_size}, "
        f"grid {grid_side}x{grid_side}, ~{W/grid_side:.0f}px/patch in render)",
        fontsize=12,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
