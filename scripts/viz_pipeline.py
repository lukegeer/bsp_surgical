"""Show a single raw frame going through the whole pipeline:
raw render → optional crop → resize to training res → DINOv2 preprocess
→ features → distance to a reference goal. A visual sanity check."""
import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from bsp_surgical.models.perception import PretrainedEncoder, preprocess_rgb_batch


TASK_REGISTRY = {
    "NeedleReach": ("surrol.tasks.needle_reach_RL", "NeedleReach"),
    "NeedlePick": ("surrol.tasks.needle_pick_RL_2", "NeedlePickRL"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_REGISTRY), default="NeedlePick")
    parser.add_argument("--backbone", default="dinov2-base")
    parser.add_argument("--crop", type=int, nargs=4, default=None)
    parser.add_argument("--match-res", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    enc = PretrainedEncoder(args.backbone, "mps")
    mod, cls = TASK_REGISTRY[args.task]
    env = getattr(importlib.import_module(mod), cls)(render_mode=None)
    np.random.seed(args.seed)
    env.reset()

    raw = env.render("rgb_array")
    print(f"raw: {raw.shape}")

    # Apply crop
    img = raw.copy()
    if args.crop is not None:
        y1, y2, x1, x2 = args.crop
        img = img[y1:y2, x1:x2]
        print(f"crop: {img.shape}")

    # Match training res if given
    if args.match_res is not None:
        import cv2
        img = cv2.resize(img, (args.match_res, args.match_res), interpolation=cv2.INTER_AREA)
        print(f"match_res: {img.shape}")

    # Final DINOv2 preprocess (resize to backbone input + normalize)
    t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    preprocessed = preprocess_rgb_batch(t.to("mps"), enc.spec.input_size)
    print(f"dinov2 input: {preprocessed.shape}")

    # Encode
    with torch.inference_mode():
        feat = enc(t)
    print(f"feature: {feat.shape}  norm={feat.norm().item():.2f}  mean={feat.mean().item():.3f}  std={feat.std().item():.3f}")

    # Plot pipeline stages
    # Denormalize preprocessed for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    pp_vis = (preprocessed.squeeze(0).cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    stages = [("raw render", raw)]
    if args.crop is not None:
        y1, y2, x1, x2 = args.crop
        crop_vis = raw[y1:y2, x1:x2]
        stages.append(("crop", crop_vis))
    if args.match_res is not None:
        stages.append((f"match-res {args.match_res}", img))
    stages.append((f"dinov2 input {enc.spec.input_size}", pp_vis))

    fig, axes = plt.subplots(1, len(stages) + 1, figsize=(4 * (len(stages) + 1), 4))
    for i, (title, im) in enumerate(stages):
        axes[i].imshow(im)
        axes[i].set_title(f"{title}\n{im.shape[:2]}")
        axes[i].axis("off")
    # Feature histogram
    axes[-1].hist(feat.squeeze(0).cpu().numpy(), bins=40)
    axes[-1].set_title(f"feature dist\nnorm={feat.norm().item():.1f}")

    fig.suptitle(f"{args.task} frame through pipeline")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
