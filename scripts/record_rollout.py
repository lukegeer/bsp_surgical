"""Record an evaluation rollout (new RGBDSeg + diffusion pipeline) as
an animated GIF you can actually watch.

Each frame is the live env render, overlaid with step number, current
action, latent distance to goal, success flag. PNG frames are also
dumped alongside so you can flip through them in an image viewer.

Example:
    python scripts/record_rollout.py \\
        --task NeedlePick \\
        --dynamics-ckpt checkpoints/pick_sd/dynamics/rgbd_dynamics.pt \\
        --diffusion-ckpt checkpoints/pick_sd/subgoal/subgoal_diffusion.pt \\
        --planner backward --binary-jaw --max-steps 80 \\
        --out viz/pick_sd_rollout.gif
"""
import argparse
import importlib
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import torch

from bsp_surgical.models.rgbd_encoder import RGBDSegEncoder, seg_to_onehot
from bsp_surgical.models.dynamics import InverseDynamics
from bsp_surgical.models.subgoal_diffusion import SubgoalDiffusion


TASK_REGISTRY = {
    "NeedleReach": ("surrol.tasks.needle_reach_RL", "NeedleReach"),
    "NeedlePick": ("surrol.tasks.needle_pick_RL_2", "NeedlePickRL"),
    "GauzeRetrieve": ("surrol.tasks.gauze_retrieve_RL", "GauzeRetrieve"),
}


def _render_rgbd(env, resolution: int):
    rgb = env.render("rgb_array")
    h, w = rgb.shape[:2]
    _, _, _, depth, seg = p.getCameraImage(
        width=w, height=h,
        viewMatrix=env._view_matrix, projectionMatrix=env._proj_matrix,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        renderer=p.ER_TINY_RENDERER,
    )
    depth = np.asarray(depth).reshape(h, w).astype(np.float32)
    seg = np.asarray(seg).reshape(h, w).astype(np.int32)
    rgb_r = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_AREA)
    seg_r = cv2.resize(seg, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    depth_r = cv2.resize(depth, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return rgb, rgb_r, seg_r, depth_r


def _encode(encoder, rgb, seg, depth, device, num_seg):
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    seg_t = torch.from_numpy(seg_to_onehot(seg, num_seg)).unsqueeze(0).to(device)
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)
    return encoder(rgb_t, seg_t, depth_t)


def _overlay(frame, lines, color=(255, 255, 255), bg=(0, 0, 0)):
    img = frame.copy()
    y0 = 24
    for i, line in enumerate(lines):
        y = y0 + i * 22
        cv2.rectangle(img, (8, y - 16), (8 + len(line) * 9, y + 4), bg, -1)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_REGISTRY), required=True)
    parser.add_argument("--dynamics-ckpt", type=Path, required=True)
    parser.add_argument("--diffusion-ckpt", type=Path, required=False)
    parser.add_argument("--planner", choices=["none", "backward"], default="none")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--num-subgoals", type=int, default=2)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--binary-jaw", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dev = "mps"
    dyn = torch.load(args.dynamics_ckpt, map_location=dev, weights_only=False)
    num_seg = dyn["num_seg_channels"]
    encoder = RGBDSegEncoder(
        num_seg_channels=num_seg, resolution=128, feature_dim=dyn["feature_dim"],
    ).to(dev)
    encoder.load_state_dict(dyn["encoder"])
    encoder.eval()
    inv = InverseDynamics(
        latent_dim=dyn["feature_dim"], action_dim=5,
        hidden=dyn["inverse_hidden"], chunk_size=dyn["chunk_size"],
    ).to(dev)
    inv.load_state_dict(dyn["inverse"])
    inv.eval()
    chunk = dyn["chunk_size"]

    diffusion = None
    if args.planner == "backward":
        assert args.diffusion_ckpt is not None, "--diffusion-ckpt required"
        dc = torch.load(args.diffusion_ckpt, map_location=dev, weights_only=False)
        diffusion = SubgoalDiffusion(
            latent_dim=dc["feature_dim"], hidden=dc["hidden"],
            num_timesteps=dc["num_timesteps"],
        ).to(dev)
        diffusion.load_state_dict(dc["diffusion"])
        diffusion.eval()

    mod, cls = TASK_REGISTRY[args.task]
    env = getattr(importlib.import_module(mod), cls)(render_mode=None)

    # 1) oracle goal
    np.random.seed(args.seed)
    obs = env.reset()
    for _ in range(150):
        a = env.get_oracle_action(obs)
        obs, _, done, info = env.step(a)
        if info.get("is_success") or done:
            break
    _, goal_rgb, goal_seg, goal_depth = _render_rgbd(env, 128)

    # 2) reset + plan
    np.random.seed(args.seed)
    obs = env.reset()
    z_goal = _encode(encoder, goal_rgb, goal_seg, goal_depth, dev, num_seg)
    full0, rgb0, seg0, depth0 = _render_rgbd(env, 128)
    z_start = _encode(encoder, rgb0, seg0, depth0, dev, num_seg)

    if args.planner == "backward":
        waypoints = diffusion.backward_bisect(
            z_start, z_goal, num_subgoals=args.num_subgoals,
            num_inference_steps=args.inference_steps,
        )
    else:
        waypoints = [z_goal]

    # Opening title
    frames_out = []
    title = _overlay(full0, [
        f"Task: {args.task}",
        f"Planner: {args.planner}  binary_jaw={args.binary_jaw}",
        f"Max steps: {args.max_steps}  seed={args.seed}",
        f"Waypoints: {len(waypoints)}",
    ])
    for _ in range(args.fps):
        frames_out.append(title)

    # Execute
    steps_per_wp = max(1, args.max_steps // len(waypoints))
    total = 0
    success = False
    for wp_idx, w in enumerate(waypoints):
        steps_this = 0
        while steps_this < steps_per_wp and total < args.max_steps:
            full, rgb, seg, depth = _render_rgbd(env, 128)
            z_now = _encode(encoder, rgb, seg, depth, dev, num_seg)
            d_wp = torch.linalg.norm(z_now - w, dim=-1).item()
            d_goal = torch.linalg.norm(z_now - z_goal, dim=-1).item()
            with torch.no_grad():
                action = inv(z_now, w)
            if chunk > 1:
                chunk_np = action.squeeze(0).cpu().numpy()
            else:
                chunk_np = action.cpu().numpy().reshape(1, -1)
            a_first = chunk_np[0]
            a_display = np.clip(a_first * args.action_scale, -1.0, 1.0)
            if args.binary_jaw:
                a_display[-1] = 1.0 if a_display[-1] > 0 else -1.0
            frames_out.append(_overlay(full, [
                f"step {total}   wp {wp_idx+1}/{len(waypoints)}",
                f"d_wp={d_wp:.2f}  d_goal={d_goal:.2f}",
                f"a=[{a_display[0]:+.2f},{a_display[1]:+.2f},{a_display[2]:+.2f},{a_display[3]:+.2f}]",
                f"jaw={a_display[4]:+.2f}",
            ]))
            for a_step in chunk_np:
                a = np.clip(a_step * args.action_scale, -1.0, 1.0)
                if args.binary_jaw:
                    a[-1] = 1.0 if a[-1] > 0 else -1.0
                obs, _, done, info = env.step(a)
                total += 1
                steps_this += 1
                if info.get("is_success"):
                    success = True; break
                if done or total >= args.max_steps:
                    break
            if success or total >= args.max_steps:
                break
        if success:
            break

    # Final title
    full_final, *_ = _render_rgbd(env, 128)
    end = _overlay(full_final, [
        f"FINAL  step {total}",
        f"Success: {success}",
    ], color=(0, 255, 0) if success else (255, 0, 0))
    for _ in range(args.fps * 2):
        frames_out.append(end)

    # Write GIF (reliable) + PNG frames (fallback)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    if str(args.out).lower().endswith(".gif"):
        imageio.mimsave(str(args.out), frames_out, fps=args.fps)
    else:
        imageio.mimsave(
            str(args.out), frames_out, fps=args.fps,
            codec="libx264", pixelformat="yuv420p", quality=8,
        )
    png_dir = args.out.with_suffix("")
    png_dir.mkdir(exist_ok=True)
    for i, f in enumerate(frames_out):
        imageio.imwrite(str(png_dir / f"frame_{i:04d}.png"), f)

    print(f"saved {args.out}  ({len(frames_out)} frames, {len(frames_out)/args.fps:.1f}s at {args.fps} fps)")
    print(f"  + PNG frames in {png_dir}/")
    print(f"success={success} total_env_steps={total}")


if __name__ == "__main__":
    main()
