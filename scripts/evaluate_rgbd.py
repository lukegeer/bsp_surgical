"""Evaluate the RGBD+Seg+Depth pipeline with diffusion backward subgoals.

Supports the compositional experiment: load a model trained on simple
tasks (e.g., needle_reach_sd + needle_pick_sd) and evaluate zero-shot
on a harder task (e.g., gauze_retrieve_sd)."""
import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import pybullet as p

from bsp_surgical.models.rgbd_encoder import RGBDSegEncoder
from bsp_surgical.models.dynamics import InverseDynamics
from bsp_surgical.models.subgoal_diffusion import SubgoalDiffusion
from bsp_surgical.models.segdepth_encoder import seg_to_onehot


TASK_REGISTRY = {
    "NeedleReach": ("surrol.tasks.needle_reach_RL", "NeedleReach"),
    "NeedlePick": ("surrol.tasks.needle_pick_RL_2", "NeedlePickRL"),
    "GauzeRetrieve": ("surrol.tasks.gauze_retrieve_RL", "GauzeRetrieve"),
}


def _device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device(req)


def _render_rgbd(env, resolution: int):
    """Returns (rgb 128x128, seg 128x128, depth 128x128) as numpy."""
    import cv2
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
    return rgb_r, seg_r, depth_r


def _encode(encoder, rgb, seg, depth, device, num_seg_channels):
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    seg_t = torch.from_numpy(seg_to_onehot(seg, num_seg_channels)).unsqueeze(0).to(device)
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)
    return encoder(rgb_t, seg_t, depth_t)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_REGISTRY), required=True)
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--max-oracle-steps", type=int, default=150)
    parser.add_argument("--dynamics-ckpt", type=Path, required=True)
    parser.add_argument("--diffusion-ckpt", type=Path, required=False,
                        help="If omitted, uses 'none' planner only (direct inverse to goal)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-subgoals", type=int, default=2)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--binary-jaw", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--planners", nargs="+", default=["none", "backward"],
                        choices=["none", "backward"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    print(f"device: {device}")

    # Load encoder + inverse
    dyn = torch.load(args.dynamics_ckpt, map_location=device, weights_only=False)
    num_seg = dyn["num_seg_channels"]
    encoder = RGBDSegEncoder(
        num_seg_channels=num_seg, resolution=128, feature_dim=dyn["feature_dim"],
    ).to(device)
    encoder.load_state_dict(dyn["encoder"])
    encoder.eval()
    inverse = InverseDynamics(
        latent_dim=dyn["feature_dim"], action_dim=5,
        hidden=dyn["inverse_hidden"], chunk_size=dyn["chunk_size"],
    ).to(device)
    inverse.load_state_dict(dyn["inverse"])
    inverse.eval()
    chunk = dyn["chunk_size"]
    print(f"loaded encoder feature_dim={dyn['feature_dim']}, inverse chunk={chunk}")

    diffusion = None
    if "backward" in args.planners:
        if args.diffusion_ckpt is None:
            raise ValueError("--diffusion-ckpt required when 'backward' planner is requested")
        diff_ckpt = torch.load(args.diffusion_ckpt, map_location=device, weights_only=False)
        diffusion = SubgoalDiffusion(
            latent_dim=diff_ckpt["feature_dim"], hidden=diff_ckpt["hidden"],
            num_timesteps=diff_ckpt["num_timesteps"],
        ).to(device)
        diffusion.load_state_dict(diff_ckpt["diffusion"])
        diffusion.eval()
        print(f"loaded diffusion num_timesteps={diff_ckpt['num_timesteps']}")

    # Env
    mod, cls = TASK_REGISTRY[args.task]
    env = getattr(importlib.import_module(mod), cls)(render_mode=None)

    results: dict[str, dict] = {}
    for planner_name in args.planners:
        print(f"\n=== planner: {planner_name} ===")
        successes, step_counts = [], []
        t0 = time.time()
        for i in range(args.num_episodes):
            # Capture goal via oracle
            np.random.seed(args.seed + i)
            obs = env.reset()
            for _ in range(args.max_oracle_steps):
                a = env.get_oracle_action(obs)
                obs, _, done, info = env.step(a)
                if info.get("is_success") or done:
                    break
            goal_rgb, goal_seg, goal_depth = _render_rgbd(env, 128)

            # Reset + plan
            np.random.seed(args.seed + i)
            obs = env.reset()
            z_goal = _encode(encoder, goal_rgb, goal_seg, goal_depth, device, num_seg)
            rgb0, seg0, depth0 = _render_rgbd(env, 128)
            z_start = _encode(encoder, rgb0, seg0, depth0, device, num_seg)

            if planner_name == "backward":
                waypoints = diffusion.backward_bisect(
                    z_start, z_goal, num_subgoals=args.num_subgoals,
                    num_inference_steps=args.inference_steps,
                )
            else:  # none
                waypoints = [z_goal]

            # Execute
            steps_per_wp = max(1, args.max_steps // len(waypoints))
            total = 0
            success = False
            for w in waypoints:
                steps_this = 0
                while steps_this < steps_per_wp and total < args.max_steps:
                    rgb, seg, depth = _render_rgbd(env, 128)
                    z_now = _encode(encoder, rgb, seg, depth, device, num_seg)
                    with torch.no_grad():
                        action = inverse(z_now, w)
                    if chunk > 1:
                        chunk_np = action.squeeze(0).cpu().numpy()
                    else:
                        chunk_np = action.cpu().numpy().reshape(1, -1)
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
                if success or total >= args.max_steps:
                    break
            successes.append(success)
            step_counts.append(total)

        dt = time.time() - t0
        rate = float(np.mean(successes))
        print(f"  success: {sum(successes)}/{args.num_episodes} = {rate:.1%}")
        print(f"  mean steps: {float(np.mean(step_counts)):.1f}")
        print(f"  wallclock: {dt:.1f}s")
        results[planner_name] = {
            "success_rate": rate, "mean_steps": float(np.mean(step_counts)),
            "per_episode_success": successes, "per_episode_steps": step_counts,
        }

    with open(args.out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
