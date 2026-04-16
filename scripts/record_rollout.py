"""Record an evaluation rollout as an MP4 you can actually watch.

Draws each frame at full resolution with step number, planner name,
current action, latent distance to goal, and success flag overlaid.

Example:
    python scripts/record_rollout.py \\
        --task NeedlePick \\
        --dynamics-ckpt checkpoints/pick_p/dynamics/dynamics.pt \\
        --subgoal-ckpt checkpoints/pick_p/subgoal/subgoal.pt \\
        --planner none --match-res 128 --binary-jaw --max-steps 60 \\
        --out viz/pick_p_rollout.mp4
"""
import argparse
import importlib
from pathlib import Path

import cv2
import numpy as np
import torch

from bsp_surgical.models.perception import PretrainedEncoder
from bsp_surgical.models.dynamics import InverseDynamics
from bsp_surgical.models.subgoal import SubgoalGenerator
from bsp_surgical.inference.planner import (
    plan_subgoals, plan_subgoals_lerp, plan_no_subgoals,
)


TASK_REGISTRY = {
    "NeedleReach": ("surrol.tasks.needle_reach_RL", "NeedleReach"),
    "NeedlePick": ("surrol.tasks.needle_pick_RL_2", "NeedlePickRL"),
    "GauzeRetrieve": ("surrol.tasks.gauze_retrieve_RL", "GauzeRetrieve"),
}


def _encode(encoder, frame, match_res=None):
    if match_res is not None:
        frame = cv2.resize(frame, (match_res, match_res), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return encoder(t)


def _overlay(frame, lines, y0=20, color=(255, 255, 255), bg=(0, 0, 0)):
    img = frame.copy()
    for i, line in enumerate(lines):
        y = y0 + i * 20
        cv2.rectangle(img, (8, y - 14), (8 + len(line) * 8, y + 6), bg, -1)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_REGISTRY), default="NeedlePick")
    parser.add_argument("--backbone", default="dinov2-base")
    parser.add_argument("--dynamics-ckpt", type=Path, required=True)
    parser.add_argument("--subgoal-ckpt", type=Path, required=True)
    parser.add_argument("--planner", choices=["backward", "lerp", "none"], default="none")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--num-subgoals", type=int, default=2)
    parser.add_argument("--match-res", type=int, default=None)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--binary-jaw", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dev = "mps"
    encoder = PretrainedEncoder(args.backbone, dev)
    dyn = torch.load(args.dynamics_ckpt, map_location=dev, weights_only=False)
    chunk = dyn.get("chunk_size", 1)
    proprio_dim = dyn.get("proprio_dim", 0)
    inv = InverseDynamics(
        dyn["feature_dim"], 5, hidden=dyn["hidden"],
        chunk_size=chunk, proprio_dim=proprio_dim,
    ).to(dev).eval()
    inv.load_state_dict(dyn["inverse"])

    sg_ckpt = torch.load(args.subgoal_ckpt, map_location=dev, weights_only=False)
    subgoal = SubgoalGenerator(sg_ckpt["feature_dim"], hidden=sg_ckpt["hidden"]).to(dev).eval()
    subgoal.load_state_dict(sg_ckpt["subgoal"])

    mod, cls = TASK_REGISTRY[args.task]
    env = getattr(importlib.import_module(mod), cls)(render_mode=None)

    # 1) Oracle goal capture
    np.random.seed(args.seed)
    obs = env.reset()
    for _ in range(150):
        a = env.get_oracle_action(obs)
        obs, _, done, info = env.step(a)
        if info.get("is_success") or done:
            break
    goal_frame = env.render("rgb_array")

    # 2) Reset + plan
    np.random.seed(args.seed)
    obs = env.reset()
    z_goal = _encode(encoder, goal_frame, args.match_res)
    z_start = _encode(encoder, env.render("rgb_array"), args.match_res)

    if args.planner == "backward":
        waypoints = plan_subgoals(subgoal, z_start, z_goal, args.num_subgoals)
    elif args.planner == "lerp":
        waypoints = plan_subgoals_lerp(z_start, z_goal, args.num_subgoals)
    else:
        waypoints = plan_no_subgoals(z_start, z_goal)

    # 3) Record rollout frames
    frames_out = []
    steps_per_wp = max(1, args.max_steps // len(waypoints))
    total = 0
    success = False
    final_jaw = 0.0
    final_action = np.zeros(5)
    final_d = 0.0

    # Opening title frame
    start_rendered = env.render("rgb_array")
    title = _overlay(start_rendered, [
        f"Task: {args.task}",
        f"Planner: {args.planner}   match_res={args.match_res}   binary_jaw={args.binary_jaw}",
        f"Max steps: {args.max_steps}   seed={args.seed}",
        f"Waypoints: {len(waypoints)} (incl. goal)",
    ], y0=30)
    for _ in range(args.fps):  # hold 1 sec
        frames_out.append(title)

    for wp_idx, w in enumerate(waypoints):
        for _ in range(steps_per_wp):
            if total >= args.max_steps:
                break
            frame = env.render("rgb_array")
            z_now = _encode(encoder, frame, args.match_res)
            d = torch.linalg.norm(z_now - w, dim=-1).item()
            d_goal = torch.linalg.norm(z_now - z_goal, dim=-1).item()

            p_now = None
            if proprio_dim > 0 and isinstance(obs, dict) and "observation" in obs:
                p_now = torch.from_numpy(np.asarray(obs["observation"], dtype=np.float32)).unsqueeze(0).to(dev)
            action = inv(z_now, w, proprio=p_now) if proprio_dim > 0 else inv(z_now, w)
            if chunk > 1:
                a_chunk = action.squeeze(0).detach().cpu().numpy()
                a_step = a_chunk[0]
            else:
                a_step = action.squeeze(0).detach().cpu().numpy()
            a = np.clip(a_step * args.action_scale, -1.0, 1.0)
            if args.binary_jaw:
                a[-1] = 1.0 if a[-1] > 0.0 else -1.0

            annotated = _overlay(frame, [
                f"step {total}   wp {wp_idx+1}/{len(waypoints)}",
                f"d_wp={d:.2f}  d_goal={d_goal:.2f}",
                f"a=[{a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},{a[3]:+.2f}]",
                f"jaw={a[4]:+.2f}   succ={bool(info.get('is_success', 0))}",
            ], y0=30)
            frames_out.append(annotated)

            obs, _, done, info = env.step(a)
            total += 1
            final_jaw = a[-1]
            final_action = a.copy()
            final_d = d_goal
            if info.get("is_success"):
                success = True
                break
        if success:
            break

    # Final closing frames
    final_rendered = env.render("rgb_array")
    end_title = _overlay(final_rendered, [
        f"FINAL  step {total}",
        f"Success: {success}",
        f"Last action: {np.round(final_action, 2).tolist()}",
        f"Last d_goal: {final_d:.2f}",
    ], y0=30, color=(0, 255, 0) if success else (255, 0, 0))
    for _ in range(args.fps * 2):  # hold 2 sec
        frames_out.append(end_title)

    # Write MP4
    args.out.parent.mkdir(parents=True, exist_ok=True)
    H, W = frames_out[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (W, H))
    for f in frames_out:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"saved {args.out}  ({len(frames_out)} frames, {len(frames_out)/args.fps:.1f}s at {args.fps} fps)")
    print(f"success={success} total_env_steps={total}")


if __name__ == "__main__":
    main()
