"""Record a single evaluation rollout and save as a grid of frames with
action annotations. Shows EXACTLY what the model does step by step.

Example:
    python scripts/viz_eval_rollout.py \\
        --task NeedlePick \\
        --dynamics-ckpt checkpoints/pick/dynamics/dynamics.pt \\
        --subgoal-ckpt checkpoints/pick/subgoal/subgoal.pt \\
        --out checkpoints/rollout.png --planner backward --match-res 128
"""
import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from bsp_surgical.models.perception import PretrainedEncoder
from bsp_surgical.models.dynamics import InverseDynamics
from bsp_surgical.models.subgoal import SubgoalGenerator
from bsp_surgical.inference.planner import plan_subgoals, plan_subgoals_lerp, plan_no_subgoals


TASK_REGISTRY = {
    "NeedleReach": ("surrol.tasks.needle_reach_RL", "NeedleReach"),
    "NeedlePick": ("surrol.tasks.needle_pick_RL_2", "NeedlePickRL"),
}


def _encode(encoder, frame, match_res=None):
    if match_res is not None:
        import cv2
        frame = cv2.resize(frame, (match_res, match_res), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return encoder(t)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_REGISTRY), default="NeedlePick")
    parser.add_argument("--backbone", default="dinov2-base")
    parser.add_argument("--dynamics-ckpt", type=Path, required=True)
    parser.add_argument("--subgoal-ckpt", type=Path, required=True)
    parser.add_argument("--planner", choices=["backward", "forward", "lerp", "none"], default="none")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--num-subgoals", type=int, default=2)
    parser.add_argument("--match-res", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid-cols", type=int, default=6)
    args = parser.parse_args()

    dev = "mps"
    encoder = PretrainedEncoder(args.backbone, dev)
    dyn = torch.load(args.dynamics_ckpt, map_location=dev, weights_only=False)
    chunk = dyn.get("chunk_size", 1)
    inv = InverseDynamics(dyn["feature_dim"], 5, hidden=dyn["hidden"], chunk_size=chunk).to(dev)
    inv.load_state_dict(dyn["inverse"])
    inv.eval()
    sg = torch.load(args.subgoal_ckpt, map_location=dev, weights_only=False)
    subgoal = SubgoalGenerator(sg["feature_dim"], hidden=sg["hidden"]).to(dev)
    subgoal.load_state_dict(sg["subgoal"])
    subgoal.eval()

    mod, cls = TASK_REGISTRY[args.task]
    env = getattr(importlib.import_module(mod), cls)(render_mode=None)

    # 1) oracle goal capture
    np.random.seed(args.seed)
    obs = env.reset()
    for _ in range(100):
        a = env.get_oracle_action(obs)
        obs, _, done, info = env.step(a)
        if info.get("is_success") or done:
            break
    goal_frame = env.render("rgb_array")

    # 2) reset + plan
    np.random.seed(args.seed)
    env.reset()
    z_goal = _encode(encoder, goal_frame, args.match_res)
    z_start = _encode(encoder, env.render("rgb_array"), args.match_res)

    if args.planner == "backward":
        waypoints = plan_subgoals(subgoal, z_start, z_goal, args.num_subgoals)
    elif args.planner == "lerp":
        waypoints = plan_subgoals_lerp(z_start, z_goal, args.num_subgoals)
    elif args.planner == "forward":
        # same as backward in this script — not importing forward variant here
        waypoints = plan_subgoals(subgoal, z_start, z_goal, args.num_subgoals)
    else:
        waypoints = plan_no_subgoals(z_start, z_goal)

    # 3) execute & record
    records = []
    steps_per_wp = max(1, args.max_steps // len(waypoints))
    total = 0
    success = False
    for wp_idx, w in enumerate(waypoints):
        for _ in range(steps_per_wp):
            if total >= args.max_steps:
                break
            frame = env.render("rgb_array")
            z_now = _encode(encoder, frame, args.match_res)
            dist = torch.linalg.norm(z_now - w, dim=-1).item()
            dist_goal = torch.linalg.norm(z_now - z_goal, dim=-1).item()
            action = inv(z_now, w)
            if chunk > 1:
                a_chunk = action.squeeze(0).detach().cpu().numpy()
                a_step = a_chunk[0]
            else:
                a_step = action.squeeze(0).detach().cpu().numpy()
            a = np.clip(a_step, -1.0, 1.0)
            records.append({
                "frame": frame, "action": a.copy(),
                "dist_wp": dist, "dist_goal": dist_goal, "wp_idx": wp_idx,
            })
            _obs, _, done, info = env.step(a)
            total += 1
            if info.get("is_success"):
                success = True
                break
        if success:
            break

    # Save final frame too
    final_frame = env.render("rgb_array")
    records.append({"frame": final_frame, "action": None, "dist_wp": 0, "dist_goal": 0, "wp_idx": -1, "final": True})

    # Build figure
    n = len(records)
    cols = args.grid_cols
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            r = records[i]
            ax.imshow(r["frame"])
            if r.get("final"):
                ax.set_title(f"final  succ={success}")
            else:
                a = r["action"]
                ax.set_title(f"t={i} wp{r['wp_idx']} d={r['dist_wp']:.1f}\\n"
                             f"a=[{a[0]:.2f},{a[1]:.2f},{a[2]:.2f},{a[3]:.2f},{a[4]:.2f}]",
                             fontsize=7)
        ax.axis("off")

    fig.suptitle(f"{args.task} rollout | planner={args.planner} | success={success} | steps={total}")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"saved {args.out} (success={success}, steps={total})")


if __name__ == "__main__":
    main()
