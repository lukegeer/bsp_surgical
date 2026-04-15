"""Evaluate a trained BSP policy against baselines on NeedleReach.

For each eval episode, seed-matched trick: the same numpy seed is used
twice to env.reset(), so the oracle's final frame (captured in the first
reset) is a valid goal image for the second reset. The env's internal
is_success reports ground-truth success.

Example:
    python scripts/evaluate.py \\
        --task NeedleReach --num-episodes 50 \\
        --phase2-ckpt checkpoints/phase2/phase2.pt \\
        --phase3-ckpt checkpoints/phase3/phase3.pt \\
        --out-dir eval/run_0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bsp_surgical.models.vae import VAE
from bsp_surgical.models.dynamics import InverseDynamics
from bsp_surgical.models.subgoal import SubgoalGenerator
from bsp_surgical.inference.planner import (
    plan_subgoals,
    plan_subgoals_lerp,
    plan_no_subgoals,
    execute_plan,
)


TASK_REGISTRY = {
    "NeedleReach": ("surrol.tasks.needle_reach_RL", "NeedleReach"),
    "NeedlePick": ("surrol.tasks.needle_pick_RL", "NeedlePick"),
}


def _make_env(task: str):
    import importlib

    mod, cls = TASK_REGISTRY[task]
    return getattr(importlib.import_module(mod), cls)(render_mode=None)


def _resolve_device(request: str) -> torch.device:
    if request == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(request)


def _capture_goal_image(env, max_oracle_steps: int) -> np.ndarray:
    """Run the oracle until success, return the final frame."""
    obs = env.reset()
    final_frame = env.render("rgb_array")
    for _ in range(max_oracle_steps):
        a = env.get_oracle_action(obs)
        obs, _r, done, info = env.step(a)
        final_frame = env.render("rgb_array")
        if info.get("is_success") or done:
            break
    return final_frame


def _run_planner(env, vae, inverse, planner_fn_at_goal, max_steps: int) -> tuple[bool, int]:
    """Planner callbacks produce waypoints; we execute them.

    Returns (success, total_steps).
    """
    total_steps = 0
    for _ in range(max_steps):
        z_now = _encode(vae, env.render("rgb_array"))
        # The planner_fn_at_goal closes over z_goal already; pass only z_now.
        waypoints = planner_fn_at_goal(z_now)
        # Step once toward the first waypoint
        target = waypoints[0]
        action = inverse(z_now, target)
        obs, _r, done, info = env.step(action.squeeze(0).detach().cpu().numpy())
        total_steps += 1
        if info.get("is_success"):
            return True, total_steps
        if done:
            return False, total_steps
    return False, total_steps


def _encode(vae, frame: np.ndarray) -> torch.Tensor:
    device = next(vae.parameters()).device
    t = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t = t.to(device)
    mu, _logvar = vae.encode(t)
    return mu


def _resize(frame: np.ndarray, resolution: int) -> np.ndarray:
    if frame.shape[0] == resolution and frame.shape[1] == resolution:
        return frame
    import cv2

    return cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_REGISTRY), default="NeedleReach")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-oracle-steps", type=int, default=100)
    parser.add_argument("--phase2-ckpt", type=Path, required=True)
    parser.add_argument("--phase3-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--planners", nargs="+",
        default=["backward", "lerp", "none"],
        choices=["backward", "lerp", "none"],
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    # Load Phase 2
    p2 = torch.load(args.phase2_ckpt, map_location=device, weights_only=False)
    p2a = p2["args"]
    vae = VAE(latent_dim=p2a["latent_dim"], resolution=p2a["resolution"]).to(device)
    vae.load_state_dict(p2["vae"])
    vae.eval()
    inverse = InverseDynamics(latent_dim=p2a["latent_dim"], action_dim=5).to(device)
    inverse.load_state_dict(p2["inverse"])
    inverse.eval()
    resolution = p2a["resolution"]

    # Load Phase 3
    p3 = torch.load(args.phase3_ckpt, map_location=device, weights_only=False)
    subgoal = SubgoalGenerator(latent_dim=p3["latent_dim"], hidden=p3["hidden"]).to(device)
    subgoal.load_state_dict(p3["subgoal"])
    subgoal.eval()

    env = _make_env(args.task)

    results: dict[str, dict] = {}
    for planner_name in args.planners:
        print(f"\n=== planner: {planner_name} ===")
        successes = []
        steps_per_ep = []
        t0 = time.time()
        for i in range(args.num_episodes):
            # 1) oracle run → goal image
            np.random.seed(args.seed + i)
            goal_frame = _capture_goal_image(env, args.max_oracle_steps)
            goal_frame_r = _resize(goal_frame, resolution)

            # 2) reset with same seed → same start state
            np.random.seed(args.seed + i)
            env.reset()
            z_goal = _encode(vae, goal_frame_r)

            if planner_name == "backward":
                planner_fn = lambda z_now, z_goal=z_goal: plan_subgoals(subgoal, z_now, z_goal, 2)
            elif planner_name == "lerp":
                planner_fn = lambda z_now, z_goal=z_goal: plan_subgoals_lerp(z_now, z_goal, 2)
            else:  # none
                planner_fn = lambda z_now, z_goal=z_goal: plan_no_subgoals(z_now, z_goal)

            success, steps = _run_planner(env, vae, inverse, planner_fn, args.max_steps)
            successes.append(success)
            steps_per_ep.append(steps)

        dt = time.time() - t0
        rate = float(np.mean(successes))
        mean_steps = float(np.mean(steps_per_ep))
        print(f"  success: {sum(successes)}/{args.num_episodes} = {rate:.1%}")
        print(f"  mean steps (episode): {mean_steps:.1f}")
        print(f"  wallclock: {dt:.1f}s")
        results[planner_name] = {
            "success_rate": rate,
            "mean_steps": mean_steps,
            "per_episode_success": successes,
            "per_episode_steps": steps_per_ep,
        }

    with open(args.out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
