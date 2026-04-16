"""Evaluate the feature-based pipeline (frozen DINOv2 + trained MLPs) on
SurRoL, comparing backward/lerp/none/forward planners.

Differences from scripts/evaluate.py (old VAE pipeline):
  - Uses a PretrainedEncoder for perception.
  - Loads dynamics + subgoal checkpoints separately.
  - Supports an additional 'forward' planner variant (flipped bisection
    direction) for the central novelty comparison.

Example:
    python scripts/evaluate_features.py \\
        --task NeedleReach --num-episodes 30 \\
        --backbone dinov2-base \\
        --dynamics-ckpt checkpoints/dynamics/dynamics.pt \\
        --subgoal-ckpt checkpoints/subgoal/subgoal.pt \\
        --out-dir eval/feat_run_0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bsp_surgical.models.dynamics import InverseDynamics
from bsp_surgical.models.subgoal import SubgoalGenerator
from bsp_surgical.models.perception import PretrainedEncoder
from bsp_surgical.inference.planner import (
    plan_subgoals,
    plan_subgoals_lerp,
    plan_no_subgoals,
)


TASK_REGISTRY = {
    "NeedleReach": ("surrol.tasks.needle_reach_RL", "NeedleReach"),
    "NeedlePick": ("surrol.tasks.needle_pick_RL_2", "NeedlePickRL"),
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


def _encode_frame(encoder: PretrainedEncoder, frame: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return encoder(t)


def _capture_goal_image(env, max_oracle_steps: int) -> np.ndarray:
    obs = env.reset()
    final = env.render("rgb_array")
    for _ in range(max_oracle_steps):
        a = env.get_oracle_action(obs)
        obs, _r, done, info = env.step(a)
        final = env.render("rgb_array")
        if info.get("is_success") or done:
            break
    return final


def plan_subgoals_forward(
    subgoal_mlp: torch.nn.Module,
    z_now: torch.Tensor,
    z_goal: torch.Tensor,
    num_subgoals: int = 2,
) -> list[torch.Tensor]:
    """Forward-bisection variant: applies the SAME subgoal MLP, but
    bisects outward from z_now rather than backward from z_goal.

        sg_1 = h(z_now, z_goal)   (midpoint, same as backward)
        sg_2 = h(sg_1, z_goal)    (three-quarter point)
        waypoints = [sg_1, sg_2, z_goal]
    """
    with torch.no_grad():
        subgoals = []
        current = z_now
        for _ in range(num_subgoals):
            sg = subgoal_mlp(current, z_goal)
            subgoals.append(sg)
            current = sg
        return subgoals + [z_goal]


def _run_planner(
    env, encoder, inverse, plan_fn,
    *, max_steps: int, epsilon: float,
) -> tuple[bool, int]:
    z_now_init = _encode_frame(encoder, env.render("rgb_array"))
    waypoints = plan_fn(z_now_init)
    steps_per_wp = max(1, max_steps // len(waypoints))
    chunk_size = getattr(inverse, "chunk_size", 1)
    total = 0
    for w in waypoints:
        steps_this_wp = 0
        while steps_this_wp < steps_per_wp and total < max_steps:
            z_now = _encode_frame(encoder, env.render("rgb_array"))
            if torch.linalg.norm(z_now - w, dim=-1).item() <= epsilon:
                break
            action = inverse(z_now, w)
            if chunk_size > 1:
                # shape (1, K, action_dim) -> (K, action_dim)
                chunk_np = action.squeeze(0).detach().cpu().numpy()
            else:
                chunk_np = action.detach().cpu().numpy()[None]  # (1, action_dim)

            # Execute the whole chunk (or until success / done / step budget)
            for a_step in chunk_np:
                a = np.clip(a_step, -1.0, 1.0)
                _obs, _r, done, info = env.step(a)
                total += 1
                steps_this_wp += 1
                if info.get("is_success"):
                    return True, total
                if done or total >= max_steps:
                    return False, total
    return False, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_REGISTRY), default="NeedleReach")
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-oracle-steps", type=int, default=100)
    parser.add_argument("--backbone", default="dinov2-base")
    parser.add_argument("--dynamics-ckpt", type=Path, required=True)
    parser.add_argument("--subgoal-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.5,
                        help="L2 latent distance to declare a waypoint reached")
    parser.add_argument("--num-subgoals", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--planners", nargs="+",
        default=["backward", "forward", "lerp", "none"],
        choices=["backward", "forward", "lerp", "none"],
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    print(f"device: {device}")

    encoder = PretrainedEncoder(name=args.backbone, device=device)
    feature_dim = encoder.feature_dim
    print(f"backbone: {args.backbone} (feature_dim={feature_dim})")

    # inverse dynamics
    dyn = torch.load(args.dynamics_ckpt, map_location=device, weights_only=False)
    chunk_size = dyn.get("chunk_size", 1)
    inverse = InverseDynamics(
        latent_dim=dyn["feature_dim"], action_dim=5, hidden=dyn["hidden"],
        chunk_size=chunk_size,
    ).to(device)
    inverse.load_state_dict(dyn["inverse"])
    inverse.eval()
    print(f"inverse chunk_size={chunk_size}")

    # subgoal
    sg_ckpt = torch.load(args.subgoal_ckpt, map_location=device, weights_only=False)
    subgoal = SubgoalGenerator(
        latent_dim=sg_ckpt["feature_dim"], hidden=sg_ckpt["hidden"],
    ).to(device)
    subgoal.load_state_dict(sg_ckpt["subgoal"])
    subgoal.eval()

    env = _make_env(args.task)

    results: dict[str, dict] = {}
    for planner_name in args.planners:
        print(f"\n=== planner: {planner_name} ===")
        successes, steps_per_ep = [], []
        t0 = time.time()
        for i in range(args.num_episodes):
            np.random.seed(args.seed + i)
            goal_frame = _capture_goal_image(env, args.max_oracle_steps)

            np.random.seed(args.seed + i)
            env.reset()
            z_goal = _encode_frame(encoder, goal_frame)

            if planner_name == "backward":
                fn = lambda z_now, z_goal=z_goal: plan_subgoals(subgoal, z_now, z_goal, args.num_subgoals)
            elif planner_name == "forward":
                fn = lambda z_now, z_goal=z_goal: plan_subgoals_forward(subgoal, z_now, z_goal, args.num_subgoals)
            elif planner_name == "lerp":
                fn = lambda z_now, z_goal=z_goal: plan_subgoals_lerp(z_now, z_goal, args.num_subgoals)
            else:  # none
                fn = lambda z_now, z_goal=z_goal: plan_no_subgoals(z_now, z_goal)

            success, steps = _run_planner(
                env, encoder, inverse, fn,
                max_steps=args.max_steps, epsilon=args.epsilon,
            )
            successes.append(success)
            steps_per_ep.append(steps)

        dt = time.time() - t0
        rate = float(np.mean(successes))
        mean_steps = float(np.mean(steps_per_ep))
        print(f"  success: {sum(successes)}/{args.num_episodes} = {rate:.1%}")
        print(f"  mean steps: {mean_steps:.1f}")
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
