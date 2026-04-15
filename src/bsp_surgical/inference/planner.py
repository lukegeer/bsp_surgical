from typing import Sequence

import numpy as np
import torch


def plan_subgoals(
    subgoal_mlp: torch.nn.Module,
    z_now: torch.Tensor,
    z_goal: torch.Tensor,
    num_subgoals: int = 2,
) -> list[torch.Tensor]:
    """Recursive backward bisection in latent space.

    Returns waypoints ordered from *nearest to farthest* from z_now:
        [sg_K, sg_{K-1}, ..., sg_1, z_goal]

    For num_subgoals=2 (the default, per LBP/HVF):
        sg_1 = h(z_now, z_goal)
        sg_2 = h(z_now, sg_1)
        → waypoints = [sg_2, sg_1, z_goal]
    """
    with torch.no_grad():
        subgoals: list[torch.Tensor] = []
        target = z_goal
        for _ in range(num_subgoals):
            sg = subgoal_mlp(z_now, target)
            subgoals.append(sg)
            target = sg
        # subgoals are [sg_1, sg_2, ...]; we want [sg_K, ..., sg_1, z_goal]
        waypoints = list(reversed(subgoals))
        waypoints.append(z_goal)
        return waypoints


def plan_subgoals_lerp(
    z_now: torch.Tensor,
    z_goal: torch.Tensor,
    num_subgoals: int = 2,
) -> list[torch.Tensor]:
    """LERP baseline: evenly-spaced linear interpolation between z_now and
    z_goal in latent space. Returned in the same near-to-far order as
    plan_subgoals."""
    fractions = [(i + 1) / (num_subgoals + 1) for i in range(num_subgoals)]
    subgoals = [z_now + f * (z_goal - z_now) for f in fractions]
    return subgoals + [z_goal]


def plan_no_subgoals(
    z_now: torch.Tensor,
    z_goal: torch.Tensor,
) -> list[torch.Tensor]:
    """Direct baseline: jump straight to the goal, no intermediate waypoints."""
    _ = z_now  # unused; parameter kept for API symmetry
    return [z_goal]


def _encode_frame(encoder, frame: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    mu, _logvar = encoder.encode(t)
    return mu


def execute_plan(
    env,
    encoder,
    inverse,
    waypoints: Sequence[torch.Tensor],
    *,
    max_steps_per_waypoint: int,
    epsilon: float,
) -> int:
    """Step env under a learned inverse dynamics model until each waypoint
    is reached (within `epsilon` L2 in latent space) or the per-waypoint
    step cap is hit. Returns the total number of env steps taken."""
    total_steps = 0
    for w in waypoints:
        for _ in range(max_steps_per_waypoint):
            z = _encode_frame(encoder, env.render("rgb_array"))
            if torch.linalg.norm(z - w, dim=-1).item() <= epsilon:
                break
            action = inverse(z, w)
            env.step(action.squeeze(0).detach().cpu().numpy())
            total_steps += 1
    return total_steps
