import numpy as np
import torch

from bsp_surgical.models.subgoal import SubgoalGenerator
from bsp_surgical.inference.planner import (
    plan_subgoals,
    plan_subgoals_lerp,
    plan_no_subgoals,
    execute_plan,
)


def test_plan_subgoals_returns_ordered_waypoints_ending_at_goal():
    torch.manual_seed(0)
    mlp = SubgoalGenerator(latent_dim=16).eval()
    z_now = torch.zeros(1, 16)
    z_goal = torch.ones(1, 16)

    waypoints = plan_subgoals(mlp, z_now, z_goal, num_subgoals=2)

    assert len(waypoints) == 3
    assert waypoints[-1] is z_goal  # the final waypoint is the goal itself
    for w in waypoints:
        assert w.shape == (1, 16)


def test_plan_subgoals_order_is_near_to_far():
    """Waypoints go z_now -> sg_2 (nearest) -> sg_1 -> z_goal."""
    torch.manual_seed(0)
    mlp = SubgoalGenerator(latent_dim=16).eval()
    z_now = torch.zeros(1, 16)
    z_goal = torch.ones(1, 16)

    waypoints = plan_subgoals(mlp, z_now, z_goal, num_subgoals=2)
    sg2, sg1, goal = waypoints

    # sg_1 = h(z_now, z_goal); sg_2 = h(z_now, sg_1)
    assert torch.allclose(sg1, mlp(z_now, z_goal))
    assert torch.allclose(sg2, mlp(z_now, sg1))


def test_lerp_baseline_produces_evenly_spaced_waypoints():
    z_now = torch.zeros(1, 4)
    z_goal = torch.ones(1, 4) * 3.0

    waypoints = plan_subgoals_lerp(z_now, z_goal, num_subgoals=2)

    assert len(waypoints) == 3
    # 1/3, 2/3, 1 of the way from z_now to z_goal (near-to-far order preserved)
    assert torch.allclose(waypoints[0], torch.ones(1, 4) * 1.0)
    assert torch.allclose(waypoints[1], torch.ones(1, 4) * 2.0)
    assert torch.allclose(waypoints[2], z_goal)


def test_no_subgoals_returns_only_the_goal():
    z_now = torch.zeros(1, 4)
    z_goal = torch.ones(1, 4)

    waypoints = plan_no_subgoals(z_now, z_goal)

    assert waypoints == [z_goal]


class FakePlanEnv:
    """Env-like: step() returns a rendered frame that linearly interpolates
    toward a target colour. execute_plan should keep stepping until the
    latent is near each waypoint."""

    def __init__(self):
        self.steps_taken = 0

    def render(self, mode="rgb_array"):
        v = min(255, self.steps_taken * 25)
        return np.full((32, 32, 3), v, dtype=np.uint8)

    def step(self, action):
        self.steps_taken += 1
        return None, 0.0, False, {}


def test_execute_plan_steps_until_each_waypoint_reached_then_moves_on():
    """With a trivial VAE (identity-ish in a small dim) and inverse
    model that always returns 1, execute_plan should just tick through
    max_steps_per_waypoint for each waypoint."""
    env = FakePlanEnv()

    class IdentityEncoder(torch.nn.Module):
        def encode(self, x):
            # map flat mean pixel to a 1-d latent
            mu = x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
            return mu, torch.zeros_like(mu)

    class ConstantInverse(torch.nn.Module):
        def forward(self, z, z_target):
            return torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0]])

    encoder = IdentityEncoder()
    inverse = ConstantInverse()

    # three waypoints; env can't reach them (always steps forward a fixed amount)
    waypoints = [torch.tensor([[100.0]]), torch.tensor([[200.0]]), torch.tensor([[300.0]])]
    n_actions = execute_plan(
        env, encoder, inverse, waypoints,
        max_steps_per_waypoint=3, epsilon=0.01,
    )

    assert n_actions == 9  # 3 waypoints × 3 steps each
    assert env.steps_taken == 9
