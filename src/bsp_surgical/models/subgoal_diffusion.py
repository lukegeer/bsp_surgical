"""Latent subgoal diffusion model (LBP-style).

Given (z_now, z_target), samples a midpoint z_mid from a learned
conditional distribution — not a deterministic MLP.

Training: given (z_start, z_mid_true, z_end) triples from demo
trajectories, add noise to z_mid_true at a random timestep t, train
denoiser to predict the noise conditioned on (z_start, z_end, t).

Inference: start from pure Gaussian noise, denoise over T steps
conditioned on (z_start, z_end), return sampled z_mid. Used
recursively for backward bisection: sg_1 = diff(z_now, z_goal),
sg_2 = diff(z_now, sg_1), etc.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) int
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SubgoalDiffusionMLP(nn.Module):
    """ε-predicting MLP denoiser. Input: (z_noised, z_start, z_end, t_emb)."""

    def __init__(
        self,
        latent_dim: int,
        hidden: int = 1024,
        time_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = nn.Sequential(
            _SinusoidalTimeEmbed(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )
        in_dim = 3 * latent_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(
        self,
        z_noised: torch.Tensor,
        z_start: torch.Tensor,
        z_end: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)
        x = torch.cat([z_noised, z_start, z_end, t_emb], dim=-1)
        return self.net(x)


class SubgoalDiffusion(nn.Module):
    """Wraps the denoiser with a linear beta schedule and DDIM sampler.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent subgoal.
    num_timesteps : int
        DDPM timesteps for training (typical 100-1000).
    beta_start, beta_end : float
        Linear beta schedule range.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden: int = 1024,
        num_timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        self.denoiser = SubgoalDiffusionMLP(latent_dim, hidden=hidden)

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def add_noise(
        self, z_clean: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(z_clean)
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        z_noisy = sqrt_ab * z_clean + sqrt_omb * noise
        return z_noisy, noise

    def training_loss(
        self,
        z_mid_true: torch.Tensor,
        z_start: torch.Tensor,
        z_end: torch.Tensor,
    ) -> torch.Tensor:
        B = z_mid_true.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=z_mid_true.device)
        z_noisy, noise = self.add_noise(z_mid_true, t)
        eps_pred = self.denoiser(z_noisy, z_start, z_end, t)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(
        self,
        z_start: torch.Tensor,
        z_end: torch.Tensor,
        num_inference_steps: int = 20,
    ) -> torch.Tensor:
        """DDIM sampling — uniform subsample of training timesteps."""
        B = z_start.shape[0]
        z = torch.randn(B, self.latent_dim, device=z_start.device)
        step_ids = torch.linspace(
            self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long,
            device=z_start.device,
        )
        for i, t in enumerate(step_ids):
            t_batch = t.expand(B)
            eps_pred = self.denoiser(z, z_start, z_end, t_batch)

            ab_t = self.alpha_bar[t]
            if i + 1 < len(step_ids):
                ab_prev = self.alpha_bar[step_ids[i + 1]]
            else:
                ab_prev = torch.tensor(1.0, device=z.device)

            # predicted clean from noise prediction
            z0_pred = (z - torch.sqrt(1 - ab_t) * eps_pred) / torch.sqrt(ab_t)
            # DDIM deterministic update
            z = torch.sqrt(ab_prev) * z0_pred + torch.sqrt(1 - ab_prev) * eps_pred
        return z

    @torch.no_grad()
    def backward_bisect(
        self,
        z_now: torch.Tensor,
        z_goal: torch.Tensor,
        num_subgoals: int = 2,
        num_inference_steps: int = 20,
    ) -> list[torch.Tensor]:
        """Recursive backward bisection via diffusion sampling.

        Returns waypoints in near-to-far order: [sg_K, ..., sg_1, z_goal].
        """
        subgoals = []
        target = z_goal
        for _ in range(num_subgoals):
            sg = self.sample(z_now, target, num_inference_steps=num_inference_steps)
            subgoals.append(sg)
            target = sg
        return list(reversed(subgoals)) + [z_goal]

    @torch.no_grad()
    def backward_bisect_with_rerank(
        self,
        z_now: torch.Tensor,
        z_goal: torch.Tensor,
        anchors: torch.Tensor,
        num_subgoals: int = 2,
        num_candidates: int = 8,
        num_inference_steps: int = 20,
    ) -> list[torch.Tensor]:
        """Compositional rerank: at each bisection level, sample
        num_candidates subgoals from the diffusion model and pick the
        one with highest cosine similarity to any anchor.

        `anchors`: (N, latent_dim) tensor of in-distribution training
        states. Reranking forces subgoals to look like states the
        inverse model was actually trained to reach — crucial for
        transferring to unseen tasks via composition.
        """
        import torch.nn.functional as F

        anchors_norm = F.normalize(anchors, dim=-1)
        subgoals = []
        target = z_goal
        for _ in range(num_subgoals):
            # Sample num_candidates subgoals in one batched forward
            z_now_b = z_now.expand(num_candidates, -1)
            target_b = target.expand(num_candidates, -1)
            cands = self.sample(z_now_b, target_b, num_inference_steps=num_inference_steps)
            # Cosine similarity each candidate vs every anchor → max over anchors
            cands_norm = F.normalize(cands, dim=-1)
            sim = cands_norm @ anchors_norm.T              # (num_candidates, N)
            best_scores, _ = sim.max(dim=1)                 # (num_candidates,)
            best = cands[best_scores.argmax()].unsqueeze(0)
            subgoals.append(best)
            target = best
        return list(reversed(subgoals)) + [z_goal]
