"""DDPM noise scheduler for diffusion models."""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class DDPMScheduler:
    """Denoising Diffusion Probabilistic Model scheduler.

    Implements both linear and cosine beta schedules.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
    ):
        """
        Args:
            num_timesteps: Number of diffusion steps
            beta_schedule: "linear" or "cosine"
            beta_start: Starting beta for linear schedule
            beta_end: Ending beta for linear schedule
            clip_sample: Whether to clip samples to [-1, 1]
            prediction_type: "epsilon" (predict noise) or "v" (predict velocity)
        """
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type

        # Compute betas
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Store as buffers
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]), self.alphas_cumprod[:-1]
        ])

        # Pre-compute values for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Pre-compute values for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine beta schedule from Nichol & Dhariwal."""
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def to(self, device: torch.device) -> "DDPMScheduler":
        """Move scheduler tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to x_0 according to q(x_t | x_0).

        Args:
            x_0: Clean samples (B, ...)
            noise: Random noise (B, ...)
            timesteps: Timestep indices (B,)

        Returns:
            x_t: Noisy samples (B, ...)
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Expand for broadcasting
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        x_t: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Single denoising step.

        Args:
            model_output: Model prediction (noise or velocity)
            timestep: Current timestep
            x_t: Current noisy sample
            generator: Random number generator

        Returns:
            x_{t-1}: Less noisy sample
        """
        t = timestep

        # Get predicted x_0
        if self.prediction_type == "epsilon":
            # model_output is noise prediction
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
            pred_x_0 = (x_t - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t
        elif self.prediction_type == "v":
            # model_output is velocity prediction
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
            pred_x_0 = sqrt_alpha_t * x_t - sqrt_one_minus_alpha_t * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Clip prediction
        if self.clip_sample:
            pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)

        # Compute posterior mean
        posterior_mean = (
            self.posterior_mean_coef1[t] * pred_x_0 +
            self.posterior_mean_coef2[t] * x_t
        )

        # Add noise for t > 0
        if t > 0:
            if generator is not None:
                noise = torch.randn(
                    x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator
                )
            else:
                noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t]
            x_prev = posterior_mean + torch.sqrt(variance) * noise
        else:
            x_prev = posterior_mean

        return x_prev

    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample from the diffusion model.

        Args:
            model: Denoising model that takes (x_t, t) and returns noise prediction
            shape: Shape of samples to generate
            device: Device to sample on
            num_inference_steps: Number of steps for sampling (None = all steps)
            generator: Random number generator

        Returns:
            x_0: Generated samples
        """
        # Start from pure noise
        x_t = torch.randn(shape, device=device, generator=generator)

        # Determine timesteps
        if num_inference_steps is None or num_inference_steps >= self.num_timesteps:
            timesteps = list(range(self.num_timesteps - 1, -1, -1))
        else:
            # DDIM-like spacing
            step_ratio = self.num_timesteps // num_inference_steps
            timesteps = list(range(self.num_timesteps - 1, -1, -step_ratio))

        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            model_output = model(x_t, t_batch)
            x_t = self.step(model_output, t, x_t, generator=generator)

        return x_t

    def get_velocity(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity target v = sqrt(alpha) * noise - sqrt(1-alpha) * x_0.

        Args:
            x_0: Clean samples
            noise: Random noise
            timesteps: Timestep indices

        Returns:
            v: Velocity target
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * noise - sqrt_one_minus_alpha * x_0
