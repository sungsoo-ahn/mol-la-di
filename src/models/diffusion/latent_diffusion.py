"""Latent diffusion model for molecule generation.

Uses RAE (Representation Autoencoder) architecture:
- Frozen MAE encoder + trainable RAE decoder with noise augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from src.models.diffusion.noise_scheduler import DDPMScheduler
from src.models.diffusion.dit_block import DiTBlock, TimestepEmbedding


class LatentDiffusionModel(nn.Module):
    """Latent diffusion model using DiT architecture.

    Operates on node-wise latent representations from the encoder.
    """

    def __init__(
        self,
        d_latent: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        max_atoms: int = 9,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
    ):
        super().__init__()

        self.d_latent = d_latent
        self.d_model = d_model
        self.max_atoms = max_atoms
        self.num_timesteps = num_timesteps

        # Input projection
        self.input_proj = nn.Linear(d_latent, d_model)

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_atoms, d_model)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(d_model)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm and projection
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_linear = nn.Linear(d_model, d_latent)

        # Initialize final layer to zero (AdaLN-Zero style)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

        # Final AdaLN modulation
        self.final_adaLN = nn.Linear(d_model, d_model * 2)
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)

        # Noise scheduler
        self.scheduler = DDPMScheduler(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
        )

    def forward(
        self,
        z_noisy: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from noisy latents.

        Args:
            z_noisy: Noisy latents (B, N, d_latent)
            timesteps: Timestep indices (B,)

        Returns:
            Noise prediction (B, N, d_latent)
        """
        B, N, _ = z_noisy.shape
        device = z_noisy.device

        # Input projection
        x = self.input_proj(z_noisy)  # (B, N, d_model)

        # Add positional embeddings
        positions = torch.arange(N, device=device)
        x = x + self.pos_embedding(positions)

        # Timestep conditioning
        c = self.time_embed(timesteps)  # (B, d_model)

        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, c)

        # Final projection with AdaLN
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)

        x = self.final_norm(x)
        x = x * (1 + scale) + shift
        noise_pred = self.final_linear(x)

        return noise_pred

    def compute_loss(
        self,
        z_0: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute diffusion loss.

        Args:
            z_0: Clean latents (B, N, d_latent)

        Returns:
            Dictionary with loss and metrics
        """
        B = z_0.size(0)
        device = z_0.device

        # Ensure scheduler is on correct device
        self.scheduler.to(device)

        # Sample random timesteps
        timesteps = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(z_0)

        # Add noise to latents
        z_noisy = self.scheduler.add_noise(z_0, noise, timesteps)

        # Predict noise
        noise_pred = self.forward(z_noisy, timesteps)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return {
            "loss": loss,
            "mse": loss,
        }

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample latents from the diffusion model.

        Args:
            num_samples: Number of samples to generate
            device: Device to sample on
            num_inference_steps: Number of denoising steps

        Returns:
            z_0: Sampled latents (B, N, d_latent)
        """
        self.eval()

        shape = (num_samples, self.max_atoms, self.d_latent)

        # Move scheduler to device
        self.scheduler.to(device)

        # Start from pure noise
        z_t = torch.randn(shape, device=device)

        # Determine timesteps
        if num_inference_steps is None or num_inference_steps >= self.num_timesteps:
            timesteps = list(range(self.num_timesteps - 1, -1, -1))
        else:
            step_ratio = self.num_timesteps // num_inference_steps
            timesteps = list(range(self.num_timesteps - 1, -1, -step_ratio))

        # Denoising loop
        for t in timesteps:
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = self.forward(z_t, t_batch)
            z_t = self.scheduler.step(noise_pred, t, z_t)

        return z_t


class LatentDiffusionWithRAE(nn.Module):
    """Combined model for end-to-end sampling with RAE decoder.

    Uses frozen MAE encoder + RAE decoder + diffusion model.
    """

    def __init__(
        self,
        encoder_adapter: nn.Module,
        rae_decoder: nn.Module,
        diffusion: LatentDiffusionModel,
    ):
        """Initialize combined model.

        Args:
            encoder_adapter: MAE encoder adapter (for encoding training data)
            rae_decoder: RAE decoder (for decoding sampled latents)
            diffusion: Latent diffusion model
        """
        super().__init__()
        self.encoder_adapter = encoder_adapter
        self.rae_decoder = rae_decoder
        self.diffusion = diffusion

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        num_inference_steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample molecules end-to-end.

        Args:
            num_samples: Number of molecules to generate
            device: Device to sample on
            num_inference_steps: Number of diffusion steps
            temperature: Decoding temperature

        Returns:
            node_types: (B, N) atom type indices
            adj_matrix: (B, N, N) bond types
        """
        self.eval()

        # Sample latents from diffusion model
        z = self.diffusion.sample(
            num_samples=num_samples,
            device=device,
            num_inference_steps=num_inference_steps,
        )

        # Decode to molecules using RAE decoder
        node_types, adj_matrix = self.rae_decoder.decode(z, temperature=temperature, hard=True)

        return node_types, adj_matrix

    def encode_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode entire dataset to latents for diffusion training.

        Args:
            dataloader: DataLoader with molecules
            device: Device

        Returns:
            all_latents: (N, max_atoms, d_latent)
        """
        self.encoder_adapter.eval()
        all_latents = []

        with torch.no_grad():
            for batch in dataloader:
                node_features = batch['node_features'].to(device)
                adj_matrix = batch['adj_matrix'].to(device)

                # Encode to latents using MAE encoder adapter
                z = self.encoder_adapter.encode(node_features, adj_matrix)
                all_latents.append(z.cpu())

        return torch.cat(all_latents, dim=0)
