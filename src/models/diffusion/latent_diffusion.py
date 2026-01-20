"""Latent diffusion model for molecule generation.

Supports two latent modes:
- "nodes_only": Node-wise latents from RAE encoder adapter (B, N, d_latent)
- "nodes_and_edges": Combined node+edge latents from MAE encoder (B, N+E, d_latent)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from src.models.diffusion.noise_scheduler import DDPMScheduler
from src.models.diffusion.dit_block import DiTBlock, TimestepEmbedding


class LatentDiffusionModel(nn.Module):
    """Latent diffusion model using DiT architecture.

    Supports two latent modes:
    - "nodes_only": Operates on node-wise latent representations (B, N, d_latent)
    - "nodes_and_edges": Operates on combined node+edge sequence (B, N+E, d_latent)
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
        latent_mode: str = "nodes_only",
    ):
        """Initialize LatentDiffusionModel.

        Args:
            d_latent: Dimension of latent space
            d_model: Hidden dimension of diffusion transformer
            nhead: Number of attention heads
            num_layers: Number of DiT blocks
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            max_atoms: Maximum number of atoms (N)
            num_timesteps: Number of diffusion timesteps
            beta_schedule: Noise schedule type ("linear" or "cosine")
            latent_mode: "nodes_only" or "nodes_and_edges"
        """
        super().__init__()

        if latent_mode not in ("nodes_only", "nodes_and_edges"):
            raise ValueError(f"latent_mode must be 'nodes_only' or 'nodes_and_edges', got {latent_mode}")

        self.d_latent = d_latent
        self.d_model = d_model
        self.max_atoms = max_atoms
        self.num_timesteps = num_timesteps
        self.latent_mode = latent_mode

        # Compute sequence length based on mode
        if latent_mode == "nodes_only":
            self.num_edges = 0
            self.seq_len = max_atoms
        else:  # nodes_and_edges
            self.num_edges = max_atoms * (max_atoms - 1) // 2
            self.seq_len = max_atoms + self.num_edges

        # Input projection
        self.input_proj = nn.Linear(d_latent, d_model)

        # Token type embedding (only for nodes_and_edges mode)
        if latent_mode == "nodes_and_edges":
            self.token_type_embedding = nn.Embedding(2, d_model)
        else:
            self.token_type_embedding = None

        # Positional embedding
        self.pos_embedding = nn.Embedding(self.seq_len, d_model)

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
            z_noisy: Noisy latents (B, seq_len, d_latent)
            timesteps: Timestep indices (B,)

        Returns:
            Noise prediction (B, seq_len, d_latent)
        """
        B, seq_len, _ = z_noisy.shape
        device = z_noisy.device

        # Input projection
        x = self.input_proj(z_noisy)  # (B, seq_len, d_model)

        # Add token type embeddings (only for nodes_and_edges mode)
        if self.token_type_embedding is not None:
            N = self.max_atoms
            E = self.num_edges
            token_types = torch.cat([
                torch.zeros(N, device=device, dtype=torch.long),
                torch.ones(E, device=device, dtype=torch.long),
            ])  # (N+E,)
            x = x + self.token_type_embedding(token_types)

        # Add positional embeddings
        positions = torch.arange(self.seq_len, device=device)
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
            z_0: Clean latents (B, seq_len, d_latent)

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
            z_0: Sampled latents (B, seq_len, d_latent)
        """
        self.eval()

        shape = (num_samples, self.seq_len, self.d_latent)

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
        z_mean: Optional[torch.Tensor] = None,
        z_std: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample molecules end-to-end.

        Args:
            num_samples: Number of molecules to generate
            device: Device to sample on
            num_inference_steps: Number of diffusion steps
            temperature: Decoding temperature
            z_mean: Optional latent mean for denormalization (1, 1, d_latent)
            z_std: Optional latent std for denormalization (1, 1, d_latent)

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

        # Denormalize if normalization params provided
        if z_mean is not None and z_std is not None:
            z = z * z_std.to(device) + z_mean.to(device)

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


class LatentDiffusionWithMAE(nn.Module):
    """Combined model for end-to-end sampling with MAE.

    Uses frozen MAE encoder/decoder + diffusion model operating on
    combined node+edge latent sequences.
    """

    def __init__(
        self,
        mae: nn.Module,
        diffusion: LatentDiffusionModel,
    ):
        """Initialize combined model.

        Args:
            mae: Trained MoleculeMAE model (frozen)
            diffusion: LatentDiffusionModel with latent_mode='nodes_and_edges'
        """
        super().__init__()
        self.mae = mae
        self.diffusion = diffusion

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        num_inference_steps: Optional[int] = None,
        temperature: float = 1.0,
        z_mean: Optional[torch.Tensor] = None,
        z_std: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample molecules end-to-end.

        Args:
            num_samples: Number of molecules to generate
            device: Device to sample on
            num_inference_steps: Number of diffusion steps
            temperature: Decoding temperature
            z_mean: Optional latent mean for denormalization (1, 1, d_latent)
            z_std: Optional latent std for denormalization (1, 1, d_latent)

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

        # Denormalize if normalization params provided
        if z_mean is not None and z_std is not None:
            z = z * z_std.to(device) + z_mean.to(device)

        # Decode latents to molecules using MAE decoder
        node_types, adj_matrix = self.decode_latents(z, temperature=temperature)

        return node_types, adj_matrix

    def decode_latents(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode diffusion latents to molecules using MAE decoder.

        Uses mask=False so decoder uses actual latent content (not mask tokens).
        Position embeddings are still added by the decoder.

        Args:
            z: Latent representations from diffusion (B, N+E, d_latent)
            temperature: Sampling temperature

        Returns:
            node_types: (B, N) atom type indices
            adj_matrix: (B, N, N) bond types
        """
        # Import here to avoid circular imports
        from src.models.mae.masking import reconstruct_adj_from_edges

        B = z.shape[0]
        N = self.mae.max_atoms
        E = self.mae.num_edges
        device = z.device

        # No masking = decoder uses actual latent content
        node_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        edge_mask = torch.zeros(B, E, dtype=torch.bool, device=device)

        # Decode using MAE decoder
        node_logits, edge_logits = self.mae.decoder(z, node_mask, edge_mask)

        # Apply temperature
        if temperature != 1.0:
            node_logits = node_logits / temperature
            edge_logits = edge_logits / temperature

        # Sample or argmax
        if temperature > 0:
            node_probs = F.softmax(node_logits, dim=-1)
            edge_probs = F.softmax(edge_logits, dim=-1)

            node_types = torch.multinomial(
                node_probs.view(-1, self.mae.num_atom_types), 1
            ).view(B, N)
            edge_types = torch.multinomial(
                edge_probs.view(-1, self.mae.num_bond_types), 1
            ).view(B, E)
        else:
            node_types = node_logits.argmax(dim=-1)
            edge_types = edge_logits.argmax(dim=-1)

        # Reconstruct adjacency matrix from edge predictions
        adj_matrix = reconstruct_adj_from_edges(edge_types.float(), N).long()

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
            all_latents: (N_samples, seq_len, d_model) where seq_len = N + E
        """
        self.mae.eval()
        all_latents = []

        with torch.no_grad():
            for batch in dataloader:
                node_features = batch['node_features'].to(device)
                adj_matrix = batch['adj_matrix'].to(device)

                # Encode using MAE encoder (without masking)
                z = self.mae.encode(node_features, adj_matrix)
                all_latents.append(z.cpu())

        return torch.cat(all_latents, dim=0)
