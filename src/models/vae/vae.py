"""Molecular VAE combining encoder and decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from src.models.vae.encoder import TransformerEncoder
from src.models.vae.decoder import OneShotDecoder


class MoleculeVAE(nn.Module):
    """Variational Autoencoder for molecules.

    Encodes molecules into node-wise latent representations and
    reconstructs both atom types and bond types.
    """

    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        d_model: int = 256,
        d_latent: int = 64,
        nhead: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_atoms: int = 9,
    ):
        super().__init__()

        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.d_model = d_model
        self.d_latent = d_latent
        self.max_atoms = max_atoms

        self.encoder = TransformerEncoder(
            num_atom_types=num_atom_types,
            num_bond_types=num_bond_types,
            d_model=d_model,
            d_latent=d_latent,
            nhead=nhead,
            num_layers=encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_atoms=max_atoms,
        )

        self.decoder = OneShotDecoder(
            num_atom_types=num_atom_types,
            num_bond_types=num_bond_types,
            d_latent=d_latent,
            d_model=d_model,
            nhead=nhead,
            num_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_atoms=max_atoms,
        )

    def encode(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode molecules to latent space.

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)
            sample: If True, sample from distribution

        Returns:
            z: Latent representation (B, N, d_latent)
            mu: Mean (B, N, d_latent)
            log_var: Log variance (B, N, d_latent)
        """
        mu, log_var = self.encoder(node_features, adj_matrix)
        if sample:
            z = self.encoder.reparameterize(mu, log_var)
        else:
            z = mu
        return z, mu, log_var

    def decode(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latents to molecule logits.

        Args:
            z: Latent representation (B, N, d_latent)

        Returns:
            node_logits: (B, N, num_atom_types)
            edge_logits: (B, N, N, num_bond_types)
        """
        return self.decoder(z)

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)

        Returns:
            Dictionary with node_logits, edge_logits, mu, log_var, z
        """
        z, mu, log_var = self.encode(node_features, adj_matrix, sample=True)
        node_logits, edge_logits = self.decode(z)

        return {
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "mu": mu,
            "log_var": log_var,
            "z": z,
        }

    def compute_loss(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        beta: float = 1.0,
        edge_weight: float = 1.0,
        symmetric_edge_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss.

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)
            beta: KL divergence weight (for beta-VAE)
            edge_weight: Weight for edge reconstruction loss
            symmetric_edge_loss: If True, only compute edge loss on upper triangle

        Returns:
            Dictionary with total_loss, node_loss, edge_loss, kl_loss
        """
        outputs = self.forward(node_features, adj_matrix)

        node_logits = outputs["node_logits"]
        edge_logits = outputs["edge_logits"]
        mu = outputs["mu"]
        log_var = outputs["log_var"]

        # Node reconstruction loss (cross-entropy)
        # node_features is one-hot, get targets as indices
        node_targets = torch.argmax(node_features, dim=-1)  # (B, N)
        node_loss = F.cross_entropy(
            node_logits.view(-1, self.num_atom_types),
            node_targets.view(-1),
        )

        # Edge reconstruction loss (cross-entropy)
        # adj_matrix contains bond types as targets
        if symmetric_edge_loss:
            # Only compute loss on upper triangle (no diagonal, no redundant lower triangle)
            B, N, _ = adj_matrix.shape
            # Create upper triangle mask (exclude diagonal)
            triu_mask = torch.triu(torch.ones(N, N, device=adj_matrix.device), diagonal=1).bool()
            # Apply mask: (B, N, N) -> (B, num_upper_edges)
            edge_logits_masked = edge_logits[:, triu_mask, :]  # (B, N*(N-1)/2, num_bond_types)
            adj_targets_masked = adj_matrix[:, triu_mask]  # (B, N*(N-1)/2)
            edge_loss = F.cross_entropy(
                edge_logits_masked.reshape(-1, self.num_bond_types),
                adj_targets_masked.reshape(-1),
            )
        else:
            edge_loss = F.cross_entropy(
                edge_logits.view(-1, self.num_bond_types),
                adj_matrix.view(-1),
            )

        # KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        total_loss = node_loss + edge_weight * edge_loss + beta * kl_loss

        return {
            "total_loss": total_loss,
            "node_loss": node_loss,
            "edge_loss": edge_loss,
            "kl_loss": kl_loss,
        }

    def sample(
        self,
        num_samples: int,
        device: torch.device,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample molecules from prior.

        Args:
            num_samples: Number of molecules to sample
            device: Device to generate on
            temperature: Sampling temperature

        Returns:
            node_types: (B, N) atom type indices
            adj_matrix: (B, N, N) bond types
        """
        # Sample from prior N(0, 1)
        z = torch.randn(num_samples, self.max_atoms, self.d_latent, device=device)

        # Decode to molecules
        return self.decoder.sample(z, temperature=temperature)

    def reconstruct(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct molecules through VAE.

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)
            temperature: Sampling temperature for decoding

        Returns:
            node_types: (B, N) reconstructed atom type indices
            adj_matrix: (B, N, N) reconstructed bond types
        """
        z, _, _ = self.encode(node_features, adj_matrix, sample=False)
        return self.decoder.decode(z, temperature=temperature, hard=True)

    def get_latent(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        sample: bool = False,
    ) -> torch.Tensor:
        """Get latent representation for diffusion training.

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)
            sample: If True, sample from posterior; else use mean

        Returns:
            z: Latent representation (B, N, d_latent)
        """
        z, _, _ = self.encode(node_features, adj_matrix, sample=sample)
        return z
