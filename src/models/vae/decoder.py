"""One-shot decoder for molecular VAE."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class OneShotDecoder(nn.Module):
    """One-shot decoder for molecular VAE.

    Decodes latent representations to molecules in parallel.
    - Node head: predicts atom types for each position
    - Edge head: predicts bond types for each pair using pairwise MLP
    """

    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        d_latent: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_atoms: int = 9,
    ):
        super().__init__()

        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.d_latent = d_latent
        self.d_model = d_model
        self.max_atoms = max_atoms

        # Input projection from latent to model dimension
        self.input_proj = nn.Linear(d_latent, d_model)

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_atoms, d_model)

        # Transformer decoder layers (self-attention only, no cross-attention)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

        # Node head: predicts atom types
        self.node_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_atom_types),
        )

        # Edge head: pairwise MLP for bond prediction
        # Takes concatenated pair representations
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_bond_types),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Latent representation (B, N, d_latent)

        Returns:
            node_logits: (B, N, num_atom_types)
            edge_logits: (B, N, N, num_bond_types)
        """
        B, N, _ = z.shape
        device = z.device

        # Project to model dimension
        x = self.input_proj(z)  # (B, N, d_model)

        # Add positional embeddings
        positions = torch.arange(N, device=device)
        x = x + self.pos_embedding(positions)

        # Apply transformer layers
        x = self.transformer(x)
        x = self.norm(x)

        # Node predictions
        node_logits = self.node_head(x)  # (B, N, num_atom_types)

        # Edge predictions via pairwise MLP
        # Create all pairs of node representations
        x_i = x.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, d_model)
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, d_model)
        pair_features = torch.cat([x_i, x_j], dim=-1)  # (B, N, N, 2*d_model)

        edge_logits = self.edge_mlp(pair_features)  # (B, N, N, num_bond_types)

        return node_logits, edge_logits

    def decode(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latents to molecules with sampling.

        Args:
            z: Latent representation (B, N, d_latent)
            temperature: Sampling temperature
            hard: If True, return discrete values; else return probabilities

        Returns:
            node_types: (B, N) atom type indices or (B, N, num_atom_types) probs
            adj_matrix: (B, N, N) bond types or (B, N, N, num_bond_types) probs
        """
        node_logits, edge_logits = self.forward(z)

        if temperature != 1.0:
            node_logits = node_logits / temperature
            edge_logits = edge_logits / temperature

        if hard:
            # Sample discrete values
            node_types = torch.argmax(node_logits, dim=-1)  # (B, N)
            edge_types = torch.argmax(edge_logits, dim=-1)  # (B, N, N)

            # Make adjacency symmetric
            edge_types = (edge_types + edge_types.transpose(1, 2)) // 2
            # Set diagonal to 0 (no self-loops)
            edge_types = edge_types * (1 - torch.eye(z.size(1), device=z.device).long())

            return node_types, edge_types
        else:
            node_probs = F.softmax(node_logits, dim=-1)
            edge_probs = F.softmax(edge_logits, dim=-1)
            return node_probs, edge_probs

    def sample(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample molecules from latents using multinomial sampling.

        Args:
            z: Latent representation (B, N, d_latent)
            temperature: Sampling temperature

        Returns:
            node_types: (B, N) atom type indices
            adj_matrix: (B, N, N) bond types
        """
        B, N, _ = z.shape
        device = z.device

        node_logits, edge_logits = self.forward(z)

        # Apply temperature
        node_logits = node_logits / temperature
        edge_logits = edge_logits / temperature

        # Sample nodes
        node_probs = F.softmax(node_logits, dim=-1)  # (B, N, num_atom_types)
        node_types = torch.multinomial(
            node_probs.view(B * N, -1), num_samples=1
        ).view(B, N)

        # Sample edges
        edge_probs = F.softmax(edge_logits, dim=-1)  # (B, N, N, num_bond_types)
        edge_types = torch.multinomial(
            edge_probs.view(B * N * N, -1), num_samples=1
        ).view(B, N, N)

        # Make adjacency symmetric (take upper triangle)
        mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
        upper = edge_types * mask.long()
        edge_types = upper + upper.transpose(1, 2)

        return node_types, edge_types
