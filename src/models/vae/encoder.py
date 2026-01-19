"""Transformer encoder with edge bias for molecular VAE."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EdgeBiasAttention(nn.Module):
    """Multi-head attention with edge bias from adjacency matrix."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        num_bond_types: int = 5,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Edge bias: bond type -> attention bias per head
        self.edge_bias = nn.Embedding(num_bond_types, nhead)
        nn.init.zeros_(self.edge_bias.weight)  # Start with no bias

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (B, N, d_model)
            adj_matrix: Bond types (B, N, N) with values 0-4

        Returns:
            Updated node features (B, N, d_model)
        """
        B, N, _ = x.shape

        # Compute Q, K, V
        Q = self.q_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        # (B, nhead, N, head_dim)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (B, nhead, N, N)

        # Add edge bias from adjacency matrix
        edge_bias = self.edge_bias(adj_matrix)  # (B, N, N, nhead)
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # (B, nhead, N, N)
        attn_scores = attn_scores + edge_bias

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, nhead, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)

        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with edge bias attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_bond_types: int = 5,
    ):
        super().__init__()

        self.self_attn = EdgeBiasAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            num_bond_types=num_bond_types,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.dropout(self.self_attn(self.norm1(x), adj_matrix))

        # Feedforward with residual (pre-norm style)
        h = self.norm2(x)
        h = self.linear2(self.dropout(F.gelu(self.linear1(h))))
        x = x + self.dropout(h)

        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder for molecular VAE.

    Encodes molecules into node-wise latent representations.
    Uses edge bias attention to incorporate bond information.
    """

    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        d_model: int,
        d_latent: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_atoms: int = 9,
    ):
        super().__init__()

        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.d_model = d_model
        self.d_latent = d_latent
        self.max_atoms = max_atoms

        # Atom type embedding (from one-hot to d_model)
        self.atom_embedding = nn.Linear(num_atom_types, d_model)

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_atoms, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_bond_types=num_bond_types,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output projection to latent space (mu and log_var)
        self.mu_proj = nn.Linear(d_model, d_latent)
        self.logvar_proj = nn.Linear(d_model, d_latent)

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N) with values 0-4

        Returns:
            mu: Mean of latent distribution (B, N, d_latent)
            log_var: Log variance of latent distribution (B, N, d_latent)
        """
        B, N, _ = node_features.shape
        device = node_features.device

        # Embed atoms
        x = self.atom_embedding(node_features)  # (B, N, d_model)

        # Add positional embeddings
        positions = torch.arange(N, device=device)
        x = x + self.pos_embedding(positions)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, adj_matrix)

        x = self.norm(x)

        # Project to latent space
        mu = self.mu_proj(x)
        log_var = self.logvar_proj(x)

        return mu, log_var

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick for VAE training.

        Args:
            mu: Mean (B, N, d_latent)
            log_var: Log variance (B, N, d_latent)

        Returns:
            z: Sampled latent (B, N, d_latent)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """Encode molecules to latent space.

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)
            sample: If True, sample from distribution; else return mean

        Returns:
            z: Latent representation (B, N, d_latent)
        """
        mu, log_var = self.forward(node_features, adj_matrix)
        if sample:
            return self.reparameterize(mu, log_var)
        return mu
