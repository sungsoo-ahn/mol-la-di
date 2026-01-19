"""MAE Encoder - Transformer operating on full node+edge sequence."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.models.mae.masking import extract_upper_triangular


class MAEEncoder(nn.Module):
    """Masked Autoencoder Encoder.

    Embeds nodes and edges into a single sequence, replaces masked positions
    with learnable [MASK] tokens, and processes through a Transformer.

    Sequence structure: [NODE_0, ..., NODE_{N-1}, EDGE_{0,1}, EDGE_{0,2}, ..., EDGE_{N-2,N-1}]
    Total sequence length: N + N*(N-1)/2
    """

    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_atoms: int = 9,
    ):
        """Initialize MAE Encoder.

        Args:
            num_atom_types: Number of atom type classes (including empty)
            num_bond_types: Number of bond type classes (including no-bond)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            max_atoms: Maximum number of atoms (N)
        """
        super().__init__()

        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.d_model = d_model
        self.max_atoms = max_atoms
        self.num_edges = max_atoms * (max_atoms - 1) // 2
        self.seq_len = max_atoms + self.num_edges

        # Token embeddings
        self.node_embedding = nn.Embedding(num_atom_types, d_model)
        self.edge_embedding = nn.Embedding(num_bond_types, d_model)

        # Learnable [MASK] tokens (separate for nodes and edges)
        self.node_mask_token = nn.Parameter(torch.randn(d_model))
        self.edge_mask_token = nn.Parameter(torch.randn(d_model))

        # Token type embeddings (0 = node, 1 = edge)
        self.token_type_embedding = nn.Embedding(2, d_model)

        # Position embeddings for full sequence
        self.position_embedding = nn.Embedding(self.seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.node_mask_token, std=0.02)
        nn.init.normal_(self.edge_mask_token, std=0.02)

    def forward(
        self,
        node_types: torch.Tensor,
        adj_matrix: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            node_types: Atom type indices (B, N)
            adj_matrix: Bond type adjacency matrix (B, N, N)
            node_mask: Boolean mask for nodes (B, N), True = masked
            edge_mask: Boolean mask for edges (B, E), True = masked

        Returns:
            encoded: Encoded sequence (B, N+E, d_model)
        """
        B, N = node_types.shape
        device = node_types.device
        E = self.num_edges

        # Extract edge types from upper triangular of adj_matrix
        edge_types = extract_upper_triangular(adj_matrix)  # (B, E)

        # Embed nodes and edges
        node_embeds = self.node_embedding(node_types)  # (B, N, d_model)
        edge_embeds = self.edge_embedding(edge_types)  # (B, E, d_model)

        # Replace masked positions with [MASK] tokens
        node_mask_expanded = node_mask.unsqueeze(-1).expand_as(node_embeds)
        node_embeds = torch.where(
            node_mask_expanded,
            self.node_mask_token.expand(B, N, -1),
            node_embeds,
        )

        edge_mask_expanded = edge_mask.unsqueeze(-1).expand_as(edge_embeds)
        edge_embeds = torch.where(
            edge_mask_expanded,
            self.edge_mask_token.expand(B, E, -1),
            edge_embeds,
        )

        # Concatenate to single sequence: [nodes, edges]
        sequence = torch.cat([node_embeds, edge_embeds], dim=1)  # (B, N+E, d_model)

        # Add token type embeddings
        token_types = torch.cat([
            torch.zeros(N, device=device, dtype=torch.long),
            torch.ones(E, device=device, dtype=torch.long),
        ])  # (N+E,)
        sequence = sequence + self.token_type_embedding(token_types)

        # Add position embeddings
        positions = torch.arange(self.seq_len, device=device)
        sequence = sequence + self.position_embedding(positions)

        # Apply transformer
        encoded = self.transformer(sequence)
        encoded = self.norm(encoded)

        return encoded

    def get_node_edge_split(self, encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split encoded sequence back into node and edge representations.

        Args:
            encoded: Encoded sequence (B, N+E, d_model)

        Returns:
            node_encoded: (B, N, d_model)
            edge_encoded: (B, E, d_model)
        """
        node_encoded = encoded[:, :self.max_atoms, :]
        edge_encoded = encoded[:, self.max_atoms:, :]
        return node_encoded, edge_encoded
