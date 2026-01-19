"""MAE Decoder - Lightweight decoder for masked token prediction."""

import torch
import torch.nn as nn
from typing import Tuple


class MAEDecoder(nn.Module):
    """Lightweight MAE Decoder.

    Predicts original tokens at masked positions using the encoder output.
    Uses a smaller dimension than the encoder for efficiency.
    """

    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        d_encoder: int = 256,
        d_decoder: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_atoms: int = 9,
    ):
        """Initialize MAE Decoder.

        Args:
            num_atom_types: Number of atom type classes
            num_bond_types: Number of bond type classes
            d_encoder: Encoder output dimension
            d_decoder: Decoder hidden dimension (smaller than encoder)
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            max_atoms: Maximum number of atoms
        """
        super().__init__()

        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.d_encoder = d_encoder
        self.d_decoder = d_decoder
        self.max_atoms = max_atoms
        self.num_edges = max_atoms * (max_atoms - 1) // 2

        # Project from encoder dimension to decoder dimension
        self.input_proj = nn.Linear(d_encoder, d_decoder)

        # Learnable mask tokens for decoder input
        self.node_mask_token = nn.Parameter(torch.randn(d_decoder))
        self.edge_mask_token = nn.Parameter(torch.randn(d_decoder))

        # Token type embeddings
        self.token_type_embedding = nn.Embedding(2, d_decoder)

        # Position embeddings
        seq_len = max_atoms + self.num_edges
        self.position_embedding = nn.Embedding(seq_len, d_decoder)

        # Transformer decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_decoder,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Final layer norm
        self.norm = nn.LayerNorm(d_decoder)

        # Prediction heads
        self.node_head = nn.Linear(d_decoder, num_atom_types)
        self.edge_head = nn.Linear(d_decoder, num_bond_types)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.node_mask_token, std=0.02)
        nn.init.normal_(self.edge_mask_token, std=0.02)

    def forward(
        self,
        encoded: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through decoder.

        Args:
            encoded: Encoder output (B, N+E, d_encoder)
            node_mask: Boolean mask for nodes (B, N), True = masked
            edge_mask: Boolean mask for edges (B, E), True = masked

        Returns:
            node_logits: Predictions for all node positions (B, N, num_atom_types)
            edge_logits: Predictions for all edge positions (B, E, num_bond_types)
        """
        B = encoded.shape[0]
        N = self.max_atoms
        E = self.num_edges
        device = encoded.device

        # Project encoder output to decoder dimension
        x = self.input_proj(encoded)  # (B, N+E, d_decoder)

        # Split into nodes and edges
        node_encoded = x[:, :N, :]  # (B, N, d_decoder)
        edge_encoded = x[:, N:, :]  # (B, E, d_decoder)

        # Replace masked positions with decoder mask tokens
        # This helps the decoder focus on reconstructing masked positions
        node_mask_expanded = node_mask.unsqueeze(-1).expand_as(node_encoded)
        node_encoded = torch.where(
            node_mask_expanded,
            self.node_mask_token.expand(B, N, -1),
            node_encoded,
        )

        edge_mask_expanded = edge_mask.unsqueeze(-1).expand_as(edge_encoded)
        edge_encoded = torch.where(
            edge_mask_expanded,
            self.edge_mask_token.expand(B, E, -1),
            edge_encoded,
        )

        # Reassemble sequence
        sequence = torch.cat([node_encoded, edge_encoded], dim=1)  # (B, N+E, d_decoder)

        # Add token type embeddings
        token_types = torch.cat([
            torch.zeros(N, device=device, dtype=torch.long),
            torch.ones(E, device=device, dtype=torch.long),
        ])
        sequence = sequence + self.token_type_embedding(token_types)

        # Add position embeddings
        positions = torch.arange(N + E, device=device)
        sequence = sequence + self.position_embedding(positions)

        # Apply transformer
        decoded = self.transformer(sequence)
        decoded = self.norm(decoded)

        # Split back and apply prediction heads
        node_decoded = decoded[:, :N, :]
        edge_decoded = decoded[:, N:, :]

        node_logits = self.node_head(node_decoded)  # (B, N, num_atom_types)
        edge_logits = self.edge_head(edge_decoded)  # (B, E, num_bond_types)

        return node_logits, edge_logits
