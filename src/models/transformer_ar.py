"""Transformer-based Autoregressive Model for Molecule Generation.

Generates molecules by predicting adjacency matrix entries autoregressively.
The generation order is: first predict all node types, then predict adjacency
matrix entries row by row (upper triangular).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]


class TransformerARModel(nn.Module):
    """Autoregressive Transformer for molecule adjacency matrix generation.

    Generation sequence:
    1. Node types: [n1, n2, ..., n_max]
    2. Adjacency (upper triangular, row-major): [a_{0,1}, a_{0,2}, ..., a_{max-2,max-1}]

    Total sequence length = max_atoms + max_atoms*(max_atoms-1)//2
    """

    def __init__(
        self,
        max_atoms: int = 9,
        num_atom_types: int = 5,  # 0: empty, 1-4: atom types (e.g., C, N, O, F)
        num_bond_types: int = 5,  # 0: none, 1-4: bond types
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_atoms = max_atoms
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.d_model = d_model

        # Sequence lengths
        self.num_node_tokens = max_atoms
        self.num_edge_tokens = max_atoms * (max_atoms - 1) // 2
        self.seq_len = self.num_node_tokens + self.num_edge_tokens

        # Token embeddings (index 0 = empty/none for both)
        self.node_embedding = nn.Embedding(num_atom_types, d_model)
        self.edge_embedding = nn.Embedding(num_bond_types, d_model)

        # Token type embedding (0: node, 1: edge)
        self.token_type_embedding = nn.Embedding(2, d_model)

        # Position embedding
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.seq_len + 1)

        # Start token
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output heads
        self.node_head = nn.Linear(d_model, num_atom_types)
        self.edge_head = nn.Linear(d_model, num_bond_types)

        # Causal mask
        self._init_causal_mask()

    def _init_causal_mask(self):
        """Initialize causal attention mask."""
        mask = torch.triu(torch.ones(self.seq_len + 1, self.seq_len + 1), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def _flatten_adj_to_seq(self, adj: torch.Tensor) -> torch.Tensor:
        """Convert upper triangular adjacency matrix to sequence.

        Args:
            adj: (batch, max_atoms, max_atoms) adjacency matrix

        Returns:
            (batch, num_edge_tokens) flattened upper triangular
        """
        batch_size = adj.size(0)
        edge_seq = []
        for i in range(self.max_atoms):
            for j in range(i + 1, self.max_atoms):
                edge_seq.append(adj[:, i, j])
        return torch.stack(edge_seq, dim=1)

    def _seq_to_adj(self, edge_seq: torch.Tensor) -> torch.Tensor:
        """Convert sequence back to adjacency matrix.

        Args:
            edge_seq: (batch, num_edge_tokens) flattened upper triangular

        Returns:
            (batch, max_atoms, max_atoms) symmetric adjacency matrix
        """
        batch_size = edge_seq.size(0)
        adj = torch.zeros(batch_size, self.max_atoms, self.max_atoms,
                         dtype=edge_seq.dtype, device=edge_seq.device)
        idx = 0
        for i in range(self.max_atoms):
            for j in range(i + 1, self.max_atoms):
                adj[:, i, j] = edge_seq[:, idx]
                adj[:, j, i] = edge_seq[:, idx]  # Symmetric
                idx += 1
        return adj

    def _get_edge_position_info(self, seq_idx: int) -> Tuple[int, int]:
        """Get (row, col) indices for an edge sequence position."""
        idx = 0
        for i in range(self.max_atoms):
            for j in range(i + 1, self.max_atoms):
                if idx == seq_idx:
                    return (i, j)
                idx += 1
        raise ValueError(f"Invalid sequence index: {seq_idx}")

    def forward(
        self,
        node_types: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            node_types: (batch, max_atoms) with values in [0, num_atom_types)
            adj_matrix: (batch, max_atoms, max_atoms) with values in [0, num_bond_types)

        Returns:
            node_logits: (batch, max_atoms, num_atom_types)
            edge_logits: (batch, num_edge_tokens, num_bond_types)
        """
        batch_size = node_types.size(0)
        device = node_types.device

        # Flatten adjacency to sequence
        edge_seq = self._flatten_adj_to_seq(adj_matrix)  # (batch, num_edge_tokens)

        # Create input sequence (shifted right for autoregressive)
        # Start token + nodes[:-1] + edges[:-1]
        start = self.start_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)

        # Embed nodes (shift: use 0 to max_atoms-1 as input, predict 0 to max_atoms-1)
        node_embed = self.node_embedding(node_types)  # (batch, max_atoms, d_model)

        # Embed edges
        edge_embed = self.edge_embedding(edge_seq)  # (batch, num_edge_tokens, d_model)

        # Combine into single sequence: [start, node_0, ..., node_{n-1}, edge_0, ..., edge_{m-1}]
        # For predicting position i, we use [start, ..., token_{i-1}]
        seq_embed = torch.cat([start, node_embed, edge_embed], dim=1)  # (batch, seq_len+1, d_model)

        # Add token type embeddings
        token_types = torch.cat([
            torch.zeros(batch_size, 1 + self.num_node_tokens, dtype=torch.long, device=device),
            torch.ones(batch_size, self.num_edge_tokens, dtype=torch.long, device=device),
        ], dim=1)
        seq_embed = seq_embed + self.token_type_embedding(token_types)

        # Add positional encoding
        seq_embed = self.pos_encoding(seq_embed)

        # Apply transformer with causal mask
        # We use decoder-only architecture, so memory is just zeros
        memory = torch.zeros(batch_size, 1, self.d_model, device=device)
        output = self.transformer(
            seq_embed,
            memory,
            tgt_mask=self.causal_mask,
        )  # (batch, seq_len+1, d_model)

        # Split output into node and edge predictions
        # Position 0 predicts node 0, position max_atoms predicts edge 0, etc.
        node_output = output[:, :self.num_node_tokens]  # (batch, max_atoms, d_model)
        edge_output = output[:, self.num_node_tokens:-1]  # (batch, num_edge_tokens, d_model)

        node_logits = self.node_head(node_output)  # (batch, max_atoms, num_atom_types)
        edge_logits = self.edge_head(edge_output)  # (batch, num_edge_tokens, num_bond_types)

        return node_logits, edge_logits

    def compute_loss(
        self,
        node_types: torch.Tensor,
        adj_matrix: torch.Tensor,
        num_atoms: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute autoregressive loss.

        Args:
            node_types: (batch, max_atoms) ground truth node types
            adj_matrix: (batch, max_atoms, max_atoms) ground truth adjacency
            num_atoms: (batch,) actual number of atoms (for masking padding)

        Returns:
            loss: scalar loss
            metrics: dict with individual losses
        """
        batch_size = node_types.size(0)
        device = node_types.device

        # Get logits
        node_logits, edge_logits = self.forward(node_types, adj_matrix)

        # Compute node loss
        node_loss = F.cross_entropy(
            node_logits.reshape(-1, self.num_atom_types),
            node_types.reshape(-1),
            reduction='none',
        ).reshape(batch_size, -1)

        # Compute edge loss
        edge_targets = self._flatten_adj_to_seq(adj_matrix)
        edge_loss = F.cross_entropy(
            edge_logits.reshape(-1, self.num_bond_types),
            edge_targets.reshape(-1),
            reduction='none',
        ).reshape(batch_size, -1)

        # Apply masking if num_atoms provided
        if num_atoms is not None:
            # Create node mask
            node_mask = torch.arange(self.max_atoms, device=device).unsqueeze(0) < num_atoms.unsqueeze(1)
            node_loss = (node_loss * node_mask.float()).sum() / node_mask.sum()

            # Create edge mask (only edges between valid nodes)
            edge_mask = []
            for i in range(self.max_atoms):
                for j in range(i + 1, self.max_atoms):
                    valid = (i < num_atoms) & (j < num_atoms)
                    edge_mask.append(valid)
            edge_mask = torch.stack(edge_mask, dim=1).float()
            edge_loss = (edge_loss * edge_mask).sum() / edge_mask.sum().clamp(min=1)
        else:
            node_loss = node_loss.mean()
            edge_loss = edge_loss.mean()

        total_loss = node_loss + edge_loss

        metrics = {
            'loss': total_loss.item(),
            'node_loss': node_loss.item(),
            'edge_loss': edge_loss.item(),
        }

        return total_loss, metrics

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample molecules autoregressively.

        Args:
            batch_size: number of molecules to sample
            temperature: sampling temperature
            device: device to sample on

        Returns:
            node_types: (batch, max_atoms) sampled node types
            adj_matrix: (batch, max_atoms, max_atoms) sampled adjacency
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        # Initialize with start token
        seq_embed = self.start_token.expand(batch_size, -1, -1).clone()

        sampled_nodes = []
        sampled_edges = []

        # Sample nodes first
        for i in range(self.num_node_tokens):
            # Add token type and position encoding
            token_types = torch.zeros(batch_size, seq_embed.size(1), dtype=torch.long, device=device)
            curr_embed = seq_embed + self.token_type_embedding(token_types)
            curr_embed = self.pos_encoding(curr_embed)

            # Get causal mask for current length
            curr_mask = self.causal_mask[:seq_embed.size(1), :seq_embed.size(1)]

            # Forward pass
            memory = torch.zeros(batch_size, 1, self.d_model, device=device)
            output = self.transformer(curr_embed, memory, tgt_mask=curr_mask)

            # Get logits for next token
            logits = self.node_head(output[:, -1]) / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)

            sampled_nodes.append(next_token)

            # Embed and append
            next_embed = self.node_embedding(next_token).unsqueeze(1)
            seq_embed = torch.cat([seq_embed, next_embed], dim=1)

        # Sample edges
        for i in range(self.num_edge_tokens):
            # Add token type and position encoding
            token_types = torch.cat([
                torch.zeros(batch_size, 1 + self.num_node_tokens, dtype=torch.long, device=device),
                torch.ones(batch_size, i, dtype=torch.long, device=device),
            ], dim=1) if i > 0 else torch.zeros(batch_size, 1 + self.num_node_tokens, dtype=torch.long, device=device)

            curr_embed = seq_embed + self.token_type_embedding(token_types)
            curr_embed = self.pos_encoding(curr_embed)

            curr_mask = self.causal_mask[:seq_embed.size(1), :seq_embed.size(1)]

            memory = torch.zeros(batch_size, 1, self.d_model, device=device)
            output = self.transformer(curr_embed, memory, tgt_mask=curr_mask)

            logits = self.edge_head(output[:, -1]) / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)

            sampled_edges.append(next_token)

            next_embed = self.edge_embedding(next_token).unsqueeze(1)
            seq_embed = torch.cat([seq_embed, next_embed], dim=1)

        # Convert to tensors
        node_types = torch.stack(sampled_nodes, dim=1)
        edge_seq = torch.stack(sampled_edges, dim=1)
        adj_matrix = self._seq_to_adj(edge_seq)

        return node_types, adj_matrix


def build_model(config: dict) -> TransformerARModel:
    """Build model from config."""
    model_config = config['model']
    data_config = config['data']

    return TransformerARModel(
        max_atoms=data_config.get('max_atoms', 9),
        num_atom_types=model_config.get('num_atom_types', 5),
        num_bond_types=model_config.get('num_bond_types', 5),
        d_model=model_config.get('d_model', 256),
        nhead=model_config.get('nhead', 8),
        num_layers=model_config.get('num_layers', 6),
        dim_feedforward=model_config.get('dim_feedforward', 1024),
        dropout=model_config.get('dropout', 0.1),
    )
