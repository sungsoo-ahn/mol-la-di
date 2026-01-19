"""RAE-style decoder with noise-robust reconstruction.

Following the RAE (Representation Autoencoder) approach:
- Deeper architecture than standard VAE decoder
- Wider FFN following DDT head design
- Pairwise bilinear edge prediction
- Designed to be robust to noisy latent inputs from diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RAEDecoder(nn.Module):
    """RAE-style decoder with noise-robust reconstruction.

    Improvements over standard OneShotDecoder:
    1. Deeper architecture (6 vs 4 layers)
    2. Wider FFN (2048-dim following DDT head design)
    3. Pairwise bilinear edge prediction (more expressive than concatenation)
    4. No dropout (following diffusion conventions)
    """

    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        d_latent: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        max_atoms: int = 9,
    ):
        """Initialize RAE decoder.

        Args:
            num_atom_types: Number of atom type classes
            num_bond_types: Number of bond type classes
            d_latent: Input latent dimension
            d_model: Model dimension (wider than typical VAE decoder)
            nhead: Number of attention heads
            num_layers: Number of transformer layers (deeper than typical VAE decoder)
            dim_feedforward: Feedforward dimension (DDT-style wide)
            dropout: Dropout probability (0.0 following diffusion conventions)
            max_atoms: Maximum number of atoms
        """
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
        # Using pre-norm (norm_first=True) for better training stability
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

        # Node head: 2-layer MLP -> num_atom_types
        self.node_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_atom_types),
        )

        # Edge head: Bilinear approach for more expressive pairwise prediction
        # Query and Key projections for bilinear attention
        self.edge_query = nn.Linear(d_model, d_model)
        self.edge_key = nn.Linear(d_model, d_model)

        # MLP to combine bilinear features for bond type prediction
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_bond_types),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize projection layers
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Initialize edge projections
        nn.init.xavier_uniform_(self.edge_query.weight)
        nn.init.zeros_(self.edge_query.bias)
        nn.init.xavier_uniform_(self.edge_key.weight)
        nn.init.zeros_(self.edge_key.bias)

    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through decoder.

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

        # Edge predictions via bilinear approach
        # Q(x_i) * K(x_j) -> more expressive than concatenation
        q = self.edge_query(x)  # (B, N, d_model)
        k = self.edge_key(x)    # (B, N, d_model)

        # Compute pairwise bilinear features
        # For each pair (i, j), compute element-wise product of Q_i and K_j
        q_expanded = q.unsqueeze(2)  # (B, N, 1, d_model)
        k_expanded = k.unsqueeze(1)  # (B, 1, N, d_model)
        pair_features = q_expanded * k_expanded  # (B, N, N, d_model)

        # Predict bond types
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
