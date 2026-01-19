"""Molecular Masked Autoencoder combining encoder and decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from src.models.mae.masking import MaskingStrategy, extract_upper_triangular
from src.models.mae.encoder import MAEEncoder
from src.models.mae.decoder import MAEDecoder


class MoleculeMAE(nn.Module):
    """Masked Autoencoder for molecules.

    Treats atoms and bonds as separate tokens, randomly masks them,
    and trains to reconstruct only the masked positions (BERT/MAE style).
    """

    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        d_model: int = 256,
        d_decoder: int = 128,
        nhead: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_atoms: int = 9,
        node_mask_ratio: float = 0.15,
        edge_mask_ratio: float = 0.50,
    ):
        """Initialize MoleculeMAE.

        Args:
            num_atom_types: Number of atom type classes
            num_bond_types: Number of bond type classes
            d_model: Encoder model dimension
            d_decoder: Decoder model dimension (smaller)
            nhead: Number of attention heads
            encoder_layers: Number of encoder transformer layers
            decoder_layers: Number of decoder transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            max_atoms: Maximum number of atoms (N)
            node_mask_ratio: Fraction of nodes to mask during training
            edge_mask_ratio: Fraction of edges to mask during training
        """
        super().__init__()

        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.d_model = d_model
        self.max_atoms = max_atoms
        self.num_edges = max_atoms * (max_atoms - 1) // 2

        # Masking strategy
        self.masking = MaskingStrategy(
            node_mask_ratio=node_mask_ratio,
            edge_mask_ratio=edge_mask_ratio,
        )

        # Encoder
        self.encoder = MAEEncoder(
            num_atom_types=num_atom_types,
            num_bond_types=num_bond_types,
            d_model=d_model,
            nhead=nhead,
            num_layers=encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_atoms=max_atoms,
        )

        # Decoder
        self.decoder = MAEDecoder(
            num_atom_types=num_atom_types,
            num_bond_types=num_bond_types,
            d_encoder=d_model,
            d_decoder=d_decoder,
            nhead=nhead // 2 if nhead > 2 else nhead,  # Smaller for decoder
            num_layers=decoder_layers,
            dim_feedforward=dim_feedforward // 2,
            dropout=dropout,
            max_atoms=max_atoms,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through MAE.

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)
            node_mask: Optional pre-computed node mask (B, N)
            edge_mask: Optional pre-computed edge mask (B, E)

        Returns:
            Dictionary with node_logits, edge_logits, node_mask, edge_mask
        """
        B, N, _ = node_features.shape
        device = node_features.device

        # Convert one-hot to indices
        node_types = node_features.argmax(dim=-1)  # (B, N)

        # Create masks if not provided
        if node_mask is None or edge_mask is None:
            node_mask, edge_mask = self.masking.create_mask(
                batch_size=B,
                num_nodes=N,
                num_edges=self.num_edges,
                device=device,
            )

        # Encode
        encoded = self.encoder(node_types, adj_matrix, node_mask, edge_mask)

        # Decode
        node_logits, edge_logits = self.decoder(encoded, node_mask, edge_mask)

        return {
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
        }

    def compute_loss(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        edge_class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute masked reconstruction loss.

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)
            edge_class_weights: Optional class weights for edge loss (handles imbalance)

        Returns:
            Dictionary with total_loss, node_loss, edge_loss, and accuracies
        """
        # Forward pass
        outputs = self.forward(node_features, adj_matrix)

        node_logits = outputs["node_logits"]  # (B, N, num_atom_types)
        edge_logits = outputs["edge_logits"]  # (B, E, num_bond_types)
        node_mask = outputs["node_mask"]  # (B, N)
        edge_mask = outputs["edge_mask"]  # (B, E)

        # Get targets
        node_targets = node_features.argmax(dim=-1)  # (B, N)
        edge_targets = extract_upper_triangular(adj_matrix)  # (B, E)

        # Compute node loss only on masked positions
        masked_node_logits = node_logits[node_mask]  # (num_masked_nodes, num_atom_types)
        masked_node_targets = node_targets[node_mask]  # (num_masked_nodes,)

        if masked_node_logits.numel() > 0:
            node_loss = F.cross_entropy(masked_node_logits, masked_node_targets)
            node_preds = masked_node_logits.argmax(dim=-1)
            node_accuracy = (node_preds == masked_node_targets).float().mean()
        else:
            node_loss = torch.tensor(0.0, device=node_logits.device)
            node_accuracy = torch.tensor(0.0, device=node_logits.device)

        # Compute edge loss only on masked positions
        masked_edge_logits = edge_logits[edge_mask]  # (num_masked_edges, num_bond_types)
        masked_edge_targets = edge_targets[edge_mask]  # (num_masked_edges,)

        if masked_edge_logits.numel() > 0:
            if edge_class_weights is not None:
                edge_loss = F.cross_entropy(
                    masked_edge_logits, masked_edge_targets, weight=edge_class_weights
                )
            else:
                edge_loss = F.cross_entropy(masked_edge_logits, masked_edge_targets)
            edge_preds = masked_edge_logits.argmax(dim=-1)
            edge_accuracy = (edge_preds == masked_edge_targets).float().mean()
        else:
            edge_loss = torch.tensor(0.0, device=edge_logits.device)
            edge_accuracy = torch.tensor(0.0, device=edge_logits.device)

        # Total loss (equal weighting)
        total_loss = node_loss + edge_loss

        return {
            "total_loss": total_loss,
            "node_loss": node_loss,
            "edge_loss": edge_loss,
            "node_accuracy": node_accuracy,
            "edge_accuracy": edge_accuracy,
            "num_masked_nodes": node_mask.sum(),
            "num_masked_edges": edge_mask.sum(),
        }

    def encode(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Encode molecules without masking (for inference).

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)

        Returns:
            encoded: Encoded representation (B, N+E, d_model)
        """
        node_types = node_features.argmax(dim=-1)
        B, N = node_types.shape
        device = node_types.device

        # No masking for encoding
        node_mask = torch.zeros(B, N, device=device, dtype=torch.bool)
        edge_mask = torch.zeros(B, self.num_edges, device=device, dtype=torch.bool)

        return self.encoder(node_types, adj_matrix, node_mask, edge_mask)

    def predict_all(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict all positions (no masking) for evaluation.

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)

        Returns:
            node_logits: (B, N, num_atom_types)
            edge_logits: (B, E, num_bond_types)
        """
        node_types = node_features.argmax(dim=-1)
        B, N = node_types.shape
        device = node_types.device

        # Mask everything for full reconstruction
        node_mask = torch.ones(B, N, device=device, dtype=torch.bool)
        edge_mask = torch.ones(B, self.num_edges, device=device, dtype=torch.bool)

        encoded = self.encoder(node_types, adj_matrix, node_mask, edge_mask)
        node_logits, edge_logits = self.decoder(encoded, node_mask, edge_mask)

        return node_logits, edge_logits


def compute_edge_class_weights(
    train_loader: torch.utils.data.DataLoader,
    num_bond_types: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute class weights for edge loss to handle imbalance.

    Most edges are "no bond" (type 0), so we weight other bond types higher.

    Args:
        train_loader: Training data loader
        num_bond_types: Number of bond types
        device: Device to create tensor on

    Returns:
        weights: Class weights tensor (num_bond_types,)
    """
    counts = torch.zeros(num_bond_types, device=device)

    for batch in train_loader:
        adj_matrix = batch['adj_matrix'].to(device)
        edges = extract_upper_triangular(adj_matrix)  # (B, E)
        for i in range(num_bond_types):
            counts[i] += (edges == i).sum()

    # Inverse frequency weighting
    total = counts.sum()
    weights = total / (counts + 1e-6)  # Add epsilon to avoid division by zero

    # Normalize so mean weight is 1
    weights = weights / weights.mean()

    return weights
