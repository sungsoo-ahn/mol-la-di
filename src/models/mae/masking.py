"""Masking strategies for Masked Autoencoder."""

import torch
from typing import Tuple


class MaskingStrategy:
    """Masking strategy for MAE training.

    Independently masks nodes and edges with specified ratios.
    """

    def __init__(
        self,
        node_mask_ratio: float = 0.15,
        edge_mask_ratio: float = 0.50,
    ):
        """Initialize masking strategy.

        Args:
            node_mask_ratio: Fraction of nodes to mask (BERT-style, ~15%)
            edge_mask_ratio: Fraction of edges to mask (MAE-style, ~50%)
        """
        self.node_mask_ratio = node_mask_ratio
        self.edge_mask_ratio = edge_mask_ratio

    def create_mask(
        self,
        batch_size: int,
        num_nodes: int,
        num_edges: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create random masks for nodes and edges.

        Args:
            batch_size: Batch size (B)
            num_nodes: Number of nodes per molecule (N)
            num_edges: Number of edges per molecule (E = N*(N-1)/2)
            device: Device to create tensors on

        Returns:
            node_mask: Boolean mask (B, N), True = masked
            edge_mask: Boolean mask (B, E), True = masked
        """
        node_mask = torch.rand(batch_size, num_nodes, device=device) < self.node_mask_ratio
        edge_mask = torch.rand(batch_size, num_edges, device=device) < self.edge_mask_ratio

        # Ensure at least one position is masked in each batch element
        # to avoid division by zero in loss computation
        for i in range(batch_size):
            if not node_mask[i].any():
                # Randomly select one node to mask
                idx = torch.randint(0, num_nodes, (1,), device=device)
                node_mask[i, idx] = True

            if not edge_mask[i].any():
                # Randomly select one edge to mask
                idx = torch.randint(0, num_edges, (1,), device=device)
                edge_mask[i, idx] = True

        return node_mask, edge_mask

    def get_edge_indices(self, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get row and column indices for upper triangular edges.

        For a molecule with N nodes, there are N*(N-1)/2 unique undirected edges.
        This returns the mapping from edge index to (i, j) pairs.

        Args:
            num_nodes: Number of nodes (N)

        Returns:
            row_indices: (E,) tensor of row indices
            col_indices: (E,) tensor of column indices
        """
        rows = []
        cols = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                rows.append(i)
                cols.append(j)
        return torch.tensor(rows), torch.tensor(cols)


def extract_upper_triangular(adj_matrix: torch.Tensor) -> torch.Tensor:
    """Extract upper triangular edges from adjacency matrix.

    Args:
        adj_matrix: Adjacency matrix (B, N, N)

    Returns:
        edges: Flattened upper triangular values (B, N*(N-1)/2)
    """
    B, N, _ = adj_matrix.shape
    # Create upper triangular mask (exclude diagonal)
    triu_mask = torch.triu(torch.ones(N, N, device=adj_matrix.device), diagonal=1).bool()
    # Extract upper triangular values: (B, N*(N-1)/2)
    edges = adj_matrix[:, triu_mask]
    return edges


def reconstruct_adj_from_edges(edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Reconstruct symmetric adjacency matrix from upper triangular edges.

    Args:
        edges: Upper triangular edge values (B, N*(N-1)/2)
        num_nodes: Number of nodes (N)

    Returns:
        adj_matrix: Symmetric adjacency matrix (B, N, N)
    """
    B = edges.shape[0]
    device = edges.device

    # Create output adjacency matrix
    adj_matrix = torch.zeros(B, num_nodes, num_nodes, device=device, dtype=edges.dtype)

    # Create upper triangular mask
    triu_mask = torch.triu(torch.ones(num_nodes, num_nodes, device=device), diagonal=1).bool()

    # Fill upper triangular
    adj_matrix[:, triu_mask] = edges

    # Make symmetric
    adj_matrix = adj_matrix + adj_matrix.transpose(1, 2)

    return adj_matrix
