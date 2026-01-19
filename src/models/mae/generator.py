"""MAE Generator - Iterative unmasking for molecule generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

from src.models.mae.mae import MoleculeMAE
from src.models.mae.masking import extract_upper_triangular, reconstruct_adj_from_edges


class MAEGenerator:
    """Iterative unmasking generator for molecular MAE.

    Generates molecules by starting with fully masked tokens and
    progressively unmasking the most confident predictions.
    """

    def __init__(
        self,
        model: MoleculeMAE,
        num_iterations: int = 10,
        temperature: float = 1.0,
        nucleus_p: float = 0.9,
    ):
        """Initialize generator.

        Args:
            model: Trained MoleculeMAE model
            num_iterations: Number of unmasking iterations
            temperature: Sampling temperature (higher = more random)
            nucleus_p: Top-p sampling threshold (1.0 = no nucleus sampling)
        """
        self.model = model
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.nucleus_p = nucleus_p

        self.num_atoms = model.max_atoms
        self.num_edges = model.num_edges
        self.num_atom_types = model.num_atom_types
        self.num_bond_types = model.num_bond_types

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        device: torch.device,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate molecules through iterative unmasking.

        Args:
            batch_size: Number of molecules to generate
            device: Device to generate on
            return_intermediates: If True, return intermediate states

        Returns:
            node_types: Generated atom types (B, N)
            adj_matrix: Generated adjacency matrix (B, N, N)
        """
        self.model.eval()

        N = self.num_atoms
        E = self.num_edges

        # Start with fully masked tokens
        # Initialize with random values (will be masked anyway)
        node_types = torch.zeros(batch_size, N, dtype=torch.long, device=device)
        edge_types = torch.zeros(batch_size, E, dtype=torch.long, device=device)

        # All positions are masked initially
        node_mask = torch.ones(batch_size, N, dtype=torch.bool, device=device)
        edge_mask = torch.ones(batch_size, E, dtype=torch.bool, device=device)

        # Track confidence scores for ordering
        node_confidences = torch.zeros(batch_size, N, device=device)
        edge_confidences = torch.zeros(batch_size, E, device=device)

        intermediates = [] if return_intermediates else None

        # Compute number of tokens to unmask per iteration
        total_node_masked = N
        total_edge_masked = E
        nodes_per_iter = max(1, N // self.num_iterations)
        edges_per_iter = max(1, E // self.num_iterations)

        for step in range(self.num_iterations):
            # Convert current state to one-hot for model input
            node_features = F.one_hot(node_types, self.num_atom_types).float()
            adj_matrix = reconstruct_adj_from_edges(edge_types.float(), N).long()

            # Get predictions
            encoded = self.model.encoder(node_types, adj_matrix, node_mask, edge_mask)
            node_logits, edge_logits = self.model.decoder(encoded, node_mask, edge_mask)

            # Apply temperature
            node_logits = node_logits / self.temperature
            edge_logits = edge_logits / self.temperature

            # Get probabilities
            node_probs = F.softmax(node_logits, dim=-1)
            edge_probs = F.softmax(edge_logits, dim=-1)

            # Sample new values for masked positions
            if self.nucleus_p < 1.0:
                node_samples = self._nucleus_sample(node_probs)
                edge_samples = self._nucleus_sample(edge_probs)
            else:
                node_samples = torch.multinomial(
                    node_probs.view(-1, self.num_atom_types), 1
                ).view(batch_size, N)
                edge_samples = torch.multinomial(
                    edge_probs.view(-1, self.num_bond_types), 1
                ).view(batch_size, E)

            # Compute confidence as max probability
            node_max_probs = node_probs.max(dim=-1).values  # (B, N)
            edge_max_probs = edge_probs.max(dim=-1).values  # (B, E)

            # Update tokens at masked positions
            node_types = torch.where(node_mask, node_samples, node_types)
            edge_types = torch.where(edge_mask, edge_samples, edge_types)

            # Update confidences for masked positions
            node_confidences = torch.where(node_mask, node_max_probs, node_confidences)
            edge_confidences = torch.where(edge_mask, edge_max_probs, edge_confidences)

            # Determine how many tokens to unmask this iteration
            # Unmask more aggressively in later iterations
            progress = (step + 1) / self.num_iterations
            target_node_unmasked = min(N, int(N * progress) + 1)
            target_edge_unmasked = min(E, int(E * progress) + 1)

            # Unmask the most confident masked positions
            node_mask = self._unmask_top_k(
                node_mask, node_confidences, N - target_node_unmasked
            )
            edge_mask = self._unmask_top_k(
                edge_mask, edge_confidences, E - target_edge_unmasked
            )

            if return_intermediates:
                intermediates.append({
                    'step': step,
                    'node_types': node_types.clone(),
                    'edge_types': edge_types.clone(),
                    'node_mask': node_mask.clone(),
                    'edge_mask': edge_mask.clone(),
                })

        # Final pass: ensure all positions are unmasked
        node_features = F.one_hot(node_types, self.num_atom_types).float()
        adj_matrix = reconstruct_adj_from_edges(edge_types.float(), N).long()

        # Final prediction with no masking
        node_mask_final = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
        edge_mask_final = torch.zeros(batch_size, E, dtype=torch.bool, device=device)

        encoded = self.model.encoder(node_types, adj_matrix, node_mask_final, edge_mask_final)
        node_logits, edge_logits = self.model.decoder(encoded, node_mask_final, edge_mask_final)

        # Take argmax for final predictions
        node_types = node_logits.argmax(dim=-1)
        edge_types = edge_logits.argmax(dim=-1)

        # Reconstruct adjacency matrix
        adj_matrix = reconstruct_adj_from_edges(edge_types.float(), N).long()

        if return_intermediates:
            return node_types, adj_matrix, intermediates

        return node_types, adj_matrix

    def _nucleus_sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample using nucleus (top-p) sampling.

        Args:
            probs: Probability distribution (B, seq_len, num_classes)

        Returns:
            samples: Sampled indices (B, seq_len)
        """
        B, seq_len, num_classes = probs.shape

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask for nucleus
        nucleus_mask = cumsum_probs <= self.nucleus_p
        # Always keep at least one token
        nucleus_mask[..., 0] = True

        # Zero out non-nucleus probabilities
        filtered_probs = sorted_probs * nucleus_mask.float()
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

        # Sample from filtered distribution
        samples_in_sorted = torch.multinomial(
            filtered_probs.view(-1, num_classes), 1
        ).view(B, seq_len)

        # Map back to original indices
        samples = torch.gather(sorted_indices, -1, samples_in_sorted.unsqueeze(-1)).squeeze(-1)

        return samples

    def _unmask_top_k(
        self,
        mask: torch.Tensor,
        confidences: torch.Tensor,
        target_masked: int,
    ) -> torch.Tensor:
        """Unmask positions with highest confidence.

        Args:
            mask: Current mask (B, seq_len), True = masked
            confidences: Confidence scores (B, seq_len)
            target_masked: Target number of masked positions

        Returns:
            new_mask: Updated mask
        """
        B, seq_len = mask.shape
        device = mask.device

        # Only consider currently masked positions
        masked_confidences = torch.where(
            mask,
            confidences,
            torch.tensor(float('-inf'), device=device),
        )

        # Get current masked count per sample
        current_masked = mask.sum(dim=-1)  # (B,)

        # Compute how many to unmask per sample
        num_to_unmask = (current_masked - target_masked).clamp(min=0)  # (B,)

        new_mask = mask.clone()

        for b in range(B):
            if num_to_unmask[b] > 0:
                # Get indices of top-k confident masked positions
                k = int(num_to_unmask[b].item())
                _, top_indices = torch.topk(masked_confidences[b], k)
                new_mask[b, top_indices] = False

        return new_mask

    @torch.no_grad()
    def generate_parallel_decoding(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate using parallel decoding (faster but potentially lower quality).

        Predicts all positions in one forward pass.

        Args:
            batch_size: Number of molecules to generate
            device: Device to generate on

        Returns:
            node_types: Generated atom types (B, N)
            adj_matrix: Generated adjacency matrix (B, N, N)
        """
        self.model.eval()

        N = self.num_atoms
        E = self.num_edges

        # Start with random tokens
        node_types = torch.randint(0, self.num_atom_types, (batch_size, N), device=device)
        edge_types = torch.randint(0, self.num_bond_types, (batch_size, E), device=device)

        # All positions masked
        node_mask = torch.ones(batch_size, N, dtype=torch.bool, device=device)
        edge_mask = torch.ones(batch_size, E, dtype=torch.bool, device=device)

        adj_matrix = reconstruct_adj_from_edges(edge_types.float(), N).long()

        # Single forward pass
        encoded = self.model.encoder(node_types, adj_matrix, node_mask, edge_mask)
        node_logits, edge_logits = self.model.decoder(encoded, node_mask, edge_mask)

        # Apply temperature and sample
        node_logits = node_logits / self.temperature
        edge_logits = edge_logits / self.temperature

        node_types = torch.multinomial(
            F.softmax(node_logits, dim=-1).view(-1, self.num_atom_types), 1
        ).view(batch_size, N)
        edge_types = torch.multinomial(
            F.softmax(edge_logits, dim=-1).view(-1, self.num_bond_types), 1
        ).view(batch_size, E)

        adj_matrix = reconstruct_adj_from_edges(edge_types.float(), N).long()

        return node_types, adj_matrix
