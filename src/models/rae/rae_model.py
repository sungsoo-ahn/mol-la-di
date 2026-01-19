"""Combined RAE (Representation Autoencoder) model.

Combines frozen MAE encoder with trainable RAE decoder.
Supports noise augmentation during training for diffusion robustness.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from src.models.rae.encoder_adapter import MAEEncoderAdapter
from src.models.rae.decoder import RAEDecoder
from src.models.rae.loss import compute_rae_loss, add_training_noise


class RAEModel(nn.Module):
    """Combined RAE model with frozen encoder and trainable decoder.

    Architecture:
        molecule -> [Frozen MAE Encoder] -> [Adapter Projection] -> z
                                                   |
                                        add_noise(z, sigma)
                                                   |
                                            [RAE Decoder] -> reconstruction
    """

    def __init__(
        self,
        encoder_adapter: MAEEncoderAdapter,
        decoder: RAEDecoder,
    ):
        """Initialize RAE model.

        Args:
            encoder_adapter: MAE encoder adapter (encoder frozen, projection trainable)
            decoder: RAE decoder (trainable)
        """
        super().__init__()
        self.encoder_adapter = encoder_adapter
        self.decoder = decoder

        # Store dimensions for convenience
        self.d_latent = encoder_adapter.d_latent
        self.max_atoms = encoder_adapter.max_atoms
        self.num_atom_types = decoder.num_atom_types
        self.num_bond_types = decoder.num_bond_types

    def encode(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Encode molecules to latent representations.

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)

        Returns:
            z: Latent representations (B, N, d_latent)
        """
        return self.encoder_adapter(node_features, adj_matrix)

    def decode(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latents to molecules.

        Args:
            z: Latent representations (B, N, d_latent)
            temperature: Sampling temperature
            hard: If True, return discrete values

        Returns:
            node_types: (B, N) atom type indices
            adj_matrix: (B, N, N) bond types
        """
        return self.decoder.decode(z, temperature=temperature, hard=hard)

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        noise_sigma: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder and decoder.

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)
            noise_sigma: Standard deviation of noise to add to latents

        Returns:
            Dictionary with node_logits, edge_logits, and latents
        """
        # Encode
        z = self.encode(node_features, adj_matrix)

        # Add noise augmentation (for training robustness)
        if self.training and noise_sigma > 0:
            z_noisy = add_training_noise(z, noise_sigma)
        else:
            z_noisy = z

        # Decode
        node_logits, edge_logits = self.decoder(z_noisy)

        return {
            'node_logits': node_logits,
            'edge_logits': edge_logits,
            'z': z,
            'z_noisy': z_noisy if self.training else z,
        }

    def compute_loss(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        noise_sigma: float = 0.0,
        lambda_node: float = 1.0,
        lambda_edge: float = 1.0,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        edge_class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute reconstruction loss with noise augmentation.

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)
            noise_sigma: Standard deviation of noise to add to latents
            lambda_node: Weight for node loss
            lambda_edge: Weight for edge loss
            label_smoothing: Label smoothing for cross-entropy
            focal_gamma: Gamma parameter for focal loss
            edge_class_weights: Optional class weights for edge loss

        Returns:
            Dictionary with losses and metrics
        """
        # Forward pass
        outputs = self.forward(node_features, adj_matrix, noise_sigma=noise_sigma)

        # Compute loss
        loss_dict = compute_rae_loss(
            node_logits=outputs['node_logits'],
            edge_logits=outputs['edge_logits'],
            node_targets=node_features,  # One-hot
            adj_targets=adj_matrix,
            lambda_node=lambda_node,
            lambda_edge=lambda_edge,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            edge_class_weights=edge_class_weights,
        )

        # Add latent stats for monitoring
        z = outputs['z']
        loss_dict['z_mean'] = z.mean()
        loss_dict['z_std'] = z.std()

        return loss_dict

    def reconstruct(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct molecules (encode then decode).

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)
            temperature: Sampling temperature

        Returns:
            node_types: Reconstructed atom types (B, N)
            adj_matrix: Reconstructed adjacency matrix (B, N, N)
        """
        z = self.encode(node_features, adj_matrix)
        return self.decode(z, temperature=temperature, hard=True)

    @torch.no_grad()
    def get_latent(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Get latent representations without gradients.

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)

        Returns:
            z: Latent representations (B, N, d_latent)
        """
        return self.encode(node_features, adj_matrix)


def build_rae_model(
    encoder_adapter: MAEEncoderAdapter,
    config: dict,
) -> RAEModel:
    """Build RAE model from config.

    Args:
        encoder_adapter: Pre-created MAE encoder adapter
        config: Configuration dictionary with decoder settings

    Returns:
        RAE model
    """
    decoder_config = config.get('rae_decoder', {})
    data_config = config.get('data', {})

    decoder = RAEDecoder(
        num_atom_types=config['model']['num_atom_types'],
        num_bond_types=config['model']['num_bond_types'],
        d_latent=encoder_adapter.d_latent,
        d_model=decoder_config.get('d_model', 512),
        nhead=decoder_config.get('nhead', 8),
        num_layers=decoder_config.get('num_layers', 6),
        dim_feedforward=decoder_config.get('dim_feedforward', 2048),
        dropout=decoder_config.get('dropout', 0.0),
        max_atoms=data_config.get('max_atoms', encoder_adapter.max_atoms),
    )

    return RAEModel(encoder_adapter, decoder)
