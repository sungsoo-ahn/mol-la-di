"""MAE Encoder Adapter for RAE.

Adapts the MAE encoder output for use with RAE decoder and diffusion models.
The MAE encoder is frozen, while the projection layer is trainable.
"""

import torch
import torch.nn as nn

from src.models.mae import MoleculeMAE
from src.models.mae.encoder import MAEEncoder


class MAEEncoderAdapter(nn.Module):
    """Adapts MAE encoder output for RAE decoder and diffusion.

    The MAE encoder outputs (B, N+E, d_model) where N=max_atoms nodes
    and E=N*(N-1)/2 edges. This adapter:
    1. Extracts node representations: first N tokens -> (B, N, d_model)
    2. Projects to latent dimension: Linear(d_model, d_latent) -> (B, N, d_latent)

    The MAE encoder is frozen, only the projection layer is trainable.
    """

    def __init__(
        self,
        mae_encoder: MAEEncoder,
        d_latent: int,
        max_atoms: int,
        freeze_encoder: bool = True,
    ):
        """Initialize MAE encoder adapter.

        Args:
            mae_encoder: Pretrained MAE encoder module
            d_latent: Latent dimension for output
            max_atoms: Maximum number of atoms (N)
            freeze_encoder: Whether to freeze the MAE encoder (default: True)
        """
        super().__init__()

        self.mae_encoder = mae_encoder
        self.d_model = mae_encoder.d_model
        self.d_latent = d_latent
        self.max_atoms = max_atoms

        # Trainable projection layer
        self.proj = nn.Linear(self.d_model, d_latent)

        # Freeze MAE encoder if requested
        if freeze_encoder:
            for param in self.mae_encoder.parameters():
                param.requires_grad = False

    def forward(
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
        return self.encode(node_features, adj_matrix)

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
        B, N, _ = node_features.shape
        device = node_features.device

        # Convert one-hot to indices
        node_types = node_features.argmax(dim=-1)  # (B, N)

        # Create no-masking masks (all False = nothing masked)
        node_mask = torch.zeros(B, N, device=device, dtype=torch.bool)
        edge_mask = torch.zeros(B, self.mae_encoder.num_edges, device=device, dtype=torch.bool)

        # Encode with frozen MAE
        with torch.no_grad():
            encoded = self.mae_encoder(node_types, adj_matrix, node_mask, edge_mask)
            # encoded shape: (B, N+E, d_model)

        # Extract node tokens only (first N positions)
        node_encoded = encoded[:, :self.max_atoms, :]  # (B, N, d_model)

        # Project to latent dimension (this is trainable)
        z = self.proj(node_encoded)  # (B, N, d_latent)

        return z

    @torch.no_grad()
    def encode_no_grad(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Encode molecules without computing gradients (for inference/encoding datasets).

        Args:
            node_features: One-hot atom types (B, N, num_atom_types)
            adj_matrix: Bond types (B, N, N)

        Returns:
            z: Latent representations (B, N, d_latent)
        """
        return self.encode(node_features, adj_matrix)


def load_mae_encoder_adapter(
    mae_checkpoint: str,
    d_latent: int,
    device: torch.device,
    freeze_encoder: bool = True,
) -> MAEEncoderAdapter:
    """Load MAE encoder adapter from checkpoint.

    Args:
        mae_checkpoint: Path to MAE checkpoint
        d_latent: Latent dimension for adapter output
        device: Device to load model on
        freeze_encoder: Whether to freeze the MAE encoder

    Returns:
        MAEEncoderAdapter with loaded MAE encoder
    """
    checkpoint = torch.load(mae_checkpoint, map_location='cpu', weights_only=False)

    config = checkpoint['config']
    model_config = config['model']
    data_config = config['data']
    masking_config = config.get('masking', {})

    # Build MAE model to get the encoder
    mae = MoleculeMAE(
        num_atom_types=model_config['num_atom_types'],
        num_bond_types=model_config['num_bond_types'],
        d_model=model_config.get('d_model', 256),
        d_decoder=model_config.get('d_decoder', 128),
        nhead=model_config.get('nhead', 8),
        encoder_layers=model_config.get('encoder_layers', 6),
        decoder_layers=model_config.get('decoder_layers', 2),
        dim_feedforward=model_config.get('dim_feedforward', 1024),
        dropout=model_config.get('dropout', 0.1),
        max_atoms=data_config['max_atoms'],
        node_mask_ratio=masking_config.get('node_mask_ratio', 0.15),
        edge_mask_ratio=masking_config.get('edge_mask_ratio', 0.50),
    )

    mae.load_state_dict(checkpoint['model_state_dict'])

    # Create adapter with just the encoder
    adapter = MAEEncoderAdapter(
        mae_encoder=mae.encoder,
        d_latent=d_latent,
        max_atoms=data_config['max_atoms'],
        freeze_encoder=freeze_encoder,
    )

    adapter = adapter.to(device)
    return adapter
