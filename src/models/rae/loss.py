"""Loss functions and noise augmentation for RAE decoder training.

Following the RAE paper:
- Focal loss for handling "no bond" class imbalance
- Noise augmentation for diffusion robustness
- Noise curriculum for gradual noise increase during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Particularly useful for edge prediction where "no bond" (class 0)
    dominates the distribution.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        """Initialize focal loss.

        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class weights tensor (optional)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits (N, C) where C is number of classes
            targets: Target classes (N,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t

        # Apply focal term
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def add_training_noise(
    z: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Add Gaussian noise to latent representations during training.

    Following RAE paper: z_noisy = z + n where n ~ N(0, sigma^2 * I)

    Args:
        z: Latent representations (B, N, d_latent)
        sigma: Standard deviation of noise

    Returns:
        Noisy latent representations
    """
    if sigma <= 0:
        return z

    noise = torch.randn_like(z) * sigma
    return z + noise


def get_noise_sigma(
    epoch: int,
    sigma_min: float = 0.0,
    sigma_max: float = 1.0,
    curriculum: bool = True,
    warmup_epochs: int = 50,
    rampup_epochs: int = 100,
) -> float:
    """Get noise sigma following curriculum schedule.

    Curriculum schedule (following RAE paper):
    - Epochs 1-warmup: sigma ~ U(0, sigma_max * 0.3)
    - Epochs warmup-rampup: sigma ~ U(0, sigma_max * 0.6)
    - Epochs rampup+: sigma ~ U(0, sigma_max)

    Args:
        epoch: Current epoch (1-indexed)
        sigma_min: Minimum sigma (usually 0)
        sigma_max: Maximum sigma (usually 1.0)
        curriculum: Whether to use curriculum (if False, always use full range)
        warmup_epochs: End of warmup phase
        rampup_epochs: End of rampup phase (after warmup)

    Returns:
        Sampled sigma value for this epoch
    """
    if not curriculum:
        # No curriculum, sample from full range
        return sigma_min + (sigma_max - sigma_min) * torch.rand(1).item()

    # Curriculum schedule
    if epoch <= warmup_epochs:
        # Phase 1: Low noise
        max_sigma = sigma_max * 0.3
    elif epoch <= warmup_epochs + rampup_epochs:
        # Phase 2: Medium noise
        max_sigma = sigma_max * 0.6
    else:
        # Phase 3: Full noise
        max_sigma = sigma_max

    # Sample uniformly from [sigma_min, max_sigma]
    return sigma_min + (max_sigma - sigma_min) * torch.rand(1).item()


def compute_rae_loss(
    node_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    node_targets: torch.Tensor,
    adj_targets: torch.Tensor,
    lambda_node: float = 1.0,
    lambda_edge: float = 1.0,
    label_smoothing: float = 0.1,
    focal_gamma: float = 2.0,
    edge_class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute RAE reconstruction loss.

    Args:
        node_logits: Node predictions (B, N, num_atom_types)
        edge_logits: Edge predictions (B, N, N, num_bond_types)
        node_targets: Target node types (B, N) or one-hot (B, N, num_atom_types)
        adj_targets: Target adjacency matrix (B, N, N)
        lambda_node: Weight for node loss
        lambda_edge: Weight for edge loss
        label_smoothing: Label smoothing for cross-entropy
        focal_gamma: Gamma parameter for focal loss
        edge_class_weights: Optional class weights for edge loss

    Returns:
        Dictionary with losses and metrics
    """
    B, N = node_targets.shape[:2]
    device = node_logits.device

    # Handle one-hot node targets
    if node_targets.dim() == 3:
        node_targets = node_targets.argmax(dim=-1)

    # Node loss: Cross-entropy with label smoothing
    node_logits_flat = node_logits.view(-1, node_logits.size(-1))
    node_targets_flat = node_targets.view(-1)

    node_loss = F.cross_entropy(
        node_logits_flat,
        node_targets_flat,
        label_smoothing=label_smoothing,
    )

    # Node accuracy
    node_preds = node_logits.argmax(dim=-1)
    node_accuracy = (node_preds == node_targets).float().mean()

    # Edge loss: Focal loss to handle "no bond" class imbalance
    # Only use upper triangular (avoid double counting symmetric edges)
    mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
    mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

    edge_logits_upper = edge_logits[mask]  # (B * num_edges, num_bond_types)
    edge_targets_upper = adj_targets[mask].long()  # (B * num_edges,)

    # Focal loss for edges
    focal_loss = FocalLoss(gamma=focal_gamma, alpha=edge_class_weights)
    edge_loss = focal_loss(edge_logits_upper, edge_targets_upper)

    # Edge accuracy
    edge_preds_upper = edge_logits_upper.argmax(dim=-1)
    edge_accuracy = (edge_preds_upper == edge_targets_upper).float().mean()

    # Total loss
    total_loss = lambda_node * node_loss + lambda_edge * edge_loss

    return {
        'total_loss': total_loss,
        'node_loss': node_loss,
        'edge_loss': edge_loss,
        'node_accuracy': node_accuracy,
        'edge_accuracy': edge_accuracy,
    }


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
        B, N, _ = adj_matrix.shape

        # Only count upper triangular to avoid double counting
        mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
        edges = adj_matrix[:, mask]  # (B, num_edges)

        for i in range(num_bond_types):
            counts[i] += (edges == i).sum()

    # Inverse frequency weighting
    total = counts.sum()
    weights = total / (counts + 1e-6)  # Add epsilon to avoid division by zero

    # Normalize so mean weight is 1
    weights = weights / weights.mean()

    return weights
