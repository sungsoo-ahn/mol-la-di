"""Training script for RAE decoder.

Trains the RAE decoder with frozen MAE encoder.
Uses noise augmentation for robustness to diffusion outputs.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from src.utils import load_config, setup_output_dir, set_seed, get_device, count_parameters
from src.data.molecule_dataset import get_dataloader
from src.models.rae import (
    MAEEncoderAdapter,
    RAEDecoder,
    RAEModel,
    compute_edge_class_weights,
    get_noise_sigma,
)
from src.models.rae.encoder_adapter import load_mae_encoder_adapter
from src.evaluation import MoleculeEvaluator, adj_to_mol, mol_to_smiles


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger('train_rae')
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(output_dir / 'logs' / 'train.log')
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def _filter_wandb_metrics(metrics: dict, prefix: str = '') -> dict:
    """Filter metrics for W&B logging (numeric, non-NaN values only)."""
    result = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            full_key = f'{prefix}/{key}' if prefix else key
            result[full_key] = value
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            val = value.item()
            if not np.isnan(val):
                full_key = f'{prefix}/{key}' if prefix else key
                result[full_key] = val
    return result


def build_rae_decoder(config: dict, d_latent: int, max_atoms: int) -> RAEDecoder:
    """Build RAE decoder from config."""
    decoder_config = config.get('rae_decoder', {})
    model_config = config['model']

    return RAEDecoder(
        num_atom_types=model_config['num_atom_types'],
        num_bond_types=model_config['num_bond_types'],
        d_latent=d_latent,
        d_model=decoder_config.get('d_model', 512),
        nhead=decoder_config.get('nhead', 8),
        num_layers=decoder_config.get('num_layers', 6),
        dim_feedforward=decoder_config.get('dim_feedforward', 2048),
        dropout=decoder_config.get('dropout', 0.0),
        max_atoms=max_atoms,
    )


def train_epoch(
    model: RAEModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    edge_class_weights: torch.Tensor = None,
) -> dict:
    """Train for one epoch."""
    model.train()

    train_config = config['training']
    loss_config = config.get('loss', {})
    noise_config = config.get('noise_augmentation', {})

    # Get noise sigma for this epoch (following curriculum)
    if noise_config.get('enabled', True):
        sigma = get_noise_sigma(
            epoch=epoch,
            sigma_min=noise_config.get('sigma_min', 0.0),
            sigma_max=noise_config.get('sigma_max', 1.0),
            curriculum=noise_config.get('curriculum', True),
            warmup_epochs=noise_config.get('warmup_epochs', 50),
            rampup_epochs=noise_config.get('rampup_epochs', 100),
        )
    else:
        sigma = 0.0

    # Apply LR warmup
    warmup_epochs = train_config.get('warmup_epochs', 0)
    base_lr = train_config.get('lr', 1e-4)
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        warmup_factor = epoch / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * warmup_factor

    totals = {
        'total_loss': 0.0,
        'node_loss': 0.0,
        'edge_loss': 0.0,
        'node_accuracy': 0.0,
        'edge_accuracy': 0.0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        node_features = batch['node_features'].to(device)
        adj_matrix = batch['adj_matrix'].to(device)

        optimizer.zero_grad()

        loss_dict = model.compute_loss(
            node_features,
            adj_matrix,
            noise_sigma=sigma,
            lambda_node=loss_config.get('lambda_node', 1.0),
            lambda_edge=loss_config.get('lambda_edge', 1.0),
            label_smoothing=loss_config.get('label_smoothing', 0.1),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
            edge_class_weights=edge_class_weights,
        )

        loss = loss_dict['total_loss']
        loss.backward()

        max_grad_norm = train_config.get('max_grad_norm', 1.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        for key in totals:
            if key in loss_dict:
                val = loss_dict[key]
                totals[key] += val.item() if isinstance(val, torch.Tensor) else val
        n_batches += 1

        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss'].item():.4f}",
            'node_acc': f"{loss_dict['node_accuracy'].item():.4f}",
            'edge_acc': f"{loss_dict['edge_accuracy'].item():.4f}",
            'sigma': f"{sigma:.3f}",
        })

    result = {key: value / n_batches for key, value in totals.items()}
    result['noise_sigma'] = sigma
    return result


@torch.no_grad()
def validate(
    model: RAEModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict,
    edge_class_weights: torch.Tensor = None,
) -> dict:
    """Validate the model (without noise augmentation)."""
    model.eval()

    loss_config = config.get('loss', {})

    totals = {
        'total_loss': 0.0,
        'node_loss': 0.0,
        'edge_loss': 0.0,
        'node_accuracy': 0.0,
        'edge_accuracy': 0.0,
    }
    n_batches = 0

    for batch in dataloader:
        node_features = batch['node_features'].to(device)
        adj_matrix = batch['adj_matrix'].to(device)

        # No noise during validation
        loss_dict = model.compute_loss(
            node_features,
            adj_matrix,
            noise_sigma=0.0,
            lambda_node=loss_config.get('lambda_node', 1.0),
            lambda_edge=loss_config.get('lambda_edge', 1.0),
            label_smoothing=loss_config.get('label_smoothing', 0.1),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
            edge_class_weights=edge_class_weights,
        )

        for key in totals:
            if key in loss_dict:
                val = loss_dict[key]
                totals[key] += val.item() if isinstance(val, torch.Tensor) else val
        n_batches += 1

    return {key: value / n_batches for key, value in totals.items()}


@torch.no_grad()
def evaluate_noisy_reconstruction(
    model: RAEModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    sigma: float = 0.5,
    n_samples: int = 500,
) -> dict:
    """Evaluate reconstruction accuracy with noisy latents."""
    model.eval()

    node_correct = 0
    edge_correct = 0
    total_nodes = 0
    total_edges = 0
    n_processed = 0

    for batch in dataloader:
        if n_processed >= n_samples:
            break

        node_features = batch['node_features'].to(device)
        adj_matrix = batch['adj_matrix'].to(device)

        # Encode
        z = model.encode(node_features, adj_matrix)

        # Add noise
        from src.models.rae.loss import add_training_noise
        z_noisy = add_training_noise(z, sigma)

        # Decode
        node_types_pred, adj_pred = model.decode(z_noisy, temperature=1.0, hard=True)

        # Compare nodes
        node_targets = node_features.argmax(dim=-1)
        node_correct += (node_types_pred == node_targets).sum().item()
        total_nodes += node_targets.numel()

        # Compare edges (upper triangular only)
        B, N, _ = adj_matrix.shape
        mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
        edge_targets = adj_matrix[:, mask]
        edge_preds = adj_pred[:, mask]
        edge_correct += (edge_preds == edge_targets).sum().item()
        total_edges += edge_targets.numel()

        n_processed += node_features.size(0)

    return {
        'noisy_node_accuracy': node_correct / total_nodes if total_nodes > 0 else 0.0,
        'noisy_edge_accuracy': edge_correct / total_edges if total_edges > 0 else 0.0,
        'sigma': sigma,
    }


def save_checkpoint(
    model: RAEModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    config: dict,
    path: Path,
):
    """Save training checkpoint."""
    # Save only trainable parameters (decoder + encoder adapter projection)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'adapter_proj_state_dict': model.encoder_adapter.proj.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
    }, path)


def load_checkpoint(
    path: Path,
    model: RAEModel,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
) -> int:
    """Load checkpoint and return epoch number."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    # Load full model state if available
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # Load decoder and adapter projection separately
        if 'decoder_state_dict' in checkpoint:
            model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if 'adapter_proj_state_dict' in checkpoint:
            model.encoder_adapter.proj.load_state_dict(checkpoint['adapter_proj_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch']


def main(config_path: str):
    """Main training function."""
    # Load config
    config = load_config(config_path)

    # Setup
    output_dir = setup_output_dir(config)
    set_seed(config.get('seed', 42))
    device = get_device(config)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Config: {config_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Device: {device}")

    # Initialize W&B
    wandb_config = config.get('wandb', {})
    use_wandb = wandb_config.get('enabled', True)
    if use_wandb:
        wandb.init(
            project=wandb_config.get('project', 'molecule-rae'),
            name=wandb_config.get('name', f"{config['data']['dataset']}_rae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            config=config,
            dir=str(output_dir),
        )
        logger.info(f"W&B run: {wandb.run.url}")

    # Data
    logger.info("Loading data...")
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')

    train_dataset = train_loader.dataset
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Dataset stats: {train_dataset.get_statistics()}")

    # Update config with dataset-specific values
    config['data']['max_atoms'] = train_dataset.max_atoms
    config['model']['num_atom_types'] = train_dataset.num_atom_types
    config['model']['num_bond_types'] = train_dataset.num_bond_types

    # Load MAE encoder adapter
    encoder_config = config.get('encoder', {})
    mae_checkpoint = encoder_config.get('checkpoint')
    d_latent = encoder_config.get('d_latent', 64)

    if not mae_checkpoint:
        raise ValueError("encoder.checkpoint must be specified in config")

    logger.info(f"Loading MAE encoder from: {mae_checkpoint}")
    encoder_adapter = load_mae_encoder_adapter(
        mae_checkpoint=mae_checkpoint,
        d_latent=d_latent,
        device=device,
        freeze_encoder=True,
    )
    logger.info(f"MAE encoder parameters: {count_parameters(encoder_adapter.mae_encoder):,} (frozen)")
    logger.info(f"Adapter projection parameters: {count_parameters(encoder_adapter.proj):,} (trainable)")

    # Build RAE decoder
    logger.info("Building RAE decoder...")
    decoder = build_rae_decoder(config, d_latent=d_latent, max_atoms=train_dataset.max_atoms)
    decoder = decoder.to(device)
    logger.info(f"RAE decoder parameters: {count_parameters(decoder):,}")

    # Create RAE model
    model = RAEModel(encoder_adapter, decoder)
    model = model.to(device)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_trainable:,}")

    # Compute edge class weights for handling imbalance
    train_config = config['training']
    loss_config = config.get('loss', {})
    edge_class_weights = None
    if loss_config.get('use_edge_class_weights', True):
        logger.info("Computing edge class weights...")
        edge_class_weights = compute_edge_class_weights(
            train_loader, train_dataset.num_bond_types, device
        )
        logger.info(f"Edge class weights: {edge_class_weights}")

    # Optimizer and scheduler (only for trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=train_config.get('lr', 1e-4),
        weight_decay=train_config.get('weight_decay', 0.0),
    )

    scheduler = None
    if train_config.get('use_scheduler', True):
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_config['epochs'],
            eta_min=train_config.get('min_lr', 1e-6),
        )

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    results_history = []

    for epoch in range(1, train_config['epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, config,
            edge_class_weights=edge_class_weights,
        )
        logger.info(f"Epoch {epoch} - Train: loss={train_metrics['total_loss']:.4f}, "
                    f"node_acc={train_metrics['node_accuracy']:.4f}, "
                    f"edge_acc={train_metrics['edge_accuracy']:.4f}, "
                    f"sigma={train_metrics['noise_sigma']:.3f}")

        # Validate
        val_metrics = validate(model, val_loader, device, config, edge_class_weights=edge_class_weights)
        logger.info(f"Epoch {epoch} - Val: loss={val_metrics['total_loss']:.4f}, "
                    f"node_acc={val_metrics['node_accuracy']:.4f}, "
                    f"edge_acc={val_metrics['edge_accuracy']:.4f}")

        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        # Log to W&B
        if use_wandb:
            wandb_metrics = {'epoch': epoch, 'lr': current_lr}
            wandb_metrics.update(_filter_wandb_metrics(train_metrics, 'train'))
            wandb_metrics.update(_filter_wandb_metrics(val_metrics, 'val'))
            wandb.log(wandb_metrics)

        # Save checkpoint based on validation loss
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                config,
                output_dir / 'checkpoints' / 'best.pt'
            )
            logger.info("Saved best checkpoint (loss)")

        # Save checkpoint based on validation accuracy
        val_acc = (val_metrics['node_accuracy'] + val_metrics['edge_accuracy']) / 2
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                config,
                output_dir / 'checkpoints' / 'best_accuracy.pt'
            )
            logger.info("Saved best checkpoint (accuracy)")

        # Periodic checkpoint
        if epoch % train_config.get('save_every', 20) == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                config,
                output_dir / 'checkpoints' / f'epoch_{epoch}.pt'
            )

        # Periodic noisy reconstruction evaluation
        if epoch % train_config.get('eval_every', 10) == 0:
            logger.info("Evaluating noisy reconstruction...")
            noisy_metrics = evaluate_noisy_reconstruction(
                model, val_loader, device, sigma=0.5, n_samples=500
            )
            logger.info(f"Epoch {epoch} - Noisy reconstruction (sigma=0.5): "
                        f"node_acc={noisy_metrics['noisy_node_accuracy']:.4f}, "
                        f"edge_acc={noisy_metrics['noisy_edge_accuracy']:.4f}")

            if use_wandb:
                wandb.log(_filter_wandb_metrics(noisy_metrics, 'noisy_recon'))

            # Save results
            epoch_results = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
                'noisy_recon': noisy_metrics,
            }
            results_history.append(epoch_results)

            with open(output_dir / 'results' / 'metrics.json', 'w') as f:
                json.dump(results_history, f, indent=2)

    # Final evaluation
    logger.info("Final evaluation...")
    load_checkpoint(output_dir / 'checkpoints' / 'best.pt', model)

    # Clean reconstruction
    final_val_metrics = validate(model, val_loader, device, config, edge_class_weights=edge_class_weights)
    logger.info(f"Final clean reconstruction: {final_val_metrics}")

    # Noisy reconstruction at different sigma levels
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        noisy_metrics = evaluate_noisy_reconstruction(
            model, val_loader, device, sigma=sigma, n_samples=1000
        )
        logger.info(f"Final noisy reconstruction (sigma={sigma}): {noisy_metrics}")

    # Log final metrics
    if use_wandb:
        wandb.log(_filter_wandb_metrics(final_val_metrics, 'final/clean'))
        wandb.finish()

    final_metrics = {
        'val': final_val_metrics,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
    }
    with open(output_dir / 'results' / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("Training complete!")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/rae/qm9_rae_debug.yaml"
    main(config_path)
