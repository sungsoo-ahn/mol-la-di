"""Training script for autoregressive molecule generation model."""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from src.utils import load_config, setup_output_dir, set_seed, get_device, count_parameters
from src.data.molecule_dataset import MoleculeDataset, get_dataloader, ATOM_TYPES
from src.models.transformer_ar import build_model
from src.evaluation import MoleculeEvaluator


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger('train')
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
    return result


def _compute_batch_loss(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
) -> tuple:
    """Compute loss for a single batch. Returns (loss, metrics)."""
    node_features = batch['node_features'].to(device)
    adj_matrix = batch['adj_matrix'].to(device)
    num_atoms = batch['num_atoms'].to(device)

    node_types = node_features.argmax(dim=-1)
    return model.compute_loss(node_types, adj_matrix, num_atoms)


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 1.0,
    warmup_epochs: int = 0,
    base_lr: float = 1e-3,
) -> dict:
    """Train for one epoch."""
    model.train()

    # Apply warmup
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        warmup_factor = epoch / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * warmup_factor

    totals = {'loss': 0.0, 'node_loss': 0.0, 'edge_loss': 0.0}
    n_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        optimizer.zero_grad()
        loss, metrics = _compute_batch_loss(model, batch, device)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        for key in totals:
            totals[key] += metrics[key]
        n_batches += 1

        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'node': f"{metrics['node_loss']:.4f}",
            'edge': f"{metrics['edge_loss']:.4f}",
        })

    return {key: value / n_batches for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()

    totals = {'loss': 0.0, 'node_loss': 0.0, 'edge_loss': 0.0}
    n_batches = 0

    for batch in dataloader:
        _, metrics = _compute_batch_loss(model, batch, device)
        for key in totals:
            totals[key] += metrics[key]
        n_batches += 1

    return {key: value / n_batches for key, value in totals.items()}


@torch.no_grad()
def sample_and_evaluate(
    model: torch.nn.Module,
    evaluator: MoleculeEvaluator,
    n_samples: int,
    device: torch.device,
    temperature: float = 1.0,
    batch_size: int = 100,
) -> dict:
    """Sample molecules and evaluate them."""
    model.eval()

    all_node_types = []
    all_adj_matrices = []

    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc='Sampling'):
        curr_batch_size = min(batch_size, n_samples - i * batch_size)
        node_types, adj_matrix = model.sample(
            batch_size=curr_batch_size,
            temperature=temperature,
            device=device,
        )
        all_node_types.append(node_types.cpu().numpy())
        all_adj_matrices.append(adj_matrix.cpu().numpy())

    all_node_types = np.concatenate(all_node_types, axis=0)
    all_adj_matrices = np.concatenate(all_adj_matrices, axis=0)

    return evaluator.evaluate(all_node_types, all_adj_matrices)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    path: Path,
):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler = None,
) -> int:
    """Load checkpoint and return epoch number."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
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
            project=wandb_config.get('project', 'molecule-generation'),
            name=wandb_config.get('name', f"{config['data']['dataset']}_ar_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
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

    # Model
    logger.info("Building model...")
    # Update config with dataset-specific values
    config['data']['max_atoms'] = train_dataset.max_atoms
    config['model']['num_atom_types'] = train_dataset.num_atom_types
    config['model']['num_bond_types'] = train_dataset.num_bond_types

    model = build_model(config)
    model = model.to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Optimizer and scheduler
    train_config = config['training']
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.get('lr', 1e-4),
        weight_decay=train_config.get('weight_decay', 0.01),
    )

    scheduler = None
    if train_config.get('use_scheduler', True):
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_config['epochs'],
            eta_min=train_config.get('min_lr', 1e-6),
        )

    # Evaluator
    evaluator = MoleculeEvaluator(
        atom_decoder=train_dataset.atom_types,
        training_smiles=train_dataset.smiles_list if hasattr(train_dataset, 'smiles_list') else [],
    )

    # Training loop
    best_val_loss = float('inf')
    results_history = []

    # Get training hyperparameters
    max_grad_norm = train_config.get('max_grad_norm', 1.0)
    warmup_epochs = train_config.get('warmup_epochs', 0)
    base_lr = train_config.get('lr', 1e-4)

    for epoch in range(1, train_config['epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            max_grad_norm=max_grad_norm,
            warmup_epochs=warmup_epochs,
            base_lr=base_lr,
        )
        logger.info(f"Epoch {epoch} - Train: {train_metrics}")

        # Validate
        val_metrics = validate(model, val_loader, device)
        logger.info(f"Epoch {epoch} - Val: {val_metrics}")

        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"LR: {current_lr:.2e}")

        # Log to W&B
        if use_wandb:
            wandb_metrics = {'epoch': epoch, 'lr': current_lr}
            wandb_metrics.update(_filter_wandb_metrics(train_metrics, 'train'))
            wandb_metrics.update(_filter_wandb_metrics(val_metrics, 'val'))
            wandb.log(wandb_metrics)

        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                output_dir / 'checkpoints' / 'best.pt'
            )
            logger.info("Saved best checkpoint")

        # Periodic checkpoint
        if epoch % train_config.get('save_every', 10) == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                output_dir / 'checkpoints' / f'epoch_{epoch}.pt'
            )

        # Sample and evaluate
        if epoch % train_config.get('eval_every', 10) == 0:
            logger.info("Sampling and evaluating...")
            sample_metrics = sample_and_evaluate(
                model, evaluator,
                n_samples=config.get('eval', {}).get('n_samples', 1000),
                device=device,
                temperature=config.get('eval', {}).get('temperature', 1.0),
            )
            logger.info(f"Epoch {epoch} - Sample metrics: {sample_metrics}")

            # Log sample metrics to W&B
            if use_wandb:
                wandb.log(_filter_wandb_metrics(sample_metrics, 'sample'))

            # Save sample metrics
            sample_metrics['epoch'] = epoch
            results_history.append(sample_metrics)

            with open(output_dir / 'results' / 'metrics.json', 'w') as f:
                json.dump(results_history, f, indent=2)

    # Final evaluation
    logger.info("Final evaluation...")
    load_checkpoint(output_dir / 'checkpoints' / 'best.pt', model)
    final_metrics = sample_and_evaluate(
        model, evaluator,
        n_samples=config.get('eval', {}).get('final_n_samples', 10000),
        device=device,
        temperature=config.get('eval', {}).get('temperature', 1.0),
    )
    logger.info(f"Final metrics: {final_metrics}")

    # Log final metrics to W&B
    if use_wandb:
        wandb.log(_filter_wandb_metrics(final_metrics, 'final'))
        wandb.finish()

    with open(output_dir / 'results' / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("Training complete!")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/training/qm9_ar.yaml"
    main(config_path)
