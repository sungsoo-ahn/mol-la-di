"""Training script for molecular MAE."""

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
from rdkit.Chem import Draw

from src.utils import load_config, setup_output_dir, set_seed, get_device, count_parameters
from src.data.molecule_dataset import get_dataloader
from src.models.mae import MoleculeMAE, compute_edge_class_weights, extract_upper_triangular, reconstruct_adj_from_edges
from src.evaluation import MoleculeEvaluator, adj_to_mol, mol_to_smiles


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger('train_mae')
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


def build_mae(config: dict) -> MoleculeMAE:
    """Build MAE model from config."""
    model_config = config['model']
    data_config = config['data']
    masking_config = config.get('masking', {})

    return MoleculeMAE(
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


def train_epoch(
    model: MoleculeMAE,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 1.0,
    warmup_epochs: int = 0,
    base_lr: float = 1e-3,
    edge_class_weights: torch.Tensor = None,
) -> dict:
    """Train for one epoch."""
    model.train()

    # Apply LR warmup
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
            node_features, adj_matrix,
            edge_class_weights=edge_class_weights,
        )
        loss = loss_dict['total_loss']
        loss.backward()

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
        })

    return {key: value / n_batches for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: MoleculeMAE,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    edge_class_weights: torch.Tensor = None,
) -> dict:
    """Validate the model."""
    model.eval()

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

        loss_dict = model.compute_loss(
            node_features, adj_matrix,
            edge_class_weights=edge_class_weights,
        )

        for key in totals:
            if key in loss_dict:
                val = loss_dict[key]
                totals[key] += val.item() if isinstance(val, torch.Tensor) else val
        n_batches += 1

    return {key: value / n_batches for key, value in totals.items()}


@torch.no_grad()
def evaluate_reconstruction(
    model: MoleculeMAE,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 500,
) -> dict:
    """Evaluate full reconstruction accuracy (all positions)."""
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

        # Predict all positions
        node_logits, edge_logits = model.predict_all(node_features, adj_matrix)

        # Compare nodes
        node_targets = node_features.argmax(dim=-1)
        node_preds = node_logits.argmax(dim=-1)
        node_correct += (node_preds == node_targets).sum().item()
        total_nodes += node_targets.numel()

        # Compare edges
        edge_targets = extract_upper_triangular(adj_matrix)
        edge_preds = edge_logits.argmax(dim=-1)
        edge_correct += (edge_preds == edge_targets).sum().item()
        total_edges += edge_targets.numel()

        n_processed += node_features.size(0)

    return {
        'node_accuracy': node_correct / total_nodes if total_nodes > 0 else 0.0,
        'edge_accuracy': edge_correct / total_edges if total_edges > 0 else 0.0,
    }


def create_molecule_grid_image(mols, n_mols=20, n_cols=5, mol_size=(250, 250)):
    """Create a grid image of molecules for W&B logging."""
    mols_to_draw = mols[:n_mols]
    if not mols_to_draw:
        return None

    img = Draw.MolsToGridImage(
        mols_to_draw,
        molsPerRow=n_cols,
        subImgSize=mol_size,
        legends=[f"Mol {i+1}" for i in range(len(mols_to_draw))],
    )
    return img


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    config: dict,
    path: Path,
):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
    }, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
) -> int:
    """Load checkpoint and return epoch number."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
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
            project=wandb_config.get('project', 'molecule-mae'),
            name=wandb_config.get('name', f"{config['data']['dataset']}_mae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
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

    model = build_mae(config)
    model = model.to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Compute edge class weights for handling imbalance
    train_config = config['training']
    edge_class_weights = None
    if train_config.get('use_edge_class_weights', True):
        logger.info("Computing edge class weights...")
        edge_class_weights = compute_edge_class_weights(
            train_loader, train_dataset.num_bond_types, device
        )
        logger.info(f"Edge class weights: {edge_class_weights}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.get('lr', 3e-4),
        weight_decay=train_config.get('weight_decay', 0.01),
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

    # Get training hyperparameters
    max_grad_norm = train_config.get('max_grad_norm', 1.0)
    warmup_epochs = train_config.get('warmup_epochs', 5)
    base_lr = train_config.get('lr', 3e-4)

    for epoch in range(1, train_config['epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            max_grad_norm=max_grad_norm,
            warmup_epochs=warmup_epochs,
            base_lr=base_lr,
            edge_class_weights=edge_class_weights,
        )
        logger.info(f"Epoch {epoch} - Train: loss={train_metrics['total_loss']:.4f}, "
                    f"node_acc={train_metrics['node_accuracy']:.4f}, "
                    f"edge_acc={train_metrics['edge_accuracy']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, device, edge_class_weights=edge_class_weights)
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

        # Periodic full reconstruction evaluation
        if epoch % train_config.get('eval_every', 10) == 0:
            logger.info("Evaluating full reconstruction...")
            recon_metrics = evaluate_reconstruction(model, val_loader, device)
            logger.info(f"Epoch {epoch} - Full reconstruction: "
                        f"node_acc={recon_metrics['node_accuracy']:.4f}, "
                        f"edge_acc={recon_metrics['edge_accuracy']:.4f}")

            if use_wandb:
                wandb.log(_filter_wandb_metrics(recon_metrics, 'recon'))

            # Save results
            epoch_results = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
                'recon': recon_metrics,
            }
            results_history.append(epoch_results)

            with open(output_dir / 'results' / 'metrics.json', 'w') as f:
                json.dump(results_history, f, indent=2)

    # Final evaluation
    logger.info("Final evaluation...")
    load_checkpoint(output_dir / 'checkpoints' / 'best.pt', model)

    final_recon_metrics = evaluate_reconstruction(model, val_loader, device, n_samples=1000)
    logger.info(f"Final reconstruction: {final_recon_metrics}")

    # Log final metrics
    if use_wandb:
        wandb.log(_filter_wandb_metrics(final_recon_metrics, 'final/recon'))
        wandb.finish()

    final_recon_metrics['best_val_loss'] = best_val_loss
    final_recon_metrics['best_val_acc'] = best_val_acc
    with open(output_dir / 'results' / 'final_metrics.json', 'w') as f:
        json.dump(final_recon_metrics, f, indent=2)

    logger.info("Training complete!")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/mae/qm9_mae.yaml"
    main(config_path)
