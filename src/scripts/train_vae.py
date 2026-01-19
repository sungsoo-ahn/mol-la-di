"""Training script for molecular VAE."""

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
from src.models.vae import MoleculeVAE
from src.evaluation import MoleculeEvaluator, adj_to_mol, mol_to_smiles


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger('train_vae')
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


def build_vae(config: dict) -> MoleculeVAE:
    """Build VAE model from config."""
    model_config = config['model']
    data_config = config['data']

    return MoleculeVAE(
        num_atom_types=model_config['num_atom_types'],
        num_bond_types=model_config['num_bond_types'],
        d_model=model_config.get('d_model', 256),
        d_latent=model_config.get('d_latent', 64),
        nhead=model_config.get('nhead', 8),
        encoder_layers=model_config.get('encoder_layers', 4),
        decoder_layers=model_config.get('decoder_layers', 4),
        dim_feedforward=model_config.get('dim_feedforward', 1024),
        dropout=model_config.get('dropout', 0.1),
        max_atoms=data_config['max_atoms'],
    )


def get_beta(epoch: int, warmup_epochs: int) -> float:
    """Get KL divergence weight with warmup."""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / warmup_epochs)


def train_epoch(
    model: MoleculeVAE,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    beta: float,
    edge_weight: float = 1.0,
    max_grad_norm: float = 1.0,
    warmup_epochs: int = 0,
    base_lr: float = 1e-3,
) -> dict:
    """Train for one epoch."""
    model.train()

    # Apply LR warmup
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        warmup_factor = epoch / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * warmup_factor

    totals = {'total_loss': 0.0, 'node_loss': 0.0, 'edge_loss': 0.0, 'kl_loss': 0.0}
    n_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        node_features = batch['node_features'].to(device)
        adj_matrix = batch['adj_matrix'].to(device)

        optimizer.zero_grad()
        loss_dict = model.compute_loss(
            node_features, adj_matrix,
            beta=beta, edge_weight=edge_weight,
        )
        loss = loss_dict['total_loss']
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        for key in totals:
            totals[key] += loss_dict[key].item()
        n_batches += 1

        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss'].item():.4f}",
            'node': f"{loss_dict['node_loss'].item():.4f}",
            'edge': f"{loss_dict['edge_loss'].item():.4f}",
            'kl': f"{loss_dict['kl_loss'].item():.4f}",
        })

    return {key: value / n_batches for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: MoleculeVAE,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    beta: float,
    edge_weight: float = 1.0,
) -> dict:
    """Validate the model."""
    model.eval()

    totals = {'total_loss': 0.0, 'node_loss': 0.0, 'edge_loss': 0.0, 'kl_loss': 0.0}
    n_batches = 0

    for batch in dataloader:
        node_features = batch['node_features'].to(device)
        adj_matrix = batch['adj_matrix'].to(device)

        loss_dict = model.compute_loss(
            node_features, adj_matrix,
            beta=beta, edge_weight=edge_weight,
        )

        for key in totals:
            totals[key] += loss_dict[key].item()
        n_batches += 1

    return {key: value / n_batches for key, value in totals.items()}


@torch.no_grad()
def evaluate_reconstruction(
    model: MoleculeVAE,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 500,
) -> dict:
    """Evaluate reconstruction accuracy."""
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

        # Reconstruct
        recon_nodes, recon_adj = model.reconstruct(node_features, adj_matrix)

        # Compare
        node_targets = node_features.argmax(dim=-1)
        node_correct += (recon_nodes == node_targets).sum().item()
        edge_correct += (recon_adj == adj_matrix).sum().item()
        total_nodes += node_targets.numel()
        total_edges += adj_matrix.numel()

        n_processed += node_features.size(0)

    return {
        'node_accuracy': node_correct / total_nodes if total_nodes > 0 else 0.0,
        'edge_accuracy': edge_correct / total_edges if total_edges > 0 else 0.0,
    }


@torch.no_grad()
def sample_and_evaluate(
    model: MoleculeVAE,
    evaluator: MoleculeEvaluator,
    n_samples: int,
    device: torch.device,
    temperature: float = 1.0,
    batch_size: int = 100,
) -> tuple:
    """Sample molecules from prior and evaluate them."""
    model.eval()

    all_node_types = []
    all_adj_matrices = []

    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc='Sampling'):
        curr_batch_size = min(batch_size, n_samples - i * batch_size)
        node_types, adj_matrix = model.sample(
            num_samples=curr_batch_size,
            device=device,
            temperature=temperature,
        )
        all_node_types.append(node_types.cpu().numpy())
        all_adj_matrices.append(adj_matrix.cpu().numpy())

    all_node_types = np.concatenate(all_node_types, axis=0)
    all_adj_matrices = np.concatenate(all_adj_matrices, axis=0)

    metrics = evaluator.evaluate(all_node_types, all_adj_matrices)

    # Collect valid molecules for visualization
    valid_mols = []
    atom_decoder = evaluator.atom_decoder
    for i in range(len(all_node_types)):
        mol = adj_to_mol(all_node_types[i], all_adj_matrices[i], atom_decoder)
        if mol is not None:
            smiles = mol_to_smiles(mol)
            if smiles is not None:
                valid_mols.append(mol)

    return metrics, valid_mols


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
            project=wandb_config.get('project', 'molecule-vae'),
            name=wandb_config.get('name', f"{config['data']['dataset']}_vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
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

    model = build_vae(config)
    model = model.to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Optimizer and scheduler
    train_config = config['training']
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

    # Evaluator
    evaluator = MoleculeEvaluator(
        atom_decoder=train_dataset.atom_types,
        training_smiles=train_dataset.smiles_list if hasattr(train_dataset, 'smiles_list') else [],
    )

    # Training loop
    best_val_loss = float('inf')
    results_history = []

    # Get training hyperparameters
    edge_weight = train_config.get('edge_weight', 1.0)
    max_grad_norm = train_config.get('max_grad_norm', 1.0)
    warmup_epochs = train_config.get('warmup_epochs', 0)
    beta_warmup_epochs = train_config.get('beta_warmup_epochs', 50)
    base_lr = train_config.get('lr', 3e-4)

    for epoch in range(1, train_config['epochs'] + 1):
        # Compute beta with warmup
        beta = get_beta(epoch, beta_warmup_epochs)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            beta=beta, edge_weight=edge_weight,
            max_grad_norm=max_grad_norm,
            warmup_epochs=warmup_epochs,
            base_lr=base_lr,
        )
        logger.info(f"Epoch {epoch} - Train: {train_metrics}")

        # Validate
        val_metrics = validate(model, val_loader, device, beta=beta, edge_weight=edge_weight)
        logger.info(f"Epoch {epoch} - Val: {val_metrics}")

        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"LR: {current_lr:.2e}, Beta: {beta:.3f}")

        # Log to W&B
        if use_wandb:
            wandb_metrics = {'epoch': epoch, 'lr': current_lr, 'beta': beta}
            wandb_metrics.update(_filter_wandb_metrics(train_metrics, 'train'))
            wandb_metrics.update(_filter_wandb_metrics(val_metrics, 'val'))
            wandb.log(wandb_metrics)

        # Save checkpoint
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                config,
                output_dir / 'checkpoints' / 'best.pt'
            )
            logger.info("Saved best checkpoint")

        # Periodic checkpoint
        if epoch % train_config.get('save_every', 20) == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                config,
                output_dir / 'checkpoints' / f'epoch_{epoch}.pt'
            )

        # Sample and evaluate
        if epoch % train_config.get('eval_every', 20) == 0:
            logger.info("Evaluating reconstruction...")
            recon_metrics = evaluate_reconstruction(model, val_loader, device)
            logger.info(f"Epoch {epoch} - Reconstruction: {recon_metrics}")

            logger.info("Sampling from prior...")
            sample_metrics, valid_mols = sample_and_evaluate(
                model, evaluator,
                n_samples=config.get('eval', {}).get('n_samples', 1000),
                device=device,
                temperature=config.get('eval', {}).get('temperature', 1.0),
            )
            logger.info(f"Epoch {epoch} - Sample metrics: {sample_metrics}")

            # Log to W&B
            if use_wandb:
                wandb_log = _filter_wandb_metrics(recon_metrics, 'recon')
                wandb_log.update(_filter_wandb_metrics(sample_metrics, 'sample'))

                # Create and log molecule visualization
                if valid_mols:
                    mol_img = create_molecule_grid_image(valid_mols, n_mols=20, n_cols=5)
                    if mol_img is not None:
                        wandb_log['generated_molecules'] = wandb.Image(
                            mol_img,
                            caption=f"Epoch {epoch}: {len(valid_mols)} valid molecules (showing 20)"
                        )

                wandb.log(wandb_log)

            # Save sample metrics
            sample_metrics['epoch'] = epoch
            sample_metrics.update(recon_metrics)
            results_history.append(sample_metrics)

            with open(output_dir / 'results' / 'metrics.json', 'w') as f:
                json.dump(results_history, f, indent=2)

    # Final evaluation
    logger.info("Final evaluation...")
    load_checkpoint(output_dir / 'checkpoints' / 'best.pt', model)

    final_recon_metrics = evaluate_reconstruction(model, val_loader, device, n_samples=1000)
    logger.info(f"Final reconstruction: {final_recon_metrics}")

    final_metrics, final_valid_mols = sample_and_evaluate(
        model, evaluator,
        n_samples=config.get('eval', {}).get('final_n_samples', 10000),
        device=device,
        temperature=config.get('eval', {}).get('temperature', 1.0),
    )
    logger.info(f"Final metrics: {final_metrics}")

    # Log final metrics and molecules to W&B
    if use_wandb:
        final_wandb_log = _filter_wandb_metrics(final_recon_metrics, 'final/recon')
        final_wandb_log.update(_filter_wandb_metrics(final_metrics, 'final'))

        # Log final molecule visualization
        if final_valid_mols:
            final_mol_img = create_molecule_grid_image(final_valid_mols, n_mols=50, n_cols=10)
            if final_mol_img is not None:
                final_wandb_log['final_generated_molecules'] = wandb.Image(
                    final_mol_img,
                    caption=f"Final: {len(final_valid_mols)} valid molecules (showing 50)"
                )

        wandb.log(final_wandb_log)
        wandb.finish()

    final_metrics.update(final_recon_metrics)
    with open(output_dir / 'results' / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("Training complete!")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/vae/qm9_vae.yaml"
    main(config_path)
