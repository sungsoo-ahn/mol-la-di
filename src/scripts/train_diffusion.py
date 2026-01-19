"""Training script for latent diffusion model."""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from rdkit.Chem import Draw

from src.utils import load_config, setup_output_dir, set_seed, get_device, count_parameters
from src.data.molecule_dataset import get_dataloader
from src.models.vae import MoleculeVAE
from src.models.diffusion import LatentDiffusionModel
from src.models.diffusion.latent_diffusion import LatentDiffusionWithVAE
from src.evaluation import MoleculeEvaluator, adj_to_mol, mol_to_smiles


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger('train_diffusion')
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


def load_vae(vae_checkpoint: str, config: dict, device: torch.device) -> MoleculeVAE:
    """Load pretrained VAE from checkpoint."""
    checkpoint = torch.load(vae_checkpoint, map_location='cpu', weights_only=False)

    # Get VAE config from checkpoint or use current config
    vae_config = checkpoint.get('config', config)

    model_config = vae_config['model']
    data_config = vae_config['data']

    vae = MoleculeVAE(
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

    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()

    # Freeze VAE
    for param in vae.parameters():
        param.requires_grad = False

    return vae


def build_diffusion(config: dict) -> LatentDiffusionModel:
    """Build diffusion model from config."""
    model_config = config['model']
    data_config = config['data']
    diffusion_config = config.get('diffusion', {})

    return LatentDiffusionModel(
        d_latent=model_config.get('d_latent', 64),
        d_model=diffusion_config.get('d_model', 256),
        nhead=diffusion_config.get('nhead', 8),
        num_layers=diffusion_config.get('num_layers', 6),
        dim_feedforward=diffusion_config.get('dim_feedforward', 1024),
        dropout=diffusion_config.get('dropout', 0.0),
        max_atoms=data_config['max_atoms'],
        num_timesteps=diffusion_config.get('num_timesteps', 1000),
        beta_schedule=diffusion_config.get('beta_schedule', 'cosine'),
    )


@torch.no_grad()
def encode_dataset(
    vae: MoleculeVAE,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Encode entire dataset to latents."""
    vae.eval()
    all_latents = []

    for batch in tqdm(dataloader, desc='Encoding dataset'):
        node_features = batch['node_features'].to(device)
        adj_matrix = batch['adj_matrix'].to(device)

        # Encode to latents (use mean, not sampled)
        z = vae.get_latent(node_features, adj_matrix, sample=False)
        all_latents.append(z.cpu())

    return torch.cat(all_latents, dim=0)


def train_epoch(
    model: LatentDiffusionModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 1.0,
    warmup_epochs: int = 0,
    base_lr: float = 1e-4,
) -> dict:
    """Train for one epoch."""
    model.train()

    # Apply LR warmup
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        warmup_factor = epoch / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * warmup_factor

    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        if isinstance(batch, (list, tuple)):
            z_0 = batch[0].to(device)
        else:
            z_0 = batch.to(device)

        optimizer.zero_grad()
        loss_dict = model.compute_loss(z_0)
        loss = loss_dict['loss']
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return {'loss': total_loss / n_batches}


@torch.no_grad()
def validate(
    model: LatentDiffusionModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            z_0 = batch[0].to(device)
        else:
            z_0 = batch.to(device)

        loss_dict = model.compute_loss(z_0)
        total_loss += loss_dict['loss'].item()
        n_batches += 1

    return {'loss': total_loss / n_batches}


@torch.no_grad()
def sample_and_evaluate(
    combined_model: LatentDiffusionWithVAE,
    evaluator: MoleculeEvaluator,
    n_samples: int,
    device: torch.device,
    num_inference_steps: int = 100,
    temperature: float = 1.0,
    batch_size: int = 100,
) -> tuple:
    """Sample molecules and evaluate them."""
    combined_model.eval()

    all_node_types = []
    all_adj_matrices = []

    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc='Sampling'):
        curr_batch_size = min(batch_size, n_samples - i * batch_size)
        node_types, adj_matrix = combined_model.sample(
            num_samples=curr_batch_size,
            device=device,
            num_inference_steps=num_inference_steps,
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
            project=wandb_config.get('project', 'molecule-diffusion'),
            name=wandb_config.get('name', f"{config['data']['dataset']}_diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            config=config,
            dir=str(output_dir),
        )
        logger.info(f"W&B run: {wandb.run.url}")

    # Load VAE
    vae_checkpoint = config['vae']['checkpoint']
    logger.info(f"Loading VAE from: {vae_checkpoint}")
    vae = load_vae(vae_checkpoint, config, device)
    logger.info(f"VAE parameters: {count_parameters(vae):,} (frozen)")

    # Update config with VAE settings
    config['data']['max_atoms'] = vae.max_atoms
    config['model']['num_atom_types'] = vae.num_atom_types
    config['model']['num_bond_types'] = vae.num_bond_types
    config['model']['d_latent'] = vae.d_latent

    # Data - load and encode to latents
    logger.info("Loading data...")
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')

    train_dataset = train_loader.dataset
    logger.info(f"Train samples: {len(train_dataset)}")

    # Encode dataset to latents
    logger.info("Encoding training data to latents...")
    train_latents = encode_dataset(vae, train_loader, device)
    logger.info(f"Train latents shape: {train_latents.shape}")

    logger.info("Encoding validation data to latents...")
    val_latents = encode_dataset(vae, val_loader, device)
    logger.info(f"Val latents shape: {val_latents.shape}")

    # Create latent dataloaders
    train_config = config['training']
    batch_size = train_config.get('batch_size', 128)

    train_latent_dataset = TensorDataset(train_latents)
    train_latent_loader = DataLoader(
        train_latent_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_latent_dataset = TensorDataset(val_latents)
    val_latent_loader = DataLoader(
        val_latent_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Build diffusion model
    logger.info("Building diffusion model...")
    diffusion = build_diffusion(config)
    diffusion = diffusion.to(device)
    logger.info(f"Diffusion parameters: {count_parameters(diffusion):,}")

    # Combined model for sampling
    combined_model = LatentDiffusionWithVAE(vae, diffusion)

    # Optimizer and scheduler
    optimizer = AdamW(
        diffusion.parameters(),
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

    # Evaluator
    evaluator = MoleculeEvaluator(
        atom_decoder=train_dataset.atom_types,
        training_smiles=train_dataset.smiles_list if hasattr(train_dataset, 'smiles_list') else [],
    )

    # Training loop
    best_val_loss = float('inf')
    results_history = []

    max_grad_norm = train_config.get('max_grad_norm', 1.0)
    warmup_epochs = train_config.get('warmup_epochs', 0)
    base_lr = train_config.get('lr', 1e-4)
    num_inference_steps = config.get('diffusion', {}).get('inference_steps', 100)

    for epoch in range(1, train_config['epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            diffusion, train_latent_loader, optimizer, device, epoch,
            max_grad_norm=max_grad_norm,
            warmup_epochs=warmup_epochs,
            base_lr=base_lr,
        )
        logger.info(f"Epoch {epoch} - Train: {train_metrics}")

        # Validate
        val_metrics = validate(diffusion, val_latent_loader, device)
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
                diffusion, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                config,
                output_dir / 'checkpoints' / 'best.pt'
            )
            logger.info("Saved best checkpoint")

        # Periodic checkpoint
        if epoch % train_config.get('save_every', 50) == 0:
            save_checkpoint(
                diffusion, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                config,
                output_dir / 'checkpoints' / f'epoch_{epoch}.pt'
            )

        # Sample and evaluate
        if epoch % train_config.get('eval_every', 50) == 0:
            logger.info("Sampling and evaluating...")
            sample_metrics, valid_mols = sample_and_evaluate(
                combined_model, evaluator,
                n_samples=config.get('eval', {}).get('n_samples', 1000),
                device=device,
                num_inference_steps=num_inference_steps,
                temperature=config.get('eval', {}).get('temperature', 1.0),
            )
            logger.info(f"Epoch {epoch} - Sample metrics: {sample_metrics}")

            # Log to W&B
            if use_wandb:
                wandb_log = _filter_wandb_metrics(sample_metrics, 'sample')

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
            results_history.append(sample_metrics)

            with open(output_dir / 'results' / 'metrics.json', 'w') as f:
                json.dump(results_history, f, indent=2)

    # Final evaluation
    logger.info("Final evaluation...")
    load_checkpoint(output_dir / 'checkpoints' / 'best.pt', diffusion)
    combined_model = LatentDiffusionWithVAE(vae, diffusion)

    final_metrics, final_valid_mols = sample_and_evaluate(
        combined_model, evaluator,
        n_samples=config.get('eval', {}).get('final_n_samples', 10000),
        device=device,
        num_inference_steps=num_inference_steps,
        temperature=config.get('eval', {}).get('temperature', 1.0),
    )
    logger.info(f"Final metrics: {final_metrics}")

    # Log final metrics and molecules to W&B
    if use_wandb:
        final_wandb_log = _filter_wandb_metrics(final_metrics, 'final')

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

    with open(output_dir / 'results' / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("Training complete!")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/diffusion/qm9_diffusion.yaml"
    main(config_path)
