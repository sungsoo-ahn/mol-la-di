"""Sampling script for latent diffusion molecule generation."""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm
from rdkit.Chem import Draw

from src.utils import load_config, setup_output_dir, set_seed, get_device
from src.data.molecule_dataset import get_dataloader
from src.models.vae import MoleculeVAE
from src.models.diffusion import LatentDiffusionModel
from src.models.diffusion.latent_diffusion import LatentDiffusionWithVAE
from src.evaluation import MoleculeEvaluator, adj_to_mol, mol_to_smiles


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger('sample_diffusion')
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(output_dir / 'logs' / 'sample.log')
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


def load_vae(vae_checkpoint: str, device: torch.device) -> MoleculeVAE:
    """Load pretrained VAE from checkpoint."""
    checkpoint = torch.load(vae_checkpoint, map_location='cpu', weights_only=False)

    config = checkpoint['config']
    model_config = config['model']
    data_config = config['data']

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

    return vae


def load_diffusion(diffusion_checkpoint: str, config: dict, device: torch.device) -> LatentDiffusionModel:
    """Load pretrained diffusion model from checkpoint."""
    checkpoint = torch.load(diffusion_checkpoint, map_location='cpu', weights_only=False)

    # Use config from checkpoint if available
    ckpt_config = checkpoint.get('config', config)

    model_config = ckpt_config['model']
    data_config = ckpt_config['data']
    diffusion_config = ckpt_config.get('diffusion', {})

    diffusion = LatentDiffusionModel(
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

    diffusion.load_state_dict(checkpoint['model_state_dict'])
    diffusion = diffusion.to(device)
    diffusion.eval()

    return diffusion


@torch.no_grad()
def sample_molecules(
    combined_model: LatentDiffusionWithVAE,
    n_samples: int,
    device: torch.device,
    num_inference_steps: int = 100,
    temperature: float = 1.0,
    batch_size: int = 100,
) -> tuple:
    """Sample molecules from the combined model."""
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

    return all_node_types, all_adj_matrices


def create_molecule_grid_image(mols, n_mols=50, n_cols=10, mol_size=(200, 200)):
    """Create a grid image of molecules."""
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


def main(config_path: str):
    """Main sampling function."""
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

    # Load models
    vae_checkpoint = config['vae']['checkpoint']
    diffusion_checkpoint = config['diffusion']['checkpoint']

    logger.info(f"Loading VAE from: {vae_checkpoint}")
    vae = load_vae(vae_checkpoint, device)

    logger.info(f"Loading diffusion model from: {diffusion_checkpoint}")
    diffusion = load_diffusion(diffusion_checkpoint, config, device)

    # Combined model
    combined_model = LatentDiffusionWithVAE(vae, diffusion)

    # Load training data for evaluation
    logger.info("Loading training data for evaluation...")
    train_loader = get_dataloader(config, split='train')
    train_dataset = train_loader.dataset

    # Evaluator
    evaluator = MoleculeEvaluator(
        atom_decoder=train_dataset.atom_types,
        training_smiles=train_dataset.smiles_list if hasattr(train_dataset, 'smiles_list') else [],
    )

    # Sampling parameters
    sampling_config = config.get('sampling', {})
    n_samples = sampling_config.get('n_samples', 10000)
    num_inference_steps = sampling_config.get('num_inference_steps', 100)
    temperature = sampling_config.get('temperature', 1.0)
    batch_size = sampling_config.get('batch_size', 100)

    logger.info(f"Sampling {n_samples} molecules...")
    logger.info(f"Inference steps: {num_inference_steps}, Temperature: {temperature}")

    # Sample molecules
    all_node_types, all_adj_matrices = sample_molecules(
        combined_model,
        n_samples=n_samples,
        device=device,
        num_inference_steps=num_inference_steps,
        temperature=temperature,
        batch_size=batch_size,
    )

    # Evaluate
    logger.info("Evaluating generated molecules...")
    metrics = evaluator.evaluate(all_node_types, all_adj_matrices)
    logger.info(f"Metrics: {metrics}")

    # Collect valid molecules for visualization
    valid_mols = []
    valid_smiles = []
    atom_decoder = evaluator.atom_decoder
    for i in range(len(all_node_types)):
        mol = adj_to_mol(all_node_types[i], all_adj_matrices[i], atom_decoder)
        if mol is not None:
            smiles = mol_to_smiles(mol)
            if smiles is not None:
                valid_mols.append(mol)
                valid_smiles.append(smiles)

    logger.info(f"Valid molecules: {len(valid_mols)} / {n_samples}")

    # Save results
    with open(output_dir / 'results' / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / 'results' / 'smiles.txt', 'w') as f:
        for smiles in valid_smiles:
            f.write(smiles + '\n')

    # Save molecule visualization
    if valid_mols:
        mol_img = create_molecule_grid_image(valid_mols, n_mols=100, n_cols=10)
        if mol_img is not None:
            mol_img.save(output_dir / 'figures' / 'generated_molecules.png')
            logger.info(f"Saved molecule visualization to {output_dir / 'figures' / 'generated_molecules.png'}")

    # Save samples as numpy
    np.savez(
        output_dir / 'results' / 'samples.npz',
        node_types=all_node_types,
        adj_matrices=all_adj_matrices,
    )

    logger.info("Sampling complete!")

    # Print summary
    print("\n" + "=" * 50)
    print("SAMPLING RESULTS")
    print("=" * 50)
    print(f"Total samples: {n_samples}")
    print(f"Valid molecules: {len(valid_mols)} ({100*len(valid_mols)/n_samples:.1f}%)")
    print(f"Unique molecules: {len(set(valid_smiles))} ({100*len(set(valid_smiles))/len(valid_smiles):.1f}% of valid)" if valid_smiles else "N/A")
    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/diffusion/qm9_diffusion.yaml"
    main(config_path)
