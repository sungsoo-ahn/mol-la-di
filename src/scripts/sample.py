"""Sampling script for trained molecule generation models."""

import sys
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils import load_config, setup_output_dir, set_seed, get_device
from src.data.molecule_dataset import MoleculeDataset, ATOM_TYPES
from src.models.transformer_ar import build_model
from src.evaluation import MoleculeEvaluator, adj_to_mol, mol_to_smiles


def load_model(config: dict, checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_molecules(
    model,
    n_samples: int,
    device: torch.device,
    temperature: float = 1.0,
    batch_size: int = 100,
):
    """Sample molecules from the model."""
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

    return (
        np.concatenate(all_node_types, axis=0),
        np.concatenate(all_adj_matrices, axis=0),
    )


def main(config_path: str):
    """Main sampling function."""
    config = load_config(config_path)

    # Setup
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.get('seed', 42))
    device = get_device(config)

    print(f"Device: {device}")

    # Load training config from checkpoint directory
    train_output_dir = Path(config['checkpoint_dir'])
    train_config_path = train_output_dir / 'config.yaml'
    if train_config_path.exists():
        train_config = load_config(str(train_config_path))
    else:
        train_config = config

    # Ensure model config matches
    train_config['data'] = config.get('data', train_config['data'])
    train_config['model'] = config.get('model', train_config['model'])

    # Load model
    checkpoint_path = config.get('checkpoint_path', train_output_dir / 'checkpoints' / 'best.pt')
    print(f"Loading model from {checkpoint_path}")
    model = load_model(train_config, str(checkpoint_path), device)

    # Get atom decoder
    dataset_name = train_config['data'].get('dataset', 'qm9')
    atom_decoder = ATOM_TYPES.get(dataset_name, ATOM_TYPES['qm9'])

    # Load training data for novelty computation
    training_smiles = []
    if config.get('compute_novelty', True):
        try:
            train_dataset = MoleculeDataset(
                root=train_config['data']['root'],
                dataset_name=dataset_name,
                split='train',
            )
            training_smiles = train_dataset.smiles_list
            print(f"Loaded {len(training_smiles)} training SMILES for novelty computation")
        except Exception as e:
            print(f"Could not load training data: {e}")

    # Sample
    n_samples = config.get('n_samples', 10000)
    temperature = config.get('temperature', 1.0)
    batch_size = config.get('batch_size', 100)

    print(f"Sampling {n_samples} molecules with temperature {temperature}...")
    node_types, adj_matrices = sample_molecules(
        model, n_samples, device, temperature, batch_size
    )

    # Evaluate
    print("Evaluating samples...")
    evaluator = MoleculeEvaluator(
        atom_decoder=atom_decoder,
        training_smiles=training_smiles,
    )
    metrics = evaluator.evaluate(node_types, adj_matrices)

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Save results
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save generated SMILES
    smiles_list = []
    for i in range(len(node_types)):
        mol = adj_to_mol(node_types[i], adj_matrices[i], atom_decoder)
        if mol is not None:
            smiles = mol_to_smiles(mol)
            if smiles is not None:
                smiles_list.append(smiles)

    with open(output_dir / 'generated_smiles.txt', 'w') as f:
        f.write('\n'.join(smiles_list))

    print(f"\nSaved {len(smiles_list)} valid SMILES to {output_dir / 'generated_smiles.txt'}")
    print(f"Metrics saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/sample.yaml"
    main(config_path)
