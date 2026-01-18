"""Dataset classes for molecule generation."""

import pickle
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9, ZINC
from torch_geometric.data import Data
from rdkit import Chem
from tqdm import tqdm

# Atomic number to symbol mapping
ATOMIC_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}

# Atom type mappings for different datasets (heavy atoms only for QM9)
# Index 0 is reserved for "empty" atom (padding positions)
ATOM_TYPES = {
    'qm9': ['X', 'C', 'N', 'O', 'F'],  # X=empty, then heavy atoms (H added by RDKit)
    'zinc250k': ['X', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P'],
}

# Bond type mappings (0: no bond, 1: single, 2: double, 3: triple, 4: aromatic)
BOND_TYPES = {
    'single': 1,
    'double': 2,
    'triple': 3,
    'aromatic': 4,
}

# Maximum number of heavy atoms for each dataset
MAX_ATOMS = {
    'qm9': 9,  # Heavy atoms only (C, N, O, F)
    'zinc250k': 38,
}


def _extract_edge_types(data) -> np.ndarray:
    """Extract edge types from PyG data object.

    Returns array of bond types (1-indexed: 1=single, 2=double, 3=triple, 4=aromatic).
    """
    edge_index = data.edge_index.numpy()

    if not hasattr(data, 'edge_attr') or data.edge_attr is None:
        return np.ones(edge_index.shape[1], dtype=np.int64)

    edge_attr = data.edge_attr.numpy()
    if len(edge_attr.shape) > 1:
        return edge_attr.argmax(axis=1) + 1
    return edge_attr + 1


class MoleculeDataset(Dataset):
    """Base dataset for molecule generation with adjacency matrix representation."""

    def __init__(
        self,
        root: str,
        dataset_name: str = 'qm9',
        split: str = 'train',
        max_atoms: Optional[int] = None,
        transform=None,
        debug: bool = False,
        debug_samples: int = 128,
    ):
        self.root = Path(root)
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.transform = transform
        self.max_atoms = max_atoms or MAX_ATOMS.get(self.dataset_name, 50)
        self.atom_types = ATOM_TYPES.get(self.dataset_name, ATOM_TYPES['qm9'])
        self.num_atom_types = len(self.atom_types)
        self.num_bond_types = 5  # 0: none, 1: single, 2: double, 3: triple, 4: aromatic
        self.debug = debug
        self.debug_samples = debug_samples

        self.processed_path = self.root / f'{self.dataset_name}_{split}_processed.pkl'

        if self.processed_path.exists():
            self._load_processed()
        else:
            self._process_and_save()

        if self.debug:
            self._apply_debug_limit()

    def _apply_debug_limit(self):
        """Limit dataset size for debugging."""
        if len(self.adj_matrices) <= self.debug_samples:
            return

        self.node_features = self.node_features[:self.debug_samples]
        self.adj_matrices = self.adj_matrices[:self.debug_samples]
        self.num_atoms = self.num_atoms[:self.debug_samples]
        if self.smiles_list:
            self.smiles_list = self.smiles_list[:self.debug_samples]

    def _load_processed(self):
        """Load preprocessed data."""
        with open(self.processed_path, 'rb') as f:
            data = pickle.load(f)
        self.node_features = data['node_features']
        self.adj_matrices = data['adj_matrices']
        self.num_atoms = data['num_atoms']
        self.smiles_list = data.get('smiles', [])

    def _process_and_save(self):
        """Process raw data and save."""
        self.root.mkdir(parents=True, exist_ok=True)

        if self.dataset_name == 'qm9':
            self._process_qm9()
        elif self.dataset_name == 'zinc250k':
            self._process_zinc250k()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        data = {
            'node_features': self.node_features,
            'adj_matrices': self.adj_matrices,
            'num_atoms': self.num_atoms,
            'smiles': self.smiles_list,
        }
        with open(self.processed_path, 'wb') as f:
            pickle.dump(data, f)

    def _process_qm9(self):
        """Process QM9 dataset."""
        dataset = QM9(root=str(self.root / 'raw' / 'qm9'))

        # Split indices
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)

        indices = np.random.RandomState(42).permutation(n_total)
        if self.split == 'train':
            indices = indices[:n_train]
        elif self.split == 'val':
            indices = indices[n_train:n_train + n_val]
        else:  # test
            indices = indices[n_train + n_val:]

        self._process_pyg_dataset(dataset, indices)

    def _process_zinc250k(self):
        """Process ZINC250k dataset."""
        # Use ZINC subset from PyG
        split_map = {'train': 'train', 'val': 'val', 'test': 'test'}
        dataset = ZINC(
            root=str(self.root / 'raw' / 'zinc'),
            subset=True,
            split=split_map[self.split],
        )
        indices = list(range(len(dataset)))
        self._process_pyg_dataset(dataset, indices)

    def _process_pyg_dataset(self, dataset, indices: List[int]):
        """Convert PyG dataset to adjacency matrix format (heavy atoms only for QM9)."""
        self.node_features = []
        self.adj_matrices = []
        self.num_atoms = []
        self.smiles_list = []

        # Skip 'X' (empty) at index 0, real atoms start at index 1
        atom_type_to_idx = {atom: i for i, atom in enumerate(self.atom_types) if atom != 'X'}

        for idx in tqdm(indices, desc=f"Processing {self.dataset_name} {self.split}"):
            data = dataset[idx]

            if hasattr(data, 'z'):
                result = self._process_qm9_molecule(data, atom_type_to_idx)
            else:
                result = self._process_zinc_molecule(data)

            if result is None:
                continue

            node_types, num_nodes, adj = result

            # Create padded node features (one-hot encoded)
            # Index 0 = empty (padding), indices 1+ = real atom types
            node_feat = np.zeros((self.max_atoms, self.num_atom_types), dtype=np.float32)
            node_feat[:, 0] = 1.0  # Default all positions to "empty"
            for i, t in enumerate(node_types):
                if t < self.num_atom_types:
                    node_feat[i, 0] = 0.0  # Clear empty flag
                    node_feat[i, t] = 1.0  # Set actual atom type

            self.node_features.append(node_feat)
            self.adj_matrices.append(adj)
            self.num_atoms.append(num_nodes)

        self.node_features = np.array(self.node_features)
        self.adj_matrices = np.array(self.adj_matrices)
        self.num_atoms = np.array(self.num_atoms)

    def _process_qm9_molecule(
        self, data, atom_type_to_idx: dict
    ) -> Optional[Tuple[List[int], int, np.ndarray]]:
        """Process a single QM9 molecule, filtering to heavy atoms only."""
        z = data.z.numpy()
        edge_index = data.edge_index.numpy()

        # Filter to heavy atoms only (exclude H)
        heavy_indices = np.where(z != 1)[0]
        num_heavy = len(heavy_indices)

        if num_heavy > self.max_atoms or num_heavy == 0:
            return None

        # Map old indices to new indices for heavy atoms
        old_to_new = {old: new for new, old in enumerate(heavy_indices)}

        # Get heavy atom types
        node_types = []
        for atom_idx in heavy_indices:
            symbol = ATOMIC_TO_SYMBOL.get(z[atom_idx])
            if symbol is None or symbol not in atom_type_to_idx:
                return None
            node_types.append(atom_type_to_idx[symbol])

        # Create adjacency matrix with only heavy atom edges
        edge_types = _extract_edge_types(data)
        adj = np.zeros((self.max_atoms, self.max_atoms), dtype=np.int64)

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src in old_to_new and dst in old_to_new:
                adj[old_to_new[src], old_to_new[dst]] = edge_types[i]

        return node_types, num_heavy, adj

    def _process_zinc_molecule(self, data) -> Optional[Tuple[List[int], int, np.ndarray]]:
        """Process a single ZINC molecule (already heavy atoms only)."""
        num_nodes = data.x.size(0)
        if num_nodes > self.max_atoms:
            return None

        node_types = data.x.squeeze().tolist()
        if isinstance(node_types, int):
            node_types = [node_types]

        edge_index = data.edge_index.numpy()
        edge_types = _extract_edge_types(data)

        adj = np.zeros((self.max_atoms, self.max_atoms), dtype=np.int64)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            adj[src, dst] = edge_types[i]

        return node_types, num_nodes, adj

    def __len__(self) -> int:
        return len(self.adj_matrices)

    def __getitem__(self, idx: int) -> dict:
        item = {
            'node_features': torch.tensor(self.node_features[idx], dtype=torch.float32),
            'adj_matrix': torch.tensor(self.adj_matrices[idx], dtype=torch.long),
            'num_atoms': self.num_atoms[idx],
        }
        if self.transform:
            item = self.transform(item)
        return item

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        return {
            'num_samples': len(self),
            'max_atoms': self.max_atoms,
            'num_atom_types': self.num_atom_types,
            'num_bond_types': self.num_bond_types,
            'avg_atoms': float(np.mean(self.num_atoms)),
            'atom_types': self.atom_types,
        }


def get_dataloader(
    config: dict,
    split: str = 'train',
) -> torch.utils.data.DataLoader:
    """Create dataloader from config."""
    data_config = config['data']
    dataset = MoleculeDataset(
        root=data_config['root'],
        dataset_name=data_config['dataset'],
        split=split,
        max_atoms=data_config.get('max_atoms'),
        debug=data_config.get('debug', False),
        debug_samples=data_config.get('debug_samples', 128),
    )

    shuffle = (split == 'train')
    batch_size = config['training']['batch_size'] if split == 'train' else config['training'].get('eval_batch_size', 64)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=(split == 'train'),
    )
