"""Dataset classes for molecule generation."""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Token to base atom type mapping for GEEL format
# Maps token strings like '[C]', '[CH]', '[C-]' etc. to base element
TOKEN_TO_ATOM = {
    # Carbon variants
    '[C]': 'C', '[CH]': 'C', '[CH2]': 'C', '[CH3]': 'C',
    '[C-]': 'C', '[CH-]': 'C', '[C+]': 'C',
    # Nitrogen variants
    '[N]': 'N', '[NH]': 'N', '[NH2]': 'N', '[NH3]': 'N',
    '[N+]': 'N', '[N-]': 'N', '[NH+]': 'N', '[NH2+]': 'N', '[NH3+]': 'N',
    # Oxygen variants
    '[O]': 'O', '[OH]': 'O', '[O-]': 'O', '[O+]': 'O',
    # Halogens
    '[F]': 'F', '[Cl]': 'Cl', '[Br]': 'Br', '[I]': 'I',
    # Other atoms
    '[S]': 'S', '[SH]': 'S', '[S-]': 'S', '[S+]': 'S',
    '[P]': 'P', '[PH]': 'P',
}

# GEEL edge label to bond type mapping
# Bond types: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic
# GEEL uses: 5=single, 6=double, 7=triple, 8=aromatic
# We use: 1=single, 2=double, 3=triple, 4=aromatic
GEEL_EDGE_TO_BOND = {5: 1, 6: 2, 7: 3, 8: 4}

# Atom type mappings for different datasets (heavy atoms only)
# Index 0 is reserved for "empty" atom (padding positions)
ATOM_TYPES = {
    'qm9': ['X', 'C', 'N', 'O', 'F'],  # X=empty, then heavy atoms
    'zinc250k': ['X', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P'],
}

# Maximum number of heavy atoms for each dataset
MAX_ATOMS = {
    'qm9': 9,
    'zinc250k': 38,
}


class MoleculeDataset(Dataset):
    """Dataset for molecule generation with adjacency matrix representation.

    Loads data from GEEL repository pickle files containing NetworkX graphs.
    """

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

        # Build atom type lookup (skip 'X' at index 0)
        self.atom_to_idx = {atom: i for i, atom in enumerate(self.atom_types) if atom != 'X'}

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
            self._process_geel_dataset('qm9')
        elif self.dataset_name == 'zinc250k':
            self._process_geel_dataset('zinc250k')
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

    def _process_geel_dataset(self, dataset_name: str):
        """Process dataset from GEEL pickle files."""
        # Determine file paths
        if dataset_name == 'qm9':
            graph_path = self.root / 'qm9' / f'qm9_graph_{self.split}.pkl'
            smiles_path = self.root / 'qm9' / f'qm9_smiles_{self.split}.txt'
        else:  # zinc250k
            graph_path = self.root / 'zinc250k' / f'zinc_graph_{self.split}.pkl'
            smiles_path = self.root / 'zinc250k' / f'zinc_smiles_{self.split}.txt'

        if not graph_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {graph_path}. "
                f"Run 'bash scripts/data/download_datasets.sh' to download."
            )

        # Load NetworkX graphs
        with open(graph_path, 'rb') as f:
            graphs = pickle.load(f)

        # Load SMILES if available
        smiles_list = []
        if smiles_path.exists():
            with open(smiles_path, 'r') as f:
                smiles_list = [line.strip() for line in f]

        self._process_networkx_graphs(graphs, smiles_list)

    def _process_networkx_graphs(self, graphs: list, smiles_list: list[str]):
        """Convert NetworkX graphs to adjacency matrix format."""
        self.node_features = []
        self.adj_matrices = []
        self.num_atoms = []
        self.smiles_list = []

        for idx, g in enumerate(tqdm(graphs, desc=f"Processing {self.dataset_name} {self.split}")):
            result = self._process_networkx_molecule(g)
            if result is None:
                continue

            node_types, num_nodes, adj = result

            # Create padded node features (one-hot encoded)
            node_feat = np.zeros((self.max_atoms, self.num_atom_types), dtype=np.float32)
            node_feat[:, 0] = 1.0  # Default all positions to "empty"
            for i, t in enumerate(node_types):
                if t < self.num_atom_types:
                    node_feat[i, 0] = 0.0  # Clear empty flag
                    node_feat[i, t] = 1.0  # Set actual atom type

            self.node_features.append(node_feat)
            self.adj_matrices.append(adj)
            self.num_atoms.append(num_nodes)
            if idx < len(smiles_list):
                self.smiles_list.append(smiles_list[idx])

        self.node_features = np.array(self.node_features)
        self.adj_matrices = np.array(self.adj_matrices)
        self.num_atoms = np.array(self.num_atoms)

    def _process_networkx_molecule(self, g) -> Optional[tuple[list[int], int, np.ndarray]]:
        """Process a single NetworkX graph molecule."""
        num_nodes = g.number_of_nodes()
        if num_nodes > self.max_atoms or num_nodes == 0:
            return None

        # Extract node types from 'token' attribute
        node_types = []
        for node_id in range(num_nodes):
            node_data = g.nodes[node_id]
            token = node_data.get('token', node_data.get('x'))

            # Map token to base atom type
            base_atom = TOKEN_TO_ATOM.get(token)
            if base_atom is None:
                # Try to extract atom from token format [X...] -> X
                if token and token.startswith('[') and token.endswith(']'):
                    # Extract first letter(s) as element symbol
                    inner = token[1:-1]
                    # Handle cases like 'CH', 'NH+', etc.
                    if inner and inner[0].isupper():
                        elem = inner[0]
                        if len(inner) > 1 and inner[1].islower():
                            elem += inner[1]
                        base_atom = elem

            if base_atom is None or base_atom not in self.atom_to_idx:
                return None  # Skip molecules with unknown atom types

            node_types.append(self.atom_to_idx[base_atom])

        # Create adjacency matrix from edges
        adj = np.zeros((self.max_atoms, self.max_atoms), dtype=np.int64)
        for src, dst, edge_data in g.edges(data=True):
            edge_label = edge_data.get('label', edge_data.get('edge_attr', 5))
            bond_type = GEEL_EDGE_TO_BOND.get(edge_label, 1)  # Default to single bond
            adj[src, dst] = bond_type
            adj[dst, src] = bond_type  # Symmetric for undirected graphs

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
