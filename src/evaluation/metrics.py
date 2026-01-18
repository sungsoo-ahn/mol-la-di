"""Molecule generation evaluation metrics.

Adapted from GruM (https://github.com/harryjo97/GruM) with modifications.

Metrics included:
- Validity: fraction of chemically valid molecules
- Uniqueness: fraction of unique molecules
- Novelty: fraction of molecules not in training set
- FCD: Fréchet ChemNet Distance
- NSPDK MMD: Maximum Mean Discrepancy using NSPDK kernel
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from collections import Counter
import warnings

import torch
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from scipy import linalg

# Bond type mapping (module-level constant)
BOND_TYPE_MAP = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}


def _count_valid_atoms(node_types: np.ndarray, atom_decoder: List[str]) -> int:
    """Count the number of valid (non-empty) atoms.

    Index 0 is the "empty" atom type and is skipped.
    For one-hot encoded arrays, checks if argmax > 0.
    For index encoded arrays, checks if index > 0 and within valid range.
    """
    n_atoms = 0
    for i, atom_type in enumerate(node_types):
        atom_idx = _get_atom_index(atom_type)
        # Index 0 = empty, skip it
        if atom_idx > 0 and atom_idx < len(atom_decoder):
            n_atoms = i + 1

    return n_atoms


def _get_atom_index(atom_type) -> int:
    """Extract atom index from either one-hot or index encoding."""
    if isinstance(atom_type, np.ndarray):
        return int(atom_type.argmax())
    return int(atom_type)


def adj_to_mol(
    node_types: np.ndarray,
    adj_matrix: np.ndarray,
    atom_decoder: List[str],
    sanitize: bool = True,
    add_hydrogens: bool = True,
) -> Optional[Chem.Mol]:
    """Convert adjacency matrix representation to RDKit molecule.

    Args:
        node_types: (n_atoms,) array of atom type indices (0 = empty)
        adj_matrix: (n_atoms, n_atoms) adjacency matrix with bond types
        atom_decoder: list mapping indices to atom symbols (index 0 = 'X' = empty)
        sanitize: whether to sanitize the molecule
        add_hydrogens: whether to add implicit hydrogens (for heavy-atom-only models)

    Returns:
        RDKit Mol object or None if invalid
    """
    # Find valid (non-empty) atom positions and create index mapping
    valid_positions = []
    for i in range(len(node_types)):
        atom_idx = _get_atom_index(node_types[i])
        if atom_idx > 0 and atom_idx < len(atom_decoder):  # Skip empty (index 0)
            valid_positions.append(i)

    if len(valid_positions) == 0:
        return None

    # Map original positions to new molecule indices
    pos_to_mol_idx = {pos: idx for idx, pos in enumerate(valid_positions)}

    mol = Chem.RWMol()

    # Add atoms (skip empty atoms)
    for pos in valid_positions:
        atom_idx = _get_atom_index(node_types[pos])
        atom_symbol = atom_decoder[atom_idx]
        mol.AddAtom(Chem.Atom(atom_symbol))

    # Add bonds (only between valid atoms)
    for i, pos_i in enumerate(valid_positions):
        for j, pos_j in enumerate(valid_positions):
            if pos_i < pos_j:  # Upper triangular
                bond_type = int(adj_matrix[pos_i, pos_j])
                if bond_type > 0 and bond_type in BOND_TYPE_MAP:
                    mol.AddBond(i, j, BOND_TYPE_MAP[bond_type])

    try:
        mol = mol.GetMol()
        if sanitize:
            Chem.SanitizeMol(mol)
        if add_hydrogens:
            mol = Chem.AddHs(mol)
        return mol
    except Exception:
        return None


def mol_to_smiles(mol: Chem.Mol) -> Optional[str]:
    """Convert molecule to canonical SMILES."""
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def compute_validity(
    node_types: np.ndarray,
    adj_matrices: np.ndarray,
    atom_decoder: List[str],
) -> Tuple[float, List[Chem.Mol], List[str]]:
    """Compute validity and return valid molecules.

    Args:
        node_types: (n_samples, max_atoms) or (n_samples, max_atoms, n_atom_types)
        adj_matrices: (n_samples, max_atoms, max_atoms)
        atom_decoder: list mapping indices to atom symbols

    Returns:
        validity: fraction of valid molecules
        valid_mols: list of valid RDKit Mol objects
        valid_smiles: list of valid SMILES strings
    """
    valid_mols = []
    valid_smiles = []

    n_samples = len(node_types)
    for i in range(n_samples):
        mol = adj_to_mol(node_types[i], adj_matrices[i], atom_decoder)
        if mol is not None:
            smiles = mol_to_smiles(mol)
            if smiles is not None:
                valid_mols.append(mol)
                valid_smiles.append(smiles)

    validity = len(valid_mols) / n_samples if n_samples > 0 else 0.0
    return validity, valid_mols, valid_smiles


def compute_uniqueness(smiles_list: List[str]) -> float:
    """Compute uniqueness (fraction of unique molecules)."""
    if len(smiles_list) == 0:
        return 0.0
    return len(set(smiles_list)) / len(smiles_list)


def compute_novelty(
    generated_smiles: List[str],
    training_smiles: List[str],
) -> float:
    """Compute novelty (fraction not in training set)."""
    if len(generated_smiles) == 0:
        return 0.0
    training_set = set(training_smiles)
    novel = sum(1 for s in generated_smiles if s not in training_set)
    return novel / len(generated_smiles)


def compute_fcd(
    generated_mols: List[Chem.Mol],
    reference_mols: List[Chem.Mol],
) -> float:
    """Compute Fréchet ChemNet Distance.

    Uses fcd-torch if available, otherwise falls back to fingerprint-based approximation.
    """
    try:
        from fcd_torch import FCD
        fcd_calculator = FCD(device='cuda' if torch.cuda.is_available() else 'cpu')

        gen_smiles = [Chem.MolToSmiles(m) for m in generated_mols]
        ref_smiles = [Chem.MolToSmiles(m) for m in reference_mols]

        return fcd_calculator(gen_smiles, ref_smiles)
    except ImportError:
        warnings.warn("fcd-torch not installed, using fingerprint-based approximation")
        return _compute_fcd_fingerprint(generated_mols, reference_mols)


def _compute_fcd_fingerprint(
    generated_mols: List[Chem.Mol],
    reference_mols: List[Chem.Mol],
) -> float:
    """Fallback FCD computation using Morgan fingerprints."""
    def get_fingerprints(mols):
        fps = []
        for mol in mols:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros(2048)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        return np.array(fps)

    gen_fps = get_fingerprints(generated_mols)
    ref_fps = get_fingerprints(reference_mols)

    # Compute Fréchet distance in fingerprint space
    mu_gen, sigma_gen = gen_fps.mean(axis=0), np.cov(gen_fps.T)
    mu_ref, sigma_ref = ref_fps.mean(axis=0), np.cov(ref_fps.T)

    return _calculate_frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)


def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Fréchet distance between two Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def compute_snn(
    generated_mols: List[Chem.Mol],
    reference_mols: List[Chem.Mol],
) -> float:
    """Compute average similarity to nearest neighbor in reference set."""
    if len(generated_mols) == 0 or len(reference_mols) == 0:
        return 0.0

    gen_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in generated_mols]
    ref_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in reference_mols]

    similarities = []
    for gen_fp in gen_fps:
        max_sim = max(DataStructs.TanimotoSimilarity(gen_fp, ref_fp) for ref_fp in ref_fps)
        similarities.append(max_sim)

    return float(np.mean(similarities))


def compute_fragment_similarity(
    generated_mols: List[Chem.Mol],
    reference_mols: List[Chem.Mol],
) -> float:
    """Compute cosine similarity of BRICS fragment distributions."""
    from rdkit.Chem import BRICS

    def get_fragment_counts(mols):
        fragment_counts = Counter()
        for mol in mols:
            try:
                frags = BRICS.BRICSDecompose(mol)
                fragment_counts.update(frags)
            except Exception:
                pass
        return fragment_counts

    gen_counts = get_fragment_counts(generated_mols)
    ref_counts = get_fragment_counts(reference_mols)

    all_frags = set(gen_counts.keys()) | set(ref_counts.keys())
    if len(all_frags) == 0:
        return 0.0

    gen_vec = np.array([gen_counts.get(f, 0) for f in all_frags])
    ref_vec = np.array([ref_counts.get(f, 0) for f in all_frags])

    # Cosine similarity
    norm = np.linalg.norm(gen_vec) * np.linalg.norm(ref_vec)
    if norm == 0:
        return 0.0
    return float(np.dot(gen_vec, ref_vec) / norm)


def compute_scaffold_similarity(
    generated_mols: List[Chem.Mol],
    reference_mols: List[Chem.Mol],
) -> float:
    """Compute cosine similarity of Murcko scaffold distributions."""
    def get_scaffold_counts(mols):
        scaffold_counts = Counter()
        for mol in mols:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffold_counts[scaffold_smiles] += 1
            except Exception:
                pass
        return scaffold_counts

    gen_counts = get_scaffold_counts(generated_mols)
    ref_counts = get_scaffold_counts(reference_mols)

    all_scaffolds = set(gen_counts.keys()) | set(ref_counts.keys())
    if len(all_scaffolds) == 0:
        return 0.0

    gen_vec = np.array([gen_counts.get(s, 0) for s in all_scaffolds])
    ref_vec = np.array([ref_counts.get(s, 0) for s in all_scaffolds])

    norm = np.linalg.norm(gen_vec) * np.linalg.norm(ref_vec)
    if norm == 0:
        return 0.0
    return float(np.dot(gen_vec, ref_vec) / norm)


def compute_property_stats(mols: List[Chem.Mol]) -> Dict[str, Tuple[float, float]]:
    """Compute mean and std of molecular properties.

    Returns dict with keys: logp, sa, qed, mw
    """
    from src.evaluation.sa_score import compute_sa_score

    properties = {
        'logp': [],
        'sa': [],
        'qed': [],
        'mw': [],
    }

    for mol in mols:
        try:
            properties['logp'].append(Crippen.MolLogP(mol))
            properties['sa'].append(compute_sa_score(mol))
            properties['qed'].append(QED.qed(mol))
            properties['mw'].append(Descriptors.MolWt(mol))
        except Exception:
            pass

    stats = {}
    for key, values in properties.items():
        if len(values) > 0:
            stats[key] = (float(np.mean(values)), float(np.std(values)))
        else:
            stats[key] = (0.0, 0.0)

    return stats


def compute_nspdk_mmd(
    generated_mols: List[Chem.Mol],
    reference_mols: List[Chem.Mol],
) -> float:
    """Compute MMD using NSPDK graph kernel.

    This is a simplified version using Morgan fingerprints as a proxy.
    For exact NSPDK, install the eden library.
    """
    try:
        from eden.graph import vectorize

        def mol_to_nx(mol):
            G = nx.Graph()
            for atom in mol.GetAtoms():
                G.add_node(atom.GetIdx(), label=atom.GetSymbol())
            for bond in mol.GetBonds():
                G.add_edge(
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    label=str(bond.GetBondType())
                )
            return G

        gen_graphs = [mol_to_nx(m) for m in generated_mols]
        ref_graphs = [mol_to_nx(m) for m in reference_mols]

        gen_features = vectorize(gen_graphs, complexity=4)
        ref_features = vectorize(ref_graphs, complexity=4)

        # Compute MMD
        gen_mean = np.mean(gen_features.toarray(), axis=0)
        ref_mean = np.mean(ref_features.toarray(), axis=0)

        return float(np.linalg.norm(gen_mean - ref_mean))
    except ImportError:
        # Fallback to fingerprint-based MMD
        return _compute_fingerprint_mmd(generated_mols, reference_mols)


def _compute_fingerprint_mmd(
    generated_mols: List[Chem.Mol],
    reference_mols: List[Chem.Mol],
) -> float:
    """Compute MMD using Morgan fingerprints."""
    def get_fingerprints(mols):
        fps = []
        for mol in mols:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros(2048)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        return np.array(fps)

    gen_fps = get_fingerprints(generated_mols)
    ref_fps = get_fingerprints(reference_mols)

    gen_mean = gen_fps.mean(axis=0)
    ref_mean = ref_fps.mean(axis=0)

    return float(np.linalg.norm(gen_mean - ref_mean))


class MoleculeEvaluator:
    """Comprehensive molecule generation evaluator."""

    def __init__(
        self,
        atom_decoder: List[str],
        training_smiles: Optional[List[str]] = None,
        reference_mols: Optional[List[Chem.Mol]] = None,
    ):
        self.atom_decoder = atom_decoder
        self.training_smiles = training_smiles or []
        self.reference_mols = reference_mols or []

    def evaluate(
        self,
        node_types: np.ndarray,
        adj_matrices: np.ndarray,
        compute_fcd_flag: bool = True,
        compute_nspdk_flag: bool = True,
    ) -> Dict[str, float]:
        """Run full evaluation suite.

        Args:
            node_types: (n_samples, max_atoms) or (n_samples, max_atoms, n_atom_types)
            adj_matrices: (n_samples, max_atoms, max_atoms)
            compute_fcd_flag: whether to compute FCD
            compute_nspdk_flag: whether to compute NSPDK MMD

        Returns:
            Dictionary of metrics
        """
        results = {}

        # Basic metrics
        validity, valid_mols, valid_smiles = compute_validity(
            node_types, adj_matrices, self.atom_decoder
        )
        results['validity'] = validity
        results['num_valid'] = len(valid_mols)

        if len(valid_smiles) > 0:
            results['uniqueness'] = compute_uniqueness(valid_smiles)

            if len(self.training_smiles) > 0:
                results['novelty'] = compute_novelty(valid_smiles, self.training_smiles)

            if len(self.reference_mols) > 0 and len(valid_mols) > 0:
                # Distribution metrics
                if compute_fcd_flag:
                    try:
                        results['fcd'] = compute_fcd(valid_mols, self.reference_mols)
                    except Exception as e:
                        warnings.warn(f"FCD computation failed: {e}")
                        results['fcd'] = float('nan')

                if compute_nspdk_flag:
                    try:
                        results['nspdk_mmd'] = compute_nspdk_mmd(valid_mols, self.reference_mols)
                    except Exception as e:
                        warnings.warn(f"NSPDK MMD computation failed: {e}")
                        results['nspdk_mmd'] = float('nan')

                results['snn'] = compute_snn(valid_mols, self.reference_mols)
                results['frag_sim'] = compute_fragment_similarity(valid_mols, self.reference_mols)
                results['scaffold_sim'] = compute_scaffold_similarity(valid_mols, self.reference_mols)

            # Property statistics
            prop_stats = compute_property_stats(valid_mols)
            for key, (mean, std) in prop_stats.items():
                results[f'{key}_mean'] = mean
                results[f'{key}_std'] = std
        else:
            results['uniqueness'] = 0.0
            results['novelty'] = 0.0

        return results

    def evaluate_from_smiles(self, smiles_list: List[str]) -> Dict[str, float]:
        """Evaluate from SMILES strings directly."""
        results = {}

        valid_mols = []
        valid_smiles = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
                valid_smiles.append(Chem.MolToSmiles(mol))  # Canonicalize

        results['validity'] = len(valid_mols) / len(smiles_list) if len(smiles_list) > 0 else 0.0
        results['num_valid'] = len(valid_mols)

        if len(valid_smiles) > 0:
            results['uniqueness'] = compute_uniqueness(valid_smiles)

            if len(self.training_smiles) > 0:
                results['novelty'] = compute_novelty(valid_smiles, self.training_smiles)

        return results
