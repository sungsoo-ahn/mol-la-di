"""Synthetic Accessibility Score computation.

Adapted from RDKit SA_Score implementation.
"""

import math
import pickle
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# Fragment scores (will be loaded lazily)
_fragment_scores = None


def _read_fragment_scores():
    """Read fragment scores from pre-computed file or compute on the fly."""
    global _fragment_scores
    if _fragment_scores is not None:
        return _fragment_scores

    _fragment_scores = defaultdict(lambda: -4)

    # Try to load from file
    score_file = Path(__file__).parent / 'fpscores.pkl'
    if score_file.exists():
        with open(score_file, 'rb') as f:
            data = pickle.load(f)
            for frag, score in data.items():
                _fragment_scores[frag] = score
        return _fragment_scores

    # Fallback: empty scores (will use default penalty)
    return _fragment_scores


def compute_sa_score(mol) -> float:
    """Compute synthetic accessibility score for a molecule.

    Score ranges from 1 (easy to synthesize) to 10 (hard to synthesize).

    Args:
        mol: RDKit Mol object

    Returns:
        SA score (float between 1 and 10)
    """
    if mol is None:
        return 10.0

    fragment_scores = _read_fragment_scores()

    # Fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()

    score1 = 0.0
    nf = 0
    for bit_id, count in fps.items():
        nf += count
        score1 += fragment_scores[bit_id] * count
    score1 = score1 / nf if nf > 0 else 0

    # Features score
    n_atoms = mol.GetNumAtoms()
    n_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ri = mol.GetRingInfo()

    # Number of bridgehead atoms
    n_bridgehead = 0
    for ring in ri.AtomRings():
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetDegree() > 2:
                neighbors_in_ring = sum(1 for n in atom.GetNeighbors() if n.GetIdx() in ring)
                if neighbors_in_ring >= 2:
                    n_bridgehead += 1
    n_bridgehead = n_bridgehead // 2  # Each bridgehead counted twice

    # Number of spiro atoms
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)

    # Macrocycle penalty
    n_macrocycles = 0
    for ring in ri.AtomRings():
        if len(ring) > 8:
            n_macrocycles += 1

    # Size penalty
    size_penalty = n_atoms ** 1.005 - n_atoms

    # Stereo penalty
    stereo_penalty = math.log10(n_chiral_centers + 1)

    # Spiro penalty
    spiro_penalty = math.log10(n_spiro + 1)

    # Bridge penalty
    bridge_penalty = math.log10(n_bridgehead + 1)

    # Macrocycle penalty
    macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0

    # Calculate final score
    score2 = (
        -size_penalty
        - stereo_penalty
        - spiro_penalty
        - bridge_penalty
        - macrocycle_penalty
    )

    # Combine scores
    sa_score = score1 + score2

    # Normalize to 1-10 range
    min_score = -4.0
    max_score = 2.5
    sa_score = 11.0 - (sa_score - min_score) / (max_score - min_score) * 9.0
    sa_score = max(1.0, min(10.0, sa_score))

    return sa_score
