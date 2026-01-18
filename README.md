# MMM - Masked Diffusion Models for Molecule Generation

A research repository for training masked diffusion models for molecule generation, with a transformer-based autoregressive baseline.

## Overview

This repository implements:
- **Autoregressive Baseline**: Transformer-based model that generates molecules by predicting adjacency matrix entries autoregressively
- **Evaluation Metrics**: Validity, Uniqueness, Novelty, FCD, NSPDK MMD, and other molecular metrics (adapted from [GruM](https://github.com/harryjo97/GruM))

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd mmm

# Install dependencies with uv
uv sync

# Or activate the environment manually
source .venv/bin/activate
```

## Project Structure

```
mmm/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model architectures
│   ├── evaluation/     # Evaluation metrics
│   └── scripts/        # Training and sampling scripts
├── configs/
│   ├── training/       # Training configurations
│   └── experiments/    # Experiment configurations
├── scripts/
│   ├── training/       # Training bash scripts
│   └── evaluation/     # Evaluation bash scripts
├── data/               # Dataset storage and outputs
└── scratch/            # Temporary files (gitignored)
```

## Usage

### Training

Train the autoregressive baseline on QM9:

```bash
bash scripts/training/train_qm9.sh
```

Or on ZINC250k:

```bash
bash scripts/training/train_zinc250k.sh
```

### Sampling

Sample from a trained model:

```bash
bash scripts/evaluation/sample_qm9.sh
```

### Custom Configuration

Create a new config in `configs/` and run:

```bash
uv run python src/scripts/train.py configs/your_config.yaml
```

## Datasets

- **QM9**: ~134k small organic molecules with up to 9 heavy atoms
- **ZINC250k**: ~250k drug-like molecules from ZINC database

## Evaluation Metrics

The following metrics are implemented (adapted from GruM):

| Metric | Description |
|--------|-------------|
| Validity | Fraction of chemically valid molecules |
| Uniqueness | Fraction of unique molecules |
| Novelty | Fraction not in training set |
| FCD | Fréchet ChemNet Distance |
| NSPDK MMD | Maximum Mean Discrepancy using NSPDK kernel |
| SNN | Similarity to Nearest Neighbor |
| Fragment Sim | BRICS fragment distribution similarity |
| Scaffold Sim | Murcko scaffold distribution similarity |

## Citation

If using the evaluation metrics, please cite the GruM paper:

```bibtex
@article{jo2023grum,
  title={Graph Generation with Diffusion Mixture},
  author={Jo, Jaehyeong and others},
  journal={arXiv preprint arXiv:2302.03596},
  year={2023}
}
```
