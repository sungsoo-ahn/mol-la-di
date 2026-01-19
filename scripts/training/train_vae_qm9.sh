#!/bin/bash
# Train VAE on full QM9 dataset

uv run python src/scripts/train_vae.py configs/vae/qm9_vae.yaml
