#!/bin/bash
# Train VAE on full ZINC250k dataset

uv run python src/scripts/train_vae.py configs/vae/zinc250k_vae.yaml
