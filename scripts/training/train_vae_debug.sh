#!/bin/bash
# Train VAE on QM9 debug subset (fast iteration)

uv run python src/scripts/train_vae.py configs/vae/qm9_vae_debug.yaml
