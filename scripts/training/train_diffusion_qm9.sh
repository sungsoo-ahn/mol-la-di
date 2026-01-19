#!/bin/bash
# Train diffusion model on full QM9 dataset
# Requires trained VAE checkpoint at data/runs/qm9_vae/checkpoints/best.pt

uv run python src/scripts/train_diffusion.py configs/diffusion/qm9_diffusion.yaml
