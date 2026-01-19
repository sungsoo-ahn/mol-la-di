#!/bin/bash
# Train diffusion model on full ZINC250k dataset
# Requires trained VAE checkpoint at data/runs/zinc250k_vae/checkpoints/best.pt

uv run python src/scripts/train_diffusion.py configs/diffusion/zinc250k_diffusion.yaml
