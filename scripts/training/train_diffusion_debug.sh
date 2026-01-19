#!/bin/bash
# Train diffusion model on QM9 debug subset (fast iteration)
# Requires trained VAE checkpoint at data/runs/qm9_vae_debug/checkpoints/best.pt

uv run python src/scripts/train_diffusion.py configs/diffusion/qm9_diffusion_debug.yaml
