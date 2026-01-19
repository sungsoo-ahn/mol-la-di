#!/bin/bash
# Parallel VAE ablation sweep: test different d_latent values with beta=0
# Uses all 4 GPUs to run experiments simultaneously

echo "Starting VAE latent dimension sweep..."
echo "Running 4 experiments in parallel on GPUs 0-3"

# Clean up old runs if they exist
rm -rf data/runs/qm9_vae_tiny_d64
rm -rf data/runs/qm9_vae_tiny_d128
rm -rf data/runs/qm9_vae_tiny_d256
rm -rf data/runs/qm9_vae_tiny_d512

# Run all 4 experiments in parallel
CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train_vae.py configs/vae/qm9_vae_tiny_d64.yaml &
CUDA_VISIBLE_DEVICES=1 uv run python src/scripts/train_vae.py configs/vae/qm9_vae_tiny_d128.yaml &
CUDA_VISIBLE_DEVICES=2 uv run python src/scripts/train_vae.py configs/vae/qm9_vae_tiny_d256.yaml &
CUDA_VISIBLE_DEVICES=3 uv run python src/scripts/train_vae.py configs/vae/qm9_vae_tiny_d512.yaml &

# Wait for all experiments to complete
wait

echo ""
echo "All experiments completed!"
echo ""
echo "Results summary from ablation log:"
cat data/runs/ablation_log.json
