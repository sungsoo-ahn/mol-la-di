#!/bin/bash
# Train RAE decoder with debug config
# Requires: pretrained MAE checkpoint at data/runs/qm9_mae/checkpoints/best.pt

set -e

# Check if MAE checkpoint exists
MAE_CHECKPOINT="data/runs/qm9_mae/checkpoints/best.pt"
if [ ! -f "$MAE_CHECKPOINT" ]; then
    echo "MAE checkpoint not found at $MAE_CHECKPOINT"
    echo "Please train MAE first: bash scripts/training/train_mae_qm9.sh"
    exit 1
fi

echo "Training RAE decoder (debug mode)..."
uv run python src/scripts/train_rae_decoder.py configs/rae/qm9_rae_debug.yaml
