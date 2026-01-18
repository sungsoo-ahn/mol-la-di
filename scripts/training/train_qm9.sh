#!/bin/bash
# Train autoregressive transformer on QM9 dataset

set -e

echo "Training AR Transformer on QM9..."
uv run python src/scripts/train.py configs/training/qm9_ar.yaml
