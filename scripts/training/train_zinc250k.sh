#!/bin/bash
# Train autoregressive transformer on ZINC250k dataset

set -e

echo "Training AR Transformer on ZINC250k..."
uv run python src/scripts/train.py configs/training/zinc250k_ar.yaml
