#!/bin/bash
# Quick debug training run with small QM9 subset

set -e

echo "Running debug training on QM9 subset..."
uv run python src/scripts/train.py configs/training/qm9_debug.yaml
