#!/bin/bash
# Medium benchmark training run for optimization experiments

set -e

echo "Running medium benchmark training on QM9..."
uv run python src/scripts/train.py configs/training/qm9_medium.yaml
