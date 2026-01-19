#!/bin/bash
# Train MAE on full QM9 dataset

set -e

echo "Training MAE on QM9..."
uv run python src/scripts/train_mae.py configs/mae/qm9_mae.yaml

echo "Done!"
