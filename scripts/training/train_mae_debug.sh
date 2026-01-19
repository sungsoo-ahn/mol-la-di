#!/bin/bash
# Train MAE on QM9 debug subset (128 samples)
# For rapid iteration and testing

set -e

echo "Training MAE on QM9 debug dataset..."
uv run python src/scripts/train_mae.py configs/mae/qm9_mae_debug.yaml

echo "Done!"
