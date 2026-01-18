#!/bin/bash
# Sample and evaluate from trained QM9 model

set -e

echo "Sampling from trained QM9 model..."
uv run python src/scripts/sample.py configs/experiments/sample_qm9.yaml
