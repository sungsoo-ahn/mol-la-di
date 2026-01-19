#!/bin/bash
# Run QM9 and ZINC training simultaneously on separate GPUs

set -e

mkdir -p data/qm9_ar_baseline data/zinc250k_ar_baseline

echo "Starting QM9 on GPU 0 and ZINC on GPU 1..."

(export CUDA_VISIBLE_DEVICES=0 && uv run python src/scripts/train.py configs/training/qm9_ar.yaml > data/qm9_ar_baseline/train.log 2>&1) &
PID1=$!

(export CUDA_VISIBLE_DEVICES=1 && uv run python src/scripts/train.py configs/training/zinc250k_ar.yaml > data/zinc250k_ar_baseline/train.log 2>&1) &
PID2=$!

echo "QM9 PID: $PID1, ZINC PID: $PID2"
echo "Waiting for both to complete..."

wait $PID1 $PID2

echo "Both experiments completed!"
