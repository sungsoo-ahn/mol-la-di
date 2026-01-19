#!/bin/bash
# Run batch 2 experiments in parallel

set -e

mkdir -p data/qm9_medium_lr3e4 data/qm9_medium_layers8 data/qm9_medium_lr1e3_warmup data/qm9_medium_warmup10

echo "Starting batch 2 experiments..."

(export CUDA_VISIBLE_DEVICES=0 && uv run python src/scripts/train.py configs/training/qm9_medium_lr3e4.yaml > data/qm9_medium_lr3e4/train.log 2>&1) &
PID1=$!

(export CUDA_VISIBLE_DEVICES=1 && uv run python src/scripts/train.py configs/training/qm9_medium_layers8.yaml > data/qm9_medium_layers8/train.log 2>&1) &
PID2=$!

(export CUDA_VISIBLE_DEVICES=2 && uv run python src/scripts/train.py configs/training/qm9_medium_lr1e3_warmup.yaml > data/qm9_medium_lr1e3_warmup/train.log 2>&1) &
PID3=$!

(export CUDA_VISIBLE_DEVICES=3 && uv run python src/scripts/train.py configs/training/qm9_medium_warmup10.yaml > data/qm9_medium_warmup10/train.log 2>&1) &
PID4=$!

echo "PIDs: $PID1 $PID2 $PID3 $PID4"
echo "Waiting for all experiments to complete..."

wait $PID1 $PID2 $PID3 $PID4

echo "Batch 2 completed!"
