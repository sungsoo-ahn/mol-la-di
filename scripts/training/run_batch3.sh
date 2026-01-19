#!/bin/bash
# Run batch 3 experiments in parallel

set -e

mkdir -p data/qm9_medium_lr2e3_warmup20 data/qm9_medium_best_layers8 data/qm9_medium_lr1e3_warmup20 data/qm9_medium_lr15e4_warmup

echo "Starting batch 3 experiments..."

(export CUDA_VISIBLE_DEVICES=0 && uv run python src/scripts/train.py configs/training/qm9_medium_lr2e3_warmup20.yaml > data/qm9_medium_lr2e3_warmup20/train.log 2>&1) &
PID1=$!

(export CUDA_VISIBLE_DEVICES=1 && uv run python src/scripts/train.py configs/training/qm9_medium_best_layers8.yaml > data/qm9_medium_best_layers8/train.log 2>&1) &
PID2=$!

(export CUDA_VISIBLE_DEVICES=2 && uv run python src/scripts/train.py configs/training/qm9_medium_lr1e3_warmup20.yaml > data/qm9_medium_lr1e3_warmup20/train.log 2>&1) &
PID3=$!

(export CUDA_VISIBLE_DEVICES=3 && uv run python src/scripts/train.py configs/training/qm9_medium_lr15e4_warmup.yaml > data/qm9_medium_lr15e4_warmup/train.log 2>&1) &
PID4=$!

echo "PIDs: $PID1 $PID2 $PID3 $PID4"
echo "Waiting for all experiments to complete..."

wait $PID1 $PID2 $PID3 $PID4

echo "Batch 3 completed!"
