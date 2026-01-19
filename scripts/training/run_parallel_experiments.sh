#!/bin/bash
# Run 4 experiments in parallel on different GPUs

set -e

mkdir -p data/qm9_medium data/qm9_medium_lr1e3 data/qm9_medium_lr2e3 data/qm9_medium_d384

echo "Starting 4 parallel experiments..."

# Use subshells to set CUDA_VISIBLE_DEVICES for each experiment
(export CUDA_VISIBLE_DEVICES=0 && uv run python src/scripts/train.py configs/training/qm9_medium.yaml > data/qm9_medium/train.log 2>&1) &
PID1=$!

(export CUDA_VISIBLE_DEVICES=1 && uv run python src/scripts/train.py configs/training/qm9_medium_lr1e3.yaml > data/qm9_medium_lr1e3/train.log 2>&1) &
PID2=$!

(export CUDA_VISIBLE_DEVICES=2 && uv run python src/scripts/train.py configs/training/qm9_medium_lr2e3.yaml > data/qm9_medium_lr2e3/train.log 2>&1) &
PID3=$!

(export CUDA_VISIBLE_DEVICES=3 && uv run python src/scripts/train.py configs/training/qm9_medium_d384.yaml > data/qm9_medium_d384/train.log 2>&1) &
PID4=$!

echo "PIDs: $PID1 $PID2 $PID3 $PID4"
echo "Waiting for all experiments to complete..."

wait $PID1 $PID2 $PID3 $PID4

echo "All experiments completed!"
