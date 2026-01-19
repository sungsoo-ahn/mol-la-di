#!/bin/bash
# Overnight Training Pipeline for ZINC250k - 4 MAE variants with RAE Decoders and RAE Diffusion
# Uses 4 GPUs to parallelize each phase
#
# Model Variants:
#   - n15_e25: node_mask=15%, edge_mask=25%
#   - n15_e50: node_mask=15%, edge_mask=50% (baseline)
#   - n15_e75: node_mask=15%, edge_mask=75%
#   - n30_e50: node_mask=30%, edge_mask=50%
#
# Phases:
#   1. MAE Training (500 epochs) - 4 models in parallel
#   2. RAE Decoder Training (200 epochs) - 4 models in parallel
#   3. RAE Diffusion Training (1000 epochs) - 4 models in parallel

set -e

echo "=============================================="
echo "ZINC250k Overnight Training Pipeline"
echo "Started at: $(date)"
echo "=============================================="
echo ""

# Phase 1: MAE Training
echo "=== Phase 1: Training 4 ZINC250k MAEs in parallel (500 epochs) ==="
echo "Starting MAE training at: $(date)"

CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train_mae.py configs/mae/zinc250k_mae_mask_n15_e25.yaml &
PID_MAE_1=$!

CUDA_VISIBLE_DEVICES=1 uv run python src/scripts/train_mae.py configs/mae/zinc250k_mae_mask_n15_e50.yaml &
PID_MAE_2=$!

CUDA_VISIBLE_DEVICES=2 uv run python src/scripts/train_mae.py configs/mae/zinc250k_mae_mask_n15_e75.yaml &
PID_MAE_3=$!

CUDA_VISIBLE_DEVICES=3 uv run python src/scripts/train_mae.py configs/mae/zinc250k_mae_mask_n30_e50.yaml &
PID_MAE_4=$!

echo "MAE PIDs: $PID_MAE_1 $PID_MAE_2 $PID_MAE_3 $PID_MAE_4"
wait $PID_MAE_1 $PID_MAE_2 $PID_MAE_3 $PID_MAE_4

echo "Phase 1 completed at: $(date)"
echo ""

# Phase 2: RAE Decoder Training
echo "=== Phase 2: Training 4 ZINC250k RAE Decoders in parallel (200 epochs) ==="
echo "Starting RAE Decoder training at: $(date)"

CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train_rae_decoder.py configs/rae/zinc250k_rae_mae_n15_e25.yaml &
PID_RAE_1=$!

CUDA_VISIBLE_DEVICES=1 uv run python src/scripts/train_rae_decoder.py configs/rae/zinc250k_rae_mae_n15_e50.yaml &
PID_RAE_2=$!

CUDA_VISIBLE_DEVICES=2 uv run python src/scripts/train_rae_decoder.py configs/rae/zinc250k_rae_mae_n15_e75.yaml &
PID_RAE_3=$!

CUDA_VISIBLE_DEVICES=3 uv run python src/scripts/train_rae_decoder.py configs/rae/zinc250k_rae_mae_n30_e50.yaml &
PID_RAE_4=$!

echo "RAE Decoder PIDs: $PID_RAE_1 $PID_RAE_2 $PID_RAE_3 $PID_RAE_4"
wait $PID_RAE_1 $PID_RAE_2 $PID_RAE_3 $PID_RAE_4

echo "Phase 2 completed at: $(date)"
echo ""

# Phase 3: RAE Diffusion Training
echo "=== Phase 3: Training 4 ZINC250k RAE Diffusion models in parallel (1000 epochs) ==="
echo "Starting RAE Diffusion training at: $(date)"

CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train_latent_diffusion.py configs/rae_diffusion/zinc250k_rae_diffusion_n15_e25.yaml &
PID_DIFF_1=$!

CUDA_VISIBLE_DEVICES=1 uv run python src/scripts/train_latent_diffusion.py configs/rae_diffusion/zinc250k_rae_diffusion_n15_e50.yaml &
PID_DIFF_2=$!

CUDA_VISIBLE_DEVICES=2 uv run python src/scripts/train_latent_diffusion.py configs/rae_diffusion/zinc250k_rae_diffusion_n15_e75.yaml &
PID_DIFF_3=$!

CUDA_VISIBLE_DEVICES=3 uv run python src/scripts/train_latent_diffusion.py configs/rae_diffusion/zinc250k_rae_diffusion_n30_e50.yaml &
PID_DIFF_4=$!

echo "RAE Diffusion PIDs: $PID_DIFF_1 $PID_DIFF_2 $PID_DIFF_3 $PID_DIFF_4"
wait $PID_DIFF_1 $PID_DIFF_2 $PID_DIFF_3 $PID_DIFF_4

echo "Phase 3 completed at: $(date)"
echo ""

echo "=============================================="
echo "All ZINC250k training complete!"
echo "Finished at: $(date)"
echo "=============================================="
echo ""
echo "Output directories:"
echo "  MAE:          data/runs/zinc250k_mae_mask_n15_e25/"
echo "                data/runs/zinc250k_mae_mask_n15_e50/"
echo "                data/runs/zinc250k_mae_mask_n15_e75/"
echo "                data/runs/zinc250k_mae_mask_n30_e50/"
echo ""
echo "  RAE Decoder:  data/runs/zinc250k_rae_mae_n15_e25/"
echo "                data/runs/zinc250k_rae_mae_n15_e50/"
echo "                data/runs/zinc250k_rae_mae_n15_e75/"
echo "                data/runs/zinc250k_rae_mae_n30_e50/"
echo ""
echo "  RAE Diffusion: data/runs/zinc250k_rae_diffusion_n15_e25/"
echo "                 data/runs/zinc250k_rae_diffusion_n15_e50/"
echo "                 data/runs/zinc250k_rae_diffusion_n15_e75/"
echo "                 data/runs/zinc250k_rae_diffusion_n30_e50/"
