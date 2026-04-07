#!/bin/bash
# Launch script for noise-resistant training campaign
# Sets up CUDA library paths and runs parallel training

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

# Set CUDA and GCC library paths for PyTorch
export LD_LIBRARY_PATH="/home/bcheng/.conda/envs/phantom/lib:/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cusparselt/lib:/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cublas/lib:/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# Create output directories
mkdir -p results/noise_resistant/logs

# Parse arguments
PHASE="${1:-all}"
GPUS="${2:-0,1,2,3}"
EPOCHS="${3:-80}"

echo "=========================================="
echo "Noise-Resistant Training Campaign"
echo "=========================================="
echo "Phase: $PHASE"
echo "GPUs: $GPUS"
echo "Epochs: $EPOCHS"
echo "Output: results/noise_resistant"
echo "=========================================="

# Run the parallel campaign
exec python scripts/run_parallel_campaign.py \
    --phase "$PHASE" \
    --gpus "$GPUS" \
    --epochs "$EPOCHS" \
    --data "data/raw/dream_rnn_lentimpra/data/lentiMPRA_K562_activity_and_aleatoric_data.h5" \
    --out "results/noise_resistant"
