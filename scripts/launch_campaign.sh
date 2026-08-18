#!/bin/bash
# launch script for noise-resistant training campaign
# sets up CUDA library paths and runs parallel training

set -e

# navigate to project root
cd "$(dirname "$0")/.."

# set CUDA and GCC library paths for PyTorch
export LD_LIBRARY_PATH="/home/bcheng/.conda/envs/phantom/lib:/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cusparselt/lib:/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cublas/lib:/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# create output directories
mkdir -p results/noise_resistant/logs

PHASE="${1:-all}"
GPUS="${2:-0,1,2,3}"
EPOCHS="${3:-80}"

echo "Noise-Resistant Training Campaign"
echo "Phase: $PHASE"
echo "GPUs: $GPUS"
echo "Epochs: $EPOCHS"
echo "Output: results/noise_resistant"

# run the parallel campaign
exec python scripts/run_parallel_campaign.py \
    --phase "$PHASE" \
    --gpus "$GPUS" \
    --epochs "$EPOCHS" \
    --data "data/raw/dream_rnn_lentimpra/data/lentiMPRA_K562_activity_and_aleatoric_data.h5" \
    --out "results/noise_resistant"
