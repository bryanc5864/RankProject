#!/bin/bash
# Run all baseline MSE experiments across architectures
# Each architecture gets its own GPU
set -e

PYTHON=/home/bcheng/.conda/envs/mpralegnet/bin/python3
WORKDIR=/home/bcheng/RankProject/disentangle
DATA=$WORKDIR/data/processed/dream_K562.h5
SEED=42
BS=256

cd $WORKDIR
mkdir -p results

echo "Starting baseline MSE training runs..."
echo "$(date): Launching 4 architectures on 4 GPUs"

# CNN on GPU 0
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON train.py \
    --architecture cnn --condition baseline_mse \
    --data $DATA --output_dir results/cnn_baseline_mse_seed${SEED} \
    --seed $SEED --gpu 0 --batch_size $BS \
    > results/cnn_baseline_mse_seed${SEED}.log 2>&1 &
echo "CNN PID: $!"

# Dilated CNN on GPU 1
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON train.py \
    --architecture dilated_cnn --condition baseline_mse \
    --data $DATA --output_dir results/dilated_cnn_baseline_mse_seed${SEED} \
    --seed $SEED --gpu 0 --batch_size $BS \
    > results/dilated_cnn_baseline_mse_seed${SEED}.log 2>&1 &
echo "Dilated CNN PID: $!"

# BiLSTM on GPU 2
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON train.py \
    --architecture bilstm --condition baseline_mse \
    --data $DATA --output_dir results/bilstm_baseline_mse_seed${SEED} \
    --seed $SEED --gpu 0 --batch_size $BS \
    > results/bilstm_baseline_mse_seed${SEED}.log 2>&1 &
echo "BiLSTM PID: $!"

# Transformer on GPU 3
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON train.py \
    --architecture transformer --condition baseline_mse \
    --data $DATA --output_dir results/transformer_baseline_mse_seed${SEED} \
    --seed $SEED --gpu 0 --batch_size $BS \
    > results/transformer_baseline_mse_seed${SEED}.log 2>&1 &
echo "Transformer PID: $!"

echo ""
echo "All baseline runs launched. Monitor with:"
echo "  tail -f results/*_baseline_mse_seed${SEED}.log"
