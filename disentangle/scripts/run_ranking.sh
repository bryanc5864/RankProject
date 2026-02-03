#!/bin/bash
# Run ranking loss experiments across architectures (to be launched after baselines finish)
set -e

PYTHON=/home/bcheng/.conda/envs/mpralegnet/bin/python3
WORKDIR=/home/bcheng/RankProject/disentangle
DATA=$WORKDIR/data/processed/dream_K562.h5
SEED=42
BS=256

cd $WORKDIR
mkdir -p results

echo "Starting ranking loss training runs..."
echo "$(date): Launching 4 architectures on 4 GPUs"

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON train.py \
    --architecture cnn --condition ranking_only \
    --data $DATA --output_dir results/cnn_ranking_seed${SEED} \
    --seed $SEED --gpu 0 --batch_size $BS \
    > results/cnn_ranking_seed${SEED}.log 2>&1 &
echo "CNN PID: $!"

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON train.py \
    --architecture dilated_cnn --condition ranking_only \
    --data $DATA --output_dir results/dilated_cnn_ranking_seed${SEED} \
    --seed $SEED --gpu 0 --batch_size $BS \
    > results/dilated_cnn_ranking_seed${SEED}.log 2>&1 &
echo "Dilated CNN PID: $!"

CUDA_VISIBLE_DEVICES=2 nohup $PYTHON train.py \
    --architecture bilstm --condition ranking_only \
    --data $DATA --output_dir results/bilstm_ranking_seed${SEED} \
    --seed $SEED --gpu 0 --batch_size $BS \
    > results/bilstm_ranking_seed${SEED}.log 2>&1 &
echo "BiLSTM PID: $!"

CUDA_VISIBLE_DEVICES=3 nohup $PYTHON train.py \
    --architecture transformer --condition ranking_only \
    --data $DATA --output_dir results/transformer_ranking_seed${SEED} \
    --seed $SEED --gpu 0 --batch_size $BS \
    > results/transformer_ranking_seed${SEED}.log 2>&1 &
echo "Transformer PID: $!"

echo "All ranking runs launched."
