#!/bin/bash
source ~/.bashrc
conda activate mpralegnet
cd /home/bcheng/RankProject/disentangle
mkdir -p logs

# GPU 0: N1 and N2
CUDA_VISIBLE_DEVICES=0 python train.py --architecture dilated_cnn --condition heteroscedastic_mse \
  --data data/processed/dream_K562.h5 data/processed/dream_HepG2.h5 \
  --paired_data data/processed/paired_K562_HepG2.h5 \
  --output_dir results/N1_dilated_cnn_heteroscedastic_seed42 \
  --seed 42 --gpu 0 --predict_variance > logs/N1_seed42.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=0 python train.py --architecture dilated_cnn --condition heteroscedastic_mse \
  --data data/processed/dream_K562.h5 data/processed/dream_HepG2.h5 \
  --paired_data data/processed/paired_K562_HepG2.h5 \
  --output_dir results/N2_dilated_cnn_heteroscedastic_rcmixup_seed42 \
  --seed 42 --gpu 0 --predict_variance --augmentation rc_mixup > logs/N2_seed42.log 2>&1 &
PID2=$!

# GPU 1: N3 and N4
CUDA_VISIBLE_DEVICES=1 python train.py --architecture dilated_cnn --condition heteroscedastic_ranking \
  --data data/processed/dream_K562.h5 data/processed/dream_HepG2.h5 \
  --paired_data data/processed/paired_K562_HepG2.h5 \
  --output_dir results/N3_dilated_cnn_heteroscedastic_ranking_seed42 \
  --seed 42 --gpu 0 --predict_variance > logs/N3_seed42.log 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=1 python train.py --architecture dilated_cnn --condition heteroscedastic_ranking \
  --data data/processed/dream_K562.h5 data/processed/dream_HepG2.h5 \
  --paired_data data/processed/paired_K562_HepG2.h5 \
  --output_dir results/N4_dilated_cnn_heteroscedastic_ranking_rcmixup_seed42 \
  --seed 42 --gpu 0 --predict_variance --augmentation rc_mixup > logs/N4_seed42.log 2>&1 &
PID4=$!

# GPU 2: N5 and N6
CUDA_VISIBLE_DEVICES=2 python train.py --architecture dilated_cnn --condition heteroscedastic_mse \
  --data data/processed/dream_K562.h5 data/processed/dream_HepG2.h5 \
  --paired_data data/processed/paired_K562_HepG2.h5 \
  --output_dir results/N5_dilated_cnn_heteroscedastic_evoaug_seed42 \
  --seed 42 --gpu 0 --predict_variance --augmentation evoaug > logs/N5_seed42.log 2>&1 &
PID5=$!

CUDA_VISIBLE_DEVICES=2 python train.py --architecture dilated_cnn --condition heteroscedastic_mse \
  --data data/processed/dream_K562.h5 data/processed/dream_HepG2.h5 \
  --paired_data data/processed/paired_K562_HepG2.h5 \
  --output_dir results/N6_dilated_cnn_heteroscedastic_curriculum_seed42 \
  --seed 42 --gpu 0 --predict_variance --augmentation rc_mixup --noise_curriculum > logs/N6_seed42.log 2>&1 &
PID6=$!

# GPU 3: N7 and N8
CUDA_VISIBLE_DEVICES=3 python train.py --architecture bilstm --condition heteroscedastic_mse \
  --data data/processed/dream_K562.h5 data/processed/dream_HepG2.h5 \
  --paired_data data/processed/paired_K562_HepG2.h5 \
  --output_dir results/N7_bilstm_heteroscedastic_rcmixup_seed42 \
  --seed 42 --gpu 0 --predict_variance --augmentation rc_mixup > logs/N7_seed42.log 2>&1 &
PID7=$!

CUDA_VISIBLE_DEVICES=3 python train.py --architecture bilstm --condition heteroscedastic_ranking \
  --data data/processed/dream_K562.h5 data/processed/dream_HepG2.h5 \
  --paired_data data/processed/paired_K562_HepG2.h5 \
  --output_dir results/N8_bilstm_heteroscedastic_ranking_rcmixup_seed42 \
  --seed 42 --gpu 0 --predict_variance --augmentation rc_mixup > logs/N8_seed42.log 2>&1 &
PID8=$!

echo "Started 8 training jobs:"
echo "  GPU 0: N1 (PID $PID1), N2 (PID $PID2)"
echo "  GPU 1: N3 (PID $PID3), N4 (PID $PID4)"
echo "  GPU 2: N5 (PID $PID5), N6 (PID $PID6)"
echo "  GPU 3: N7 (PID $PID7), N8 (PID $PID8)"

echo ""
echo "Waiting for all jobs to complete..."
wait

echo "All training jobs completed!"
echo "Check logs/ directory for output."
