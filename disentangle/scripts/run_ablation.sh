#!/bin/bash
# Phase 5: Full Ablation Study
# 4 architectures x 6 conditions x 3 seeds = 72 runs
set -euo pipefail

cd "$(dirname "$0")/.."

ARCHITECTURES=("cnn" "dilated_cnn" "bilstm" "transformer")
CONDITIONS=("baseline_mse" "ranking_only" "contrastive_only" "consensus_only" "ranking_contrastive" "full_disentangle")
SEEDS=(42 123 456)

DATA_FILES="data/processed/encode_lentimpra_K562.h5 data/processed/dream_K562.h5 data/processed/whg_starrseq_K562.h5"
PAIRED_SEQ="data/processed/paired_sequences.h5"
SPLITS="data/processed/splits.json"

for ARCH in "${ARCHITECTURES[@]}"; do
    # Map architecture to config file
    case $ARCH in
        cnn)           MODEL_CFG="configs/models/cnn_basset.yaml" ;;
        dilated_cnn)   MODEL_CFG="configs/models/dilated_cnn_basenji.yaml" ;;
        bilstm)        MODEL_CFG="configs/models/bilstm_dream.yaml" ;;
        transformer)   MODEL_CFG="configs/models/transformer_lite.yaml" ;;
    esac

    for COND in "${CONDITIONS[@]}"; do
        TRAIN_CFG="configs/training/${COND}.yaml"

        for SEED in "${SEEDS[@]}"; do
            RUN_NAME="${ARCH}_${COND}_seed${SEED}"
            OUTPUT_DIR="results/ablation/${RUN_NAME}"

            # Skip completed runs
            if [ -f "${OUTPUT_DIR}/evaluation_complete.json" ]; then
                echo "Skipping ${RUN_NAME} (already completed)"
                continue
            fi

            echo "=== Running: ${RUN_NAME} ==="

            # Train
            python -m training.trainer \
                --architecture "$ARCH" \
                --training_condition "$COND" \
                --config "$TRAIN_CFG" \
                --model_config "$MODEL_CFG" \
                --data_files $DATA_FILES \
                --paired_sequences "$PAIRED_SEQ" \
                --splits_file "$SPLITS" \
                --output_dir "$OUTPUT_DIR" \
                --seed "$SEED" \
                --wandb_project disentangle_ablation \
                --wandb_name "$RUN_NAME"

            # Evaluate all 4 tiers
            python -m evaluation.tier1_within_experiment \
                --model_path "${OUTPUT_DIR}/best.pt" \
                --architecture "$ARCH" \
                --data_files $DATA_FILES \
                --splits_file "$SPLITS" \
                --output_file "${OUTPUT_DIR}/tier1_results.json"

            python -m evaluation.tier2_cross_experiment \
                --model_path "${OUTPUT_DIR}/best.pt" \
                --architecture "$ARCH" \
                --held_out_data data/processed/atac_starrseq_K562.h5 \
                --output_file "${OUTPUT_DIR}/tier2_results.json"

            python -m evaluation.tier3_cross_assay \
                --model_path "${OUTPUT_DIR}/best.pt" \
                --architecture "$ARCH" \
                --cagi5_data data/processed/cagi5.h5 \
                --hepg2_data data/processed/encode_lentimpra_HepG2.h5 \
                --output_file "${OUTPUT_DIR}/tier3_results.json"

            python -m evaluation.tier4_representation_quality \
                --model_path "${OUTPUT_DIR}/best.pt" \
                --architecture "$ARCH" \
                --data_files $DATA_FILES \
                --paired_sequences "$PAIRED_SEQ" \
                --output_file "${OUTPUT_DIR}/tier4_results.json"

            echo '{"status": "complete"}' > "${OUTPUT_DIR}/evaluation_complete.json"
            echo "Completed: ${RUN_NAME}"
        done
    done
done

echo "=== All ablation runs completed ==="

# Aggregate results
python scripts/aggregate_results.py \
    --results_dir results/ablation/ \
    --output_file results/ablation/aggregated_results.csv
