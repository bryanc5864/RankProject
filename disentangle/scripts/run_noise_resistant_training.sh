#!/bin/bash
# Noise-Resistant Training Campaign
# Runs all Phase 2-5 experiments across 4 GPUs in parallel
#
# Usage:
#   ./scripts/run_noise_resistant_training.sh [--dry-run] [--seeds "42 123 456"]
#
# Prerequisites:
#   - Data files in data/processed/
#   - CUDA devices 0-3 available
#   - Run from disentangle/ directory

set -e

# Activate conda environment
source ~/.bashrc
conda activate mpralegnet

# Configuration
DATA_K562="data/processed/dream_K562.h5"
DATA_HEPG2="data/processed/dream_HepG2.h5"
PAIRED_DATA="data/processed/paired_K562_HepG2.h5"
RESULTS_DIR="results"
SEEDS="${SEEDS:-42 123 456}"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Noise-Resistant Training Campaign"
echo "========================================"
echo "Seeds: $SEEDS"
echo "Dry run: $DRY_RUN"
echo ""

# Helper function to run training
run_training() {
    local name=$1
    local arch=$2
    local condition=$3
    local gpu=$4
    local extra_args=$5
    local seed=$6

    local output_dir="${RESULTS_DIR}/${name}_seed${seed}"

    if [ -f "${output_dir}/best_model.pt" ]; then
        echo "Skipping ${name}_seed${seed} (already exists)"
        return
    fi

    local cmd="python train.py \
        --architecture ${arch} \
        --condition ${condition} \
        --data ${DATA_K562} ${DATA_HEPG2} \
        --paired_data ${PAIRED_DATA} \
        --output_dir ${output_dir} \
        --seed ${seed} \
        --gpu ${gpu} \
        ${extra_args}"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run on GPU ${gpu}: ${name}_seed${seed}"
        echo "  $cmd"
    else
        echo "Starting ${name}_seed${seed} on GPU ${gpu}..."
        CUDA_VISIBLE_DEVICES=${gpu} $cmd &
    fi
}

# Phase 1: Re-evaluate existing models with matched-BN
# (No training needed, just run evaluate.py on all existing models)
echo "Phase 1: Matched-BN re-evaluation will be done separately with evaluate.py"
echo ""

# Phase 2-4: Primary training campaign
echo "Starting Phase 2-4 training campaign..."
echo ""

for seed in $SEEDS; do
    echo "=== Seed ${seed} ==="

    # N1: dilated_cnn + heteroscedastic (GPU 0)
    run_training "N1_dilated_cnn_heteroscedastic" \
        "dilated_cnn" "heteroscedastic_mse" 0 \
        "--predict_variance" $seed

    # N2: dilated_cnn + heteroscedastic + RC-Mixup (GPU 0)
    run_training "N2_dilated_cnn_heteroscedastic_rcmixup" \
        "dilated_cnn" "heteroscedastic_mse" 0 \
        "--predict_variance --augmentation rc_mixup" $seed

    # N3: dilated_cnn + heteroscedastic + ranking (GPU 1)
    run_training "N3_dilated_cnn_heteroscedastic_ranking" \
        "dilated_cnn" "heteroscedastic_ranking" 1 \
        "--predict_variance" $seed

    # N4: dilated_cnn + heteroscedastic + ranking + RC-Mixup (GPU 1)
    run_training "N4_dilated_cnn_heteroscedastic_ranking_rcmixup" \
        "dilated_cnn" "heteroscedastic_ranking" 1 \
        "--predict_variance --augmentation rc_mixup" $seed

    # N5: dilated_cnn + heteroscedastic + EvoAug (GPU 2)
    run_training "N5_dilated_cnn_heteroscedastic_evoaug" \
        "dilated_cnn" "heteroscedastic_mse" 2 \
        "--predict_variance --augmentation evoaug" $seed

    # N6: dilated_cnn + heteroscedastic + RC-Mixup + noise curriculum (GPU 2)
    run_training "N6_dilated_cnn_heteroscedastic_curriculum" \
        "dilated_cnn" "heteroscedastic_mse" 2 \
        "--predict_variance --augmentation rc_mixup --noise_curriculum" $seed

    # N7: bilstm + heteroscedastic + RC-Mixup (GPU 3)
    run_training "N7_bilstm_heteroscedastic_rcmixup" \
        "bilstm" "heteroscedastic_mse" 3 \
        "--predict_variance --augmentation rc_mixup" $seed

    # N8: bilstm + heteroscedastic + ranking + RC-Mixup (GPU 3)
    run_training "N8_bilstm_heteroscedastic_ranking_rcmixup" \
        "bilstm" "heteroscedastic_ranking" 3 \
        "--predict_variance --augmentation rc_mixup" $seed

    if [ "$DRY_RUN" = false ]; then
        echo "Waiting for seed ${seed} jobs to complete..."
        wait
    fi
    echo ""
done

echo "Phase 2-4 training complete!"
echo ""

# Phase 5: Cleanlab analysis and filtered training
echo "Phase 5: Cleanlab analysis..."

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would run cleanlab analysis"
else
    if [ ! -f "data/processed/sample_weights.npy" ]; then
        echo "Running cleanlab analysis..."
        python scripts/run_cleanlab_analysis.py \
            --data ${DATA_K562} ${DATA_HEPG2} \
            --output_dir data/processed \
            --gpu 0
    else
        echo "Cleanlab weights already exist, skipping analysis"
    fi
fi

# N9: Best combo + cleanlab filtered (after cleanlab analysis)
for seed in $SEEDS; do
    run_training "N9_dilated_cnn_heteroscedastic_filtered" \
        "dilated_cnn" "heteroscedastic_mse" 0 \
        "--predict_variance --augmentation rc_mixup --sample_weights data/processed/sample_weights.npy" $seed
done

if [ "$DRY_RUN" = false ]; then
    wait
fi

echo ""
echo "========================================"
echo "All training complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run evaluation with matched-BN CAGI5:"
echo "   python evaluate.py --results_dir ${RESULTS_DIR} --incremental"
echo ""
echo "2. Aggregate results:"
echo "   python scripts/aggregate_results.py --results_dir ${RESULTS_DIR}"
