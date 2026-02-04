#!/bin/bash
# Phase 3: New Training Runs
#
# D3: Multi-seed completion (32 runs)
# New experiments: B1, B2, A2, C2, E3 (14 runs)
# Total: 46 new runs across 4 GPUs
#
# Usage:
#   bash scripts/run_new_training.sh
#   # Or run individual GPU scripts:
#   bash scripts/run_new_training.sh --gpu 0  # CNN seeds
#   bash scripts/run_new_training.sh --gpu 1  # Transformer seeds
#   bash scripts/run_new_training.sh --gpu 2  # BiLSTM/DilatedCNN missing + E3
#   bash scripts/run_new_training.sh --gpu 3  # New experiments B1/B2/A2/C2

set -euo pipefail
cd "$(dirname "$0")/.."

# Common data paths
K562="data/processed/dream_K562.h5"
HEPG2="data/processed/dream_HepG2.h5"
PAIRED="data/processed/paired_K562_HepG2.h5"

# ============================================================================
# Helper function: run a single training job
# ============================================================================
run_train() {
    local ARCH=$1
    local COND=$2
    local SEED=$3
    local GPU=$4
    local EXTRA="${5:-}"
    local NAME="${ARCH}_${COND}_seed${SEED}"
    local OUTPUT="results/${NAME}"

    # Skip if already completed
    if [ -f "${OUTPUT}/best_model.pt" ] && [ -f "${OUTPUT}/history.json" ]; then
        echo "SKIP: ${NAME} (already exists)"
        return 0
    fi

    echo "=== Training: ${NAME} on GPU ${GPU} ==="

    # Determine if paired data is needed
    local PAIRED_ARG=""
    case $COND in
        contrastive_only|consensus_only|ranking_contrastive|full_disentangle)
            PAIRED_ARG="--paired_data ${PAIRED}"
            ;;
    esac

    # Determine data files
    local DATA_ARG="--data ${K562} ${HEPG2}"
    case $COND in
        baseline_mse|ranking_only)
            DATA_ARG="--data ${K562}"
            ;;
    esac

    # Train config override
    local CONFIG_ARG=""
    if [ -n "${EXTRA}" ]; then
        CONFIG_ARG="${EXTRA}"
    fi

    python train.py \
        --architecture "${ARCH}" \
        --condition "${COND}" \
        ${DATA_ARG} \
        ${PAIRED_ARG} \
        --output_dir "${OUTPUT}" \
        --seed "${SEED}" \
        --gpu "${GPU}" \
        ${CONFIG_ARG} \
        2>&1 | tee "${OUTPUT}.log"

    echo "DONE: ${NAME}"
}

# ============================================================================
# GPU 0: CNN seeds 123, 456 (12 runs)
# ============================================================================
gpu0_jobs() {
    echo "=== GPU 0: CNN multi-seed completion ==="
    for SEED in 123 456; do
        for COND in baseline_mse ranking_only contrastive_only consensus_only ranking_contrastive full_disentangle; do
            run_train "cnn" "${COND}" "${SEED}" 0
        done
    done
}

# ============================================================================
# GPU 1: Transformer seeds 123, 456 (12 runs)
# ============================================================================
gpu1_jobs() {
    echo "=== GPU 1: Transformer multi-seed completion ==="
    for SEED in 123 456; do
        for COND in baseline_mse ranking_only contrastive_only consensus_only ranking_contrastive full_disentangle; do
            run_train "transformer" "${COND}" "${SEED}" 1
        done
    done
}

# ============================================================================
# GPU 2: BiLSTM + Dilated CNN missing seeds (8 runs) + E3 synthetic (6 runs)
# ============================================================================
gpu2_jobs() {
    echo "=== GPU 2: Missing seeds + E3 synthetic noise ==="

    # BiLSTM missing: contrastive_only seeds 123,456 and consensus_only seeds 123,456
    for SEED in 123 456; do
        run_train "bilstm" "contrastive_only" "${SEED}" 2
        run_train "bilstm" "consensus_only" "${SEED}" 2
    done

    # Dilated CNN missing: contrastive_only seeds 123,456 and consensus_only seeds 123,456
    for SEED in 123 456; do
        run_train "dilated_cnn" "contrastive_only" "${SEED}" 2
        run_train "dilated_cnn" "consensus_only" "${SEED}" 2
    done

    # E3: Synthetic noise experiments (generate data first)
    echo "=== Generating synthetic noise data ==="
    python scripts/generate_synthetic_noise.py \
        --input "${K562}" \
        --output_dir data/processed/synthetic_noise/

    # E3: Train baseline_mse and full_disentangle on each noise type
    for NOISE_TYPE in gc_dependent random_offset multiplicative; do
        SYNTH_CLEAN="data/processed/synthetic_noise/${NOISE_TYPE}/synthetic_clean.h5"
        SYNTH_NOISY="data/processed/synthetic_noise/${NOISE_TYPE}/synthetic_noisy.h5"
        SYNTH_PAIRED="data/processed/synthetic_noise/${NOISE_TYPE}/synthetic_paired.h5"

        # Baseline MSE on noisy data
        NAME="bilstm_e3_${NOISE_TYPE}_baseline_seed42"
        OUTPUT="results/${NAME}"
        if [ ! -f "${OUTPUT}/best_model.pt" ]; then
            echo "=== Training: ${NAME} on GPU 2 ==="
            python train.py \
                --architecture bilstm \
                --condition baseline_mse \
                --data "${SYNTH_CLEAN}" "${SYNTH_NOISY}" \
                --output_dir "${OUTPUT}" \
                --seed 42 --gpu 2 \
                2>&1 | tee "${OUTPUT}.log"
        fi

        # Full disentangle on synthetic paired data
        NAME="bilstm_e3_${NOISE_TYPE}_disentangle_seed42"
        OUTPUT="results/${NAME}"
        if [ ! -f "${OUTPUT}/best_model.pt" ]; then
            echo "=== Training: ${NAME} on GPU 2 ==="
            python train.py \
                --architecture bilstm \
                --condition full_disentangle \
                --data "${SYNTH_CLEAN}" "${SYNTH_NOISY}" \
                --paired_data "${SYNTH_PAIRED}" \
                --output_dir "${OUTPUT}" \
                --seed 42 --gpu 2 \
                2>&1 | tee "${OUTPUT}.log"
        fi
    done
}

# ============================================================================
# GPU 3: New experiments B1, B2, A2, C2 (8 runs)
# ============================================================================
gpu3_jobs() {
    echo "=== GPU 3: New experiments (B1, B2, A2, C2) ==="

    # B1: Two-stage training (bilstm + dilated_cnn)
    # Stage 2 uses existing full_disentangle as stage 1 checkpoint
    for ARCH in bilstm dilated_cnn; do
        STAGE1_CKPT="results/${ARCH}_full_disentangle_seed42/best_model.pt"
        NAME="${ARCH}_two_stage_seed42"
        OUTPUT="results/${NAME}"
        if [ -f "${STAGE1_CKPT}" ] && [ ! -f "${OUTPUT}/best_model.pt" ]; then
            echo "=== Training: ${NAME} (B1 two-stage) on GPU 3 ==="
            python train.py \
                --architecture "${ARCH}" \
                --condition two_stage \
                --data "${K562}" \
                --output_dir "${OUTPUT}" \
                --seed 42 --gpu 3 \
                --two_stage \
                --stage1_checkpoint "${STAGE1_CKPT}" \
                --stage2_lr_factor 0.1 \
                2>&1 | tee "${OUTPUT}.log"
        else
            echo "SKIP: ${NAME} (stage1 checkpoint missing or already done)"
        fi
    done

    # B2: Variant-contrastive (bilstm + dilated_cnn)
    for ARCH in bilstm dilated_cnn; do
        NAME="${ARCH}_variant_contrastive_seed42"
        OUTPUT="results/${NAME}"
        if [ ! -f "${OUTPUT}/best_model.pt" ]; then
            echo "=== Training: ${NAME} (B2 variant-contrastive) on GPU 3 ==="
            python train.py \
                --architecture "${ARCH}" \
                --condition full_disentangle \
                --data "${K562}" "${HEPG2}" \
                --paired_data "${PAIRED}" \
                --output_dir "${OUTPUT}" \
                --seed 42 --gpu 3 \
                --train_config configs/training/variant_contrastive.yaml \
                2>&1 | tee "${OUTPUT}.log"
        fi
    done

    # A2: Hierarchical contrastive (bilstm + dilated_cnn)
    for ARCH in bilstm dilated_cnn; do
        NAME="${ARCH}_hierarchical_contrastive_seed42"
        OUTPUT="results/${NAME}"
        if [ ! -f "${OUTPUT}/best_model.pt" ]; then
            echo "=== Training: ${NAME} (A2 hierarchical-contrastive) on GPU 3 ==="
            python train.py \
                --architecture "${ARCH}" \
                --condition contrastive_only \
                --data "${K562}" "${HEPG2}" \
                --paired_data "${PAIRED}" \
                --output_dir "${OUTPUT}" \
                --seed 42 --gpu 3 \
                --train_config configs/training/hierarchical_contrastive.yaml \
                2>&1 | tee "${OUTPUT}.log"
        fi
    done

    # C2: Quantile-normalized MSE (bilstm + dilated_cnn)
    for ARCH in bilstm dilated_cnn; do
        NAME="${ARCH}_quantile_mse_seed42"
        OUTPUT="results/${NAME}"
        if [ ! -f "${OUTPUT}/best_model.pt" ]; then
            echo "=== Training: ${NAME} (C2 quantile MSE) on GPU 3 ==="
            python train.py \
                --architecture "${ARCH}" \
                --condition baseline_mse \
                --data "${K562}" \
                --output_dir "${OUTPUT}" \
                --seed 42 --gpu 3 \
                --train_config configs/training/quantile_mse.yaml \
                2>&1 | tee "${OUTPUT}.log"
        fi
    done
}

# ============================================================================
# Main: dispatch by --gpu flag or run all
# ============================================================================
if [ "${1:-}" = "--gpu" ]; then
    case "${2:-}" in
        0) gpu0_jobs ;;
        1) gpu1_jobs ;;
        2) gpu2_jobs ;;
        3) gpu3_jobs ;;
        *) echo "Usage: $0 [--gpu 0|1|2|3]"; exit 1 ;;
    esac
else
    # Run all 4 GPU jobs in parallel
    echo "Launching all 4 GPU jobs in parallel..."
    gpu0_jobs &
    PID0=$!
    gpu1_jobs &
    PID1=$!
    gpu2_jobs &
    PID2=$!
    gpu3_jobs &
    PID3=$!

    echo "Waiting for all jobs to complete..."
    echo "  GPU 0 (CNN seeds): PID $PID0"
    echo "  GPU 1 (Transformer seeds): PID $PID1"
    echo "  GPU 2 (Missing seeds + E3): PID $PID2"
    echo "  GPU 3 (B1/B2/A2/C2): PID $PID3"

    wait $PID0 && echo "GPU 0 complete" || echo "GPU 0 FAILED"
    wait $PID1 && echo "GPU 1 complete" || echo "GPU 1 FAILED"
    wait $PID2 && echo "GPU 2 complete" || echo "GPU 2 FAILED"
    wait $PID3 && echo "GPU 3 complete" || echo "GPU 3 FAILED"

    echo ""
    echo "=== All training runs completed ==="
    echo "Run evaluation with:"
    echo "  python evaluate.py --results_dir results/ --output results/evaluation_final_new.csv"
fi
