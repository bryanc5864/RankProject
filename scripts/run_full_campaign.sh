#!/bin/bash
# Noise-Resistant Training Campaign
# Run all 75 experiments across 4 phases

set -e  # Exit on error

# Configuration
DATA_PATH="${DATA_PATH:-data/raw/dream_rnn_lentimpra/lentiMPRA.K562.h5}"
OUTPUT_DIR="${OUTPUT_DIR:-results/noise_resistant}"
GPU="${GPU:-0}"
PARALLEL="${PARALLEL:-1}"  # Number of parallel jobs (if using multiple GPUs)

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Helper function to run a single experiment
run_experiment() {
    local name=$1
    local model=$2
    local loss=$3
    local seed=$4
    shift 4
    local extra_args="$@"

    echo "Running experiment: $name"

    python scripts/train_noise_resistant.py \
        --data "$DATA_PATH" \
        --out "$OUTPUT_DIR" \
        --experiment "$name" \
        --model "$model" \
        --loss "$loss" \
        --seed "$seed" \
        --gpu "$GPU" \
        $extra_args \
        2>&1 | tee "logs/${name}.log"

    echo "Completed: $name"
}

# ============================================================================
# PHASE 1: Core Loss Functions (36 models)
# ============================================================================
phase1() {
    echo "=========================================="
    echo "PHASE 1: Core Loss Functions"
    echo "=========================================="

    # Rank Stability (RS1-RS9)
    for seed in 42 123 456; do
        run_experiment "RS_bilstm_s${seed}" dream_rnn_single rank_stability $seed --noise_k 1.0
    done

    for seed in 42 123 456; do
        run_experiment "RS_factorized_s${seed}" factorized rank_stability $seed --noise_k 1.0
    done

    run_experiment "RS_bilstm_k2" dream_rnn_single rank_stability 42 --noise_k 2.0
    run_experiment "RS_bilstm_k05" dream_rnn_single rank_stability 42 --noise_k 0.5
    run_experiment "RS_factorized_vib" factorized_vib rank_stability 42 --noise_k 1.0 --vib_beta 0.01

    # Distributional Head (DH1-DH9)
    for seed in 42 123 456; do
        run_experiment "DH_dist_s${seed}" dream_rnn_distributional distributional $seed --lambda_var 1.0
    done

    for seed in 42 123 456; do
        run_experiment "DH_hetero_s${seed}" dream_rnn_distributional heteroscedastic_distributional $seed --lambda_var 0.5
    done

    run_experiment "DH_dist_dual" dream_rnn_distributional_dual distributional 42 --lambda_var 1.0
    run_experiment "DH_dist_lv2" dream_rnn_distributional distributional 42 --lambda_var 2.0
    run_experiment "DH_dist_lv05" dream_rnn_distributional distributional 42 --lambda_var 0.5

    # Contrastive Anchor (CA1-CA9)
    for seed in 42 123 456; do
        run_experiment "CA_contrast_s${seed}" dream_rnn_single contrastive_anchor $seed --temperature 0.1
    done

    for seed in 42 123 456; do
        run_experiment "CA_triplet_s${seed}" dream_rnn_single triplet_anchor $seed
    done

    run_experiment "CA_factorized" factorized contrastive_anchor 42 --temperature 0.1
    run_experiment "CA_temp05" dream_rnn_single contrastive_anchor 42 --temperature 0.05
    run_experiment "CA_temp02" dream_rnn_single contrastive_anchor 42 --temperature 0.2

    # Noise Gated (NG1-NG9)
    for seed in 42 123 456; do
        run_experiment "NG_base_s${seed}" dream_rnn_distributional noise_gated $seed --alpha 0.3 --beta 0.1 --noise_k 1.0
    done

    for seed in 42 123 456; do
        run_experiment "NG_adaptive_s${seed}" dream_rnn_distributional adaptive_noise_gated $seed --alpha 0.5 --beta 0.1 --warmup_epochs 10
    done

    run_experiment "NG_mse" dream_rnn_single noise_gated_mse 42 --alpha 0.3 --noise_k 1.0
    run_experiment "NG_alpha05" dream_rnn_distributional noise_gated 42 --alpha 0.5 --beta 0.1
    run_experiment "NG_alpha01" dream_rnn_distributional noise_gated 42 --alpha 0.1 --beta 0.1
}

# ============================================================================
# PHASE 2: Sampling Strategies (18 models)
# ============================================================================
phase2() {
    echo "=========================================="
    echo "PHASE 2: Sampling Strategies"
    echo "=========================================="

    # Quantile Stratified (QS1-QS6)
    for seed in 42 123 456; do
        run_experiment "QS_base_s${seed}" dream_rnn_single mse $seed --sampler quantile_stratified --n_quantiles 10
    done

    for seed in 42 123 456; do
        run_experiment "QS_noise_s${seed}" dream_rnn_single mse $seed --sampler quantile_noise_weighted --n_quantiles 10 --noise_weight 0.5
    done

    # Quantile Curriculum (QC1-QC6)
    for seed in 42 123 456; do
        run_experiment "QC_bilstm_s${seed}" dream_rnn_single mse $seed --sampler quantile_stratified --n_quantiles 5 --quantile_curriculum
    done

    for seed in 42 123 456; do
        run_experiment "QC_factorized_s${seed}" factorized mse $seed --sampler quantile_stratified --n_quantiles 5 --quantile_curriculum
    done

    # Hard Negative Mining (HN1-HN6)
    for seed in 42 123 456; do
        run_experiment "HN_bilstm_s${seed}" dream_rnn_single mse $seed --sampler hard_negative --temperature 1.0
    done

    run_experiment "HN_factorized" factorized mse 42 --sampler hard_negative --temperature 1.0
    run_experiment "HN_temp2" dream_rnn_single mse 42 --sampler hard_negative --temperature 2.0
    run_experiment "HN_temp05" dream_rnn_single mse 42 --sampler hard_negative --temperature 0.5
}

# ============================================================================
# PHASE 3: Representation Decomposition (9 models)
# ============================================================================
phase3() {
    echo "=========================================="
    echo "PHASE 3: Representation Decomposition"
    echo "=========================================="

    # Basic Factorized
    for seed in 42 123 456; do
        run_experiment "FE_base_s${seed}" factorized mse $seed
    done

    # Factorized + VIB
    for seed in 42 123 456; do
        run_experiment "FE_vib_s${seed}" factorized_vib mse $seed --vib_beta 0.01
    done

    # Factorized + GC Adversary
    run_experiment "FE_gc_adv_s42" factorized_gc_adv mse 42 --gc_bins 10
    run_experiment "FE_gc_adv_s123" factorized_gc_adv mse 123 --gc_bins 10
    run_experiment "FE_full" factorized_full mse 42 --vib_beta 0.01 --gc_bins 10
}

# ============================================================================
# PHASE 4: Ablations and Best Combinations (12 models)
# ============================================================================
phase4() {
    echo "=========================================="
    echo "PHASE 4: Ablations & Best Combinations"
    echo "=========================================="

    # Alpha sweep
    for alpha in 0.1 0.2 0.4 0.5; do
        run_experiment "ABL_ng_a${alpha}" dream_rnn_distributional noise_gated 42 --alpha $alpha --beta 0.1
    done

    # Beta sweep
    for beta in 0.05 0.2; do
        run_experiment "ABL_ng_b${beta}" dream_rnn_distributional noise_gated 42 --alpha 0.3 --beta $beta
    done

    # Noise k sweep
    for k in 1.5 2.5; do
        run_experiment "ABL_rs_k${k}" dream_rnn_single rank_stability 42 --noise_k $k
    done

    # Best combinations
    run_experiment "BEST_ng_qs" dream_rnn_distributional noise_gated 42 \
        --alpha 0.3 --beta 0.1 --sampler quantile_stratified --n_quantiles 10

    run_experiment "BEST_ng_qsn" dream_rnn_distributional noise_gated 42 \
        --alpha 0.3 --beta 0.1 --sampler quantile_noise_weighted --n_quantiles 10 --noise_weight 0.5

    run_experiment "BEST_ng_hn" dream_rnn_distributional noise_gated 42 \
        --alpha 0.3 --beta 0.1 --sampler hard_negative

    run_experiment "BEST_full" factorized_full noise_gated 42 \
        --alpha 0.3 --beta 0.1 --vib_beta 0.01 --gc_bins 10 --sampler quantile_stratified --n_quantiles 10
}

# ============================================================================
# Main execution
# ============================================================================

# Parse arguments
PHASE="${1:-all}"

case "$PHASE" in
    1|phase1)
        phase1
        ;;
    2|phase2)
        phase2
        ;;
    3|phase3)
        phase3
        ;;
    4|phase4)
        phase4
        ;;
    all)
        phase1
        phase2
        phase3
        phase4
        ;;
    *)
        echo "Usage: $0 [1|2|3|4|all]"
        echo "  1: Phase 1 - Core Loss Functions (36 models)"
        echo "  2: Phase 2 - Sampling Strategies (18 models)"
        echo "  3: Phase 3 - Representation Decomposition (9 models)"
        echo "  4: Phase 4 - Ablations & Best Combinations (12 models)"
        echo "  all: Run all phases"
        exit 1
        ;;
esac

echo "=========================================="
echo "Campaign completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# Generate summary
echo "Generating results summary..."
python -c "
import os
import json
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
results = []

for exp_dir in output_dir.iterdir():
    if exp_dir.is_dir():
        results_file = exp_dir / 'final_results.json'
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            results.append({
                'experiment': exp_dir.name,
                'test_spearman': data.get('test_metrics', {}).get('spearman', 'N/A'),
                'test_pearson': data.get('test_metrics', {}).get('pearson', 'N/A'),
            })

# Sort by test Spearman
results.sort(key=lambda x: x.get('test_spearman', 0) if isinstance(x.get('test_spearman'), float) else 0, reverse=True)

print('\nTop 10 Results by Test Spearman:')
print('-' * 60)
for r in results[:10]:
    print(f\"{r['experiment']}: Spearman={r['test_spearman']:.4f if isinstance(r['test_spearman'], float) else r['test_spearman']}\")
"
