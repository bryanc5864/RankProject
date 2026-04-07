#!/bin/bash
# Launch 1-fold (9 models) training for many loss types
# Each takes ~15 hours. With 4 GPUs we can run 4 at a time.

export TMPDIR=/home/bcheng/tmp
export LD_LIBRARY_PATH=/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cusparselt/lib:/home/bcheng/.conda/envs/physiformer/lib:$LD_LIBRARY_PATH

BASE_DIR="results/deboer_rankloss_1fold"
mkdir -p $BASE_DIR

run_one() {
    local name=$1
    local loss=$2
    local alpha=$3
    local gpu=$4

    local outdir="${BASE_DIR}/${name}"
    mkdir -p "$outdir"

    echo "Launching $name (loss=$loss, alpha=$alpha) on GPU $gpu"
    TMPDIR=/home/bcheng/tmp nohup env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        python -u scripts/train_deboer_rankloss.py \
        --loss_type "$loss" --loss_alpha "$alpha" \
        --gpu "$gpu" --output_dir "$outdir" \
        --epochs 80 --n_test_folds 1 \
        > "${BASE_DIR}/${name}.log" 2>&1 &
    echo "  PID: $!"
}

# Wave 1: 4 runs on GPUs 0-3
run_one "mse_baseline"       "mse"                    0.5  0
run_one "combined_pl_a05"    "combined_pl"             0.5  1
run_one "combined_ranknet_a05" "combined_ranknet"      0.5  2
run_one "combined_softsort_a05" "combined_softsort"    0.5  3

echo ""
echo "Wave 1 launched. Run 'bash scripts/launch_1fold_rankloss.sh wave2' after wave 1 finishes."
echo ""

if [ "$1" == "wave2" ]; then
    run_one "combined_pl_a03"    "combined_pl"             0.3  0
    run_one "combined_ranknet_a03" "combined_ranknet"      0.3  1
    run_one "combined_softsort_a03" "combined_softsort"    0.3  2
    run_one "combined_margin_ranknet" "combined_margin_ranknet" 0.5 3
fi

if [ "$1" == "wave3" ]; then
    run_one "combined_lambda_ranknet" "combined_lambda_ranknet" 0.5 0
    run_one "adaptive_softsort"  "adaptive_softsort"      0.5  1
    run_one "combined_spearman"  "combined_spearman"      0.5  2
    run_one "combined_weighted_pl" "combined_weighted_pl"  0.5  3
fi

if [ "$1" == "wave4" ]; then
    run_one "pure_pl"            "plackett_luce"          0.5  0
    run_one "pure_ranknet"       "ranknet"                0.5  1
    run_one "combined_sampled_ranknet" "combined_sampled_ranknet" 0.5 2
    run_one "combined_ranknet_a07" "combined_ranknet"     0.7  3
fi
