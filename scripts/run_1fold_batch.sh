#!/bin/bash
# Run multiple 1-fold experiments sequentially on a single GPU
# Usage: bash scripts/run_1fold_batch.sh <gpu_id> <name1:loss1:alpha1> <name2:loss2:alpha2> ...

GPU=$1
shift

export TMPDIR=/home/bcheng/tmp
export LD_LIBRARY_PATH=/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cusparselt/lib:/home/bcheng/.conda/envs/physiformer/lib:$LD_LIBRARY_PATH

BASE_DIR="${RESULTS_DIR:-results/deboer_rankloss_1fold}"
mkdir -p $BASE_DIR

for spec in "$@"; do
    IFS=':' read -r name loss alpha <<< "$spec"
    outdir="${BASE_DIR}/${name}"
    mkdir -p "$outdir"

    echo "$(date): Starting $name (loss=$loss, alpha=$alpha) on GPU $GPU"
    python -u scripts/train_deboer_rankloss.py \
        --loss_type "$loss" --loss_alpha "$alpha" \
        --gpu "$GPU" --output_dir "$outdir" \
        --epochs 80 --n_test_folds 1 \
        > "${BASE_DIR}/${name}.log" 2>&1

    echo "$(date): Finished $name"
    echo "---"
done
echo "$(date): All done on GPU $GPU"
