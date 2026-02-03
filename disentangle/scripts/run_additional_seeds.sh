#!/bin/bash
# Run key conditions with seeds 123 and 456
# Focus on the most informative conditions: baseline_mse, ranking_only, full_disentangle
# and the best architecture (bilstm) + secondary (dilated_cnn)

PYTHON=/home/bcheng/.conda/envs/mpralegnet/bin/python3
WORKDIR=/home/bcheng/RankProject/disentangle
K562=$WORKDIR/data/processed/dream_K562.h5
HEPG2=$WORKDIR/data/processed/dream_HepG2.h5
PAIRED=$WORKDIR/data/processed/paired_K562_HepG2.h5
RESDIR=$WORKDIR/results

cd "$WORKDIR"

run_experiment() {
    local gpu=$1
    local arch=$2
    local cond=$3
    local seed=$4
    local name="${arch}_${cond}_seed${seed}"

    # Skip if already done
    if [ -f "$RESDIR/$name/test_metrics.json" ]; then
        echo "Skipping $name (already complete)"
        return
    fi

    # Determine if paired data needed
    local paired_arg=""
    local data_arg="--data $K562"
    case "$cond" in
        contrastive_only|consensus_only|ranking_contrastive|full_disentangle)
            paired_arg="--paired_data $PAIRED"
            data_arg="--data $K562 $HEPG2"
            ;;
        baseline_mse|ranking_only)
            data_arg="--data $K562"
            ;;
    esac

    echo "Launching $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup $PYTHON train.py \
        --architecture "$arch" \
        --condition "$cond" \
        $data_arg \
        $paired_arg \
        --output_dir "$RESDIR/$name" \
        --seed $seed \
        --gpu 0 \
        > "$RESDIR/${name}.log" 2>&1 &
}

wait_for_completion() {
    echo "$(date): Waiting for current batch..."
    while true; do
        ACTIVE=$(ps aux | grep "train.py --architecture" | grep -v grep | wc -l)
        if [ "$ACTIVE" -le 0 ]; then
            echo "$(date): Batch complete!"
            return 0
        fi
        sleep 60
    done
}

SEED=${1:-123}
echo "=== Running seed $SEED ==="

# Batch A: baselines (single-experiment data, fast)
echo "--- Batch A: Baselines ---"
run_experiment 0 bilstm baseline_mse $SEED
run_experiment 1 dilated_cnn baseline_mse $SEED
run_experiment 2 bilstm ranking_only $SEED
run_experiment 3 dilated_cnn ranking_only $SEED
wait_for_completion

# Batch B: DISENTANGLE conditions (paired data, slower)
echo "--- Batch B: DISENTANGLE ---"
run_experiment 0 bilstm full_disentangle $SEED
run_experiment 1 dilated_cnn full_disentangle $SEED
run_experiment 2 bilstm ranking_contrastive $SEED
run_experiment 3 dilated_cnn ranking_contrastive $SEED
wait_for_completion

echo "=== Seed $SEED complete ==="
