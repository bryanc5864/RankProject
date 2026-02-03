#!/bin/bash
# Launch C2-C5 experiments across all 4 architectures
# Runs in batches of 4 (one per GPU)

PYTHON=/home/bcheng/.conda/envs/mpralegnet/bin/python3
WORKDIR=/home/bcheng/RankProject/disentangle
K562=$WORKDIR/data/processed/dream_K562.h5
HEPG2=$WORKDIR/data/processed/dream_HepG2.h5
PAIRED=$WORKDIR/data/processed/paired_K562_HepG2.h5
RESDIR=$WORKDIR/results
SEED=42

cd "$WORKDIR"

run_experiment() {
    local gpu=$1
    local arch=$2
    local cond=$3
    local name="${arch}_${cond}_seed${SEED}"
    echo "Launching $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup $PYTHON train.py \
        --architecture "$arch" \
        --condition "$cond" \
        --data "$K562" "$HEPG2" \
        --paired_data "$PAIRED" \
        --output_dir "$RESDIR/$name" \
        --seed $SEED \
        --gpu 0 \
        > "$RESDIR/${name}.log" 2>&1 &
}

BATCH=${1:-1}

case $BATCH in
    1)
        echo "=== Batch 1: C2 contrastive_only ==="
        run_experiment 0 cnn contrastive_only
        run_experiment 1 dilated_cnn contrastive_only
        run_experiment 2 bilstm contrastive_only
        run_experiment 3 transformer contrastive_only
        ;;
    2)
        echo "=== Batch 2: C3 consensus_only ==="
        run_experiment 0 cnn consensus_only
        run_experiment 1 dilated_cnn consensus_only
        run_experiment 2 bilstm consensus_only
        run_experiment 3 transformer consensus_only
        ;;
    3)
        echo "=== Batch 3: C4 ranking_contrastive ==="
        run_experiment 0 cnn ranking_contrastive
        run_experiment 1 dilated_cnn ranking_contrastive
        run_experiment 2 bilstm ranking_contrastive
        run_experiment 3 transformer ranking_contrastive
        ;;
    4)
        echo "=== Batch 4: C5 full_disentangle ==="
        run_experiment 0 cnn full_disentangle
        run_experiment 1 dilated_cnn full_disentangle
        run_experiment 2 bilstm full_disentangle
        run_experiment 3 transformer full_disentangle
        ;;
esac

echo "Launched. Monitor with: tail -f $RESDIR/*.log"
