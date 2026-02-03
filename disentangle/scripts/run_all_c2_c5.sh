#!/bin/bash
# Run all C2-C5 experiments sequentially (4 batches, each using 4 GPUs)
# Batch 1 (C2) is already running - start monitoring from batch 1

WORKDIR=/home/bcheng/RankProject/disentangle
SCRIPT=$WORKDIR/scripts/run_c2_c5.sh

wait_for_batch() {
    local batch_name=$1
    echo "$(date): Waiting for $batch_name to complete..."
    while true; do
        ACTIVE=$(ps aux | grep "train.py --architecture" | grep -v grep | wc -l)
        if [ "$ACTIVE" -le 0 ]; then
            echo "$(date): $batch_name complete!"
            return 0
        fi
        sleep 60
    done
}

# Batch 1 (C2) is already running
wait_for_batch "Batch 1 (C2 contrastive_only)"

# Launch Batch 2 (C3)
echo ""
echo "$(date): Launching Batch 2 (C3 consensus_only)..."
bash "$SCRIPT" 2
wait_for_batch "Batch 2 (C3 consensus_only)"

# Launch Batch 3 (C4)
echo ""
echo "$(date): Launching Batch 3 (C4 ranking_contrastive)..."
bash "$SCRIPT" 3
wait_for_batch "Batch 3 (C4 ranking_contrastive)"

# Launch Batch 4 (C5)
echo ""
echo "$(date): Launching Batch 4 (C5 full_disentangle)..."
bash "$SCRIPT" 4
wait_for_batch "Batch 4 (C5 full_disentangle)"

echo ""
echo "$(date): All C2-C5 experiments complete!"
echo "Results in: $WORKDIR/results/"
