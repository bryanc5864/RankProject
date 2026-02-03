#!/bin/bash
# Monitor current batch and launch next batch when done
WORKDIR=/home/bcheng/RankProject/disentangle
SCRIPT_PATH=$WORKDIR/scripts/run_c2_c5.sh

# Wait for current batch to finish
while true; do
    RUNNING=$(ps aux | grep "[t]rain.py" | grep -v grep | grep -c "contrastive_only\|consensus_only\|ranking_contrastive\|full_disentangle")
    # Only count main processes (not DataLoader workers)
    MAIN=$(ps aux | grep "[t]rain.py" | grep -v grep | grep -c "nohup")
    # Simpler: check if any train.py processes are using significant CPU
    ACTIVE=$(ps aux | grep "train.py --architecture" | grep -v grep | wc -l)
    if [ "$ACTIVE" -eq 0 ]; then
        break
    fi
    echo "$(date): $ACTIVE training processes active"
    sleep 60
done
echo "$(date): Batch complete!"
