#!/bin/bash
# Monitor training and incrementally evaluate new models
cd /home/bcheng/RankProject/disentangle

OUTPUT="results/evaluation_final_new.csv"
LOG="results/evaluation_incremental.log"

echo "Starting incremental evaluation monitor at $(date)" | tee "$LOG"

while true; do
    # Count trained models
    TRAINED=$(ls results/*/best_model.pt 2>/dev/null | wc -l)
    EVALUATED=$(tail -n +2 "$OUTPUT" 2>/dev/null | wc -l)

    echo "[$(date '+%H:%M:%S')] Trained: $TRAINED, Evaluated: $EVALUATED" | tee -a "$LOG"

    if [ "$TRAINED" -gt "$EVALUATED" ]; then
        echo "  Running incremental evaluation..." | tee -a "$LOG"
        python -u evaluate.py \
            --results_dir results/ \
            --output "$OUTPUT" \
            --gpu 0 \
            --incremental \
            2>&1 | tee -a "$LOG"
    fi

    # Check if any training is still running
    RUNNING=$(ps aux | grep "python train.py" | grep -v grep | wc -l)
    if [ "$RUNNING" -eq 0 ]; then
        echo "All training complete! Final evaluation..." | tee -a "$LOG"
        # One final incremental run
        python -u evaluate.py \
            --results_dir results/ \
            --output "$OUTPUT" \
            --gpu 0 \
            --incremental \
            2>&1 | tee -a "$LOG"

        FINAL=$(tail -n +2 "$OUTPUT" | wc -l)
        echo "DONE: $FINAL models evaluated total" | tee -a "$LOG"
        break
    fi

    # Wait 5 minutes before checking again
    sleep 300
done

echo "Monitor finished at $(date)" | tee -a "$LOG"
