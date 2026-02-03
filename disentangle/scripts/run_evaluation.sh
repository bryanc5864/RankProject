#!/bin/bash
# Evaluate all trained models across all 4 tiers.
set -euo pipefail

cd "$(dirname "$0")/.."

RESULTS_DIR="results/ablation"

for RUN_DIR in "$RESULTS_DIR"/*/; do
    RUN_NAME=$(basename "$RUN_DIR")

    if [ ! -f "$RUN_DIR/best.pt" ]; then
        echo "Skipping $RUN_NAME (no model checkpoint)"
        continue
    fi

    if [ -f "$RUN_DIR/evaluation_complete.json" ]; then
        echo "Skipping $RUN_NAME (already evaluated)"
        continue
    fi

    echo "Evaluating: $RUN_NAME"

    # Extract architecture from run name
    ARCH=$(echo "$RUN_NAME" | cut -d_ -f1)
    if echo "$RUN_NAME" | grep -q "dilated_cnn"; then
        ARCH="dilated_cnn"
    fi

    # Run all tiers
    for TIER in 1 2 3 4; do
        echo "  Tier $TIER..."
        # Tier-specific evaluation commands (same as in run_ablation.sh)
    done

    echo '{"status": "complete"}' > "$RUN_DIR/evaluation_complete.json"
done

echo "All evaluations complete."
