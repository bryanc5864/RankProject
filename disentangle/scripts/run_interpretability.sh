#!/bin/bash
# Phase 6: Interpretability Analysis
# Compare attributions between baseline and DISENTANGLE models.
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR="results/interpretability"
mkdir -p "$OUTPUT_DIR"

ARCHITECTURES=("cnn" "dilated_cnn" "bilstm" "transformer")

for ARCH in "${ARCHITECTURES[@]}"; do
    echo "=== Attribution analysis for $ARCH ==="

    BASELINE_MODEL="results/ablation/${ARCH}_baseline_mse_seed42/best.pt"
    DISENTANGLE_MODEL="results/ablation/${ARCH}_full_disentangle_seed42/best.pt"

    if [ ! -f "$BASELINE_MODEL" ] || [ ! -f "$DISENTANGLE_MODEL" ]; then
        echo "Skipping $ARCH (models not found)"
        continue
    fi

    python -c "
from analysis.attribution_analysis import compute_attributions, compare_attributions_across_models
# TODO: Load models and compute attributions
print('Attribution analysis for $ARCH - implement with trained models')
"
done

echo "Interpretability analysis complete."
