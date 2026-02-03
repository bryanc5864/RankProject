#!/bin/bash
# Phase 2: Noise Characterization
# Train separate models on each experiment, compare representations.
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR="results/noise_characterization"
mkdir -p "$OUTPUT_DIR/models" "$OUTPUT_DIR/representations" "$OUTPUT_DIR/figures"

ARCHITECTURES=("cnn" "dilated_cnn" "bilstm")
EXPERIMENTS=(
    "0:data/processed/encode_lentimpra_K562.h5"
    "2:data/processed/dream_K562.h5"
    "3:data/processed/whg_starrseq_K562.h5"
)

# Step 1: Train separate models on each experiment
for ARCH in "${ARCHITECTURES[@]}"; do
    for EXP_SPEC in "${EXPERIMENTS[@]}"; do
        EXP_ID="${EXP_SPEC%%:*}"
        DATA_FILE="${EXP_SPEC#*:}"

        RUN_DIR="$OUTPUT_DIR/models/${ARCH}_exp${EXP_ID}"

        if [ -f "$RUN_DIR/best.pt" ]; then
            echo "Skipping $ARCH exp$EXP_ID (already trained)"
            continue
        fi

        echo "Training $ARCH on experiment $EXP_ID..."
        python -m training.trainer \
            --architecture "$ARCH" \
            --training_condition baseline_mse \
            --config configs/training/baseline_mse.yaml \
            --model_config "configs/models/${ARCH}_basset.yaml" \
            --data_files "$DATA_FILE" \
            --splits_file data/processed/splits.json \
            --output_dir "$RUN_DIR" \
            --seed 42
    done
done

# Step 2: Extract representations for paired sequences
echo "Extracting representations..."
python -c "
from analysis.noise_characterization import extract_representations
# TODO: Load models and paired sequences, extract representations
print('Representation extraction - implement with trained models')
"

# Step 3: Compute CKA and generate UMAP plots
echo "Computing CKA and generating plots..."
python -c "
from analysis.noise_characterization import compute_cka, plot_umap_by_experiment, plot_cka_matrix
# TODO: Load representations and compute CKA matrix
print('CKA analysis - implement with extracted representations')
"

echo "Noise characterization complete."
