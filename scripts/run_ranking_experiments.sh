#!/bin/bash
#
# Run ranking loss experiments: B2, R1, R2, R3, R4
# These test different loss functions without curriculum learning
#

set -e  # Exit on error

# Configuration
DATA_PATH="data/raw/dream_rnn_lentimpra/data/lentiMPRA_K562_activity_and_aleatoric_data.h5"
OUTPUT_DIR="results"
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-1024}
QUICK_TEST=${QUICK_TEST:-0}  # Set to 1 for quick validation

# Quick test settings (1 epoch, 1% data)
if [ "$QUICK_TEST" -eq 1 ]; then
    EPOCHS=1
    DOWNSAMPLE=0.01
    echo "=== QUICK TEST MODE: 1 epoch, 1% data ==="
else
    DOWNSAMPLE=1.0
fi

cd "$(dirname "$0")/.."

echo "=============================================="
echo "Ranking Loss Experiments"
echo "=============================================="
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "=============================================="

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local loss=$2
    local model=$3
    local extra_args=$4

    echo ""
    echo "----------------------------------------------"
    echo "Running: $exp_name"
    echo "  Loss: $loss"
    echo "  Model: $model"
    echo "----------------------------------------------"

    python scripts/train.py \
        --data "$DATA_PATH" \
        --out "$OUTPUT_DIR" \
        --experiment "$exp_name" \
        --loss "$loss" \
        --model "$model" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --downsample "$DOWNSAMPLE" \
        --gpu "$GPU" \
        --log_every 50 \
        --save_every 10 \
        $extra_args

    echo "Completed: $exp_name"
}

# ============================================
# VALIDATION: Quick test each config first
# ============================================
echo ""
echo "=============================================="
echo "PHASE 1: Validating all configurations..."
echo "=============================================="

VALIDATION_ERRORS=0

# B2: Soft Classification
echo -n "Testing B2 (soft_classification)... "
if python -c "
import sys
sys.path.insert(0, '.')
from src.losses import SoftClassificationLoss
from src.models import DREAM_RNN
import torch

model = DREAM_RNN(n_outputs=10)
criterion = SoftClassificationLoss(n_bins=10)
x = torch.randn(4, 4, 230)
y = torch.randn(4)

out = model(x)
loss = criterion(out, y)
loss.backward()
print('OK')
" 2>/dev/null; then
    echo "PASS"
else
    echo "FAIL"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
fi

# R1: Plackett-Luce
echo -n "Testing R1 (plackett_luce)... "
if python -c "
import sys
sys.path.insert(0, '.')
from src.losses import plackett_luce_loss
from src.models import DREAM_RNN_SingleOutput
import torch

model = DREAM_RNN_SingleOutput()
x = torch.randn(4, 4, 230)
y = torch.randn(4)

out = model(x)
loss = plackett_luce_loss(out, y)
loss.backward()
print('OK')
" 2>/dev/null; then
    echo "PASS"
else
    echo "FAIL"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
fi

# R2: SoftSort (uses pure PyTorch fallback if torchsort not installed)
echo -n "Testing R2 (softsort)... "
if python -c "
import sys
sys.path.insert(0, '.')
from src.losses import softsort_loss
from src.models import DREAM_RNN_SingleOutput
import torch

model = DREAM_RNN_SingleOutput()
x = torch.randn(4, 4, 230)
y = torch.randn(4)

out = model(x)
loss = softsort_loss(out.unsqueeze(0), y.unsqueeze(0))
loss.backward()
print('OK')
" 2>/dev/null; then
    echo "PASS"
else
    echo "FAIL"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
fi

# R3: RankNet
echo -n "Testing R3 (ranknet)... "
if python -c "
import sys
sys.path.insert(0, '.')
from src.losses import ranknet_loss
from src.models import DREAM_RNN_SingleOutput
import torch

model = DREAM_RNN_SingleOutput()
x = torch.randn(4, 4, 230)
y = torch.randn(4)

out = model(x)
loss = ranknet_loss(out, y)
loss.backward()
print('OK')
" 2>/dev/null; then
    echo "PASS"
else
    echo "FAIL"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
fi

# R4: Combined
echo -n "Testing R4 (combined)... "
if python -c "
import sys
sys.path.insert(0, '.')
from src.losses import CombinedLoss
from src.models import DREAM_RNN_SingleOutput
import torch

model = DREAM_RNN_SingleOutput()
criterion = CombinedLoss(alpha=0.5, ranking_loss_fn='plackett_luce')
x = torch.randn(4, 4, 230)
y = torch.randn(4)

out = model(x)
loss = criterion(out, y)
loss.backward()
print('OK')
" 2>/dev/null; then
    echo "PASS"
else
    echo "FAIL"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
fi

echo ""
if [ "$VALIDATION_ERRORS" -gt 0 ]; then
    echo "ERROR: $VALIDATION_ERRORS validation(s) failed. Fix errors before running."
    exit 1
fi
echo "All validations passed!"

# ============================================
# TRAINING: Run experiments
# ============================================
if [ "$QUICK_TEST" -eq 1 ]; then
    echo ""
    echo "=============================================="
    echo "PHASE 2: Quick test training (1 epoch each)..."
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "PHASE 2: Running full training..."
    echo "=============================================="
fi

# B2: Soft Classification (discretization baseline)
run_experiment "B2_soft_classification" "soft_classification" "dream_rnn" "--n_bins 10"

# R1: Plackett-Luce (primary listwise ranking)
run_experiment "R1_plackett_luce" "plackett_luce" "dream_rnn_single" "--temperature 1.0"

# R2: SoftSort (differentiable sorting) - uses pure PyTorch fallback
run_experiment "R2_softsort" "softsort" "dream_rnn_single" ""

# R3: RankNet (pairwise)
run_experiment "R3_ranknet" "ranknet" "dream_rnn_single" ""

# R4: Combined MSE + Plackett-Luce
run_experiment "R4_combined" "combined" "dream_rnn_single" "--alpha 0.5 --ranking_loss plackett_luce"

echo ""
echo "=============================================="
echo "All ranking experiments completed!"
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
