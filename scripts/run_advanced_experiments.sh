#!/bin/bash
#
# Run advanced experiments: D1, D2, D3
# These test domain adversarial training and bias factorization
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
echo "Advanced Experiments (D1-D3)"
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
    local curriculum=$4
    local extra_args=$5

    echo ""
    echo "----------------------------------------------"
    echo "Running: $exp_name"
    echo "  Loss: $loss"
    echo "  Model: $model"
    echo "  Curriculum: $curriculum"
    echo "----------------------------------------------"

    python scripts/train.py \
        --data "$DATA_PATH" \
        --out "$OUTPUT_DIR" \
        --experiment "$exp_name" \
        --loss "$loss" \
        --model "$model" \
        --curriculum "$curriculum" \
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

# D1: Domain Adversarial
echo -n "Testing D1 (domain adversarial)... "
if python -c "
import sys
sys.path.insert(0, '.')
from src.losses import plackett_luce_loss
from src.models import DREAM_RNN_DomainAdversarial
import torch

model = DREAM_RNN_DomainAdversarial(n_domains=10)
x = torch.randn(4, 4, 230)
y = torch.randn(4)

activity, domain_logits = model(x)
loss = plackett_luce_loss(activity, y)
loss.backward()
print('OK')
" 2>/dev/null; then
    echo "PASS"
else
    echo "FAIL"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
fi

# D2: Bias Factorized
echo -n "Testing D2 (bias factorized)... "
if python -c "
import sys
sys.path.insert(0, '.')
from src.losses import plackett_luce_loss
from src.models import DREAM_RNN_BiasFactorized
import torch

model = DREAM_RNN_BiasFactorized()
x = torch.randn(4, 4, 230)
y = torch.randn(4)

activity, bias, residual = model(x, return_components=True)
loss = plackett_luce_loss(activity, y)
loss.backward()
print('OK')
" 2>/dev/null; then
    echo "PASS"
else
    echo "FAIL"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
fi

# D3: Full Advanced
echo -n "Testing D3 (full advanced)... "
if python -c "
import sys
sys.path.insert(0, '.')
from src.losses import plackett_luce_loss
from src.models import DREAM_RNN_FullAdvanced
import torch

model = DREAM_RNN_FullAdvanced(n_domains=10)
x = torch.randn(4, 4, 230)
y = torch.randn(4)

activity, domain_logits, bias, residual = model(x, return_all=True)
loss = plackett_luce_loss(activity, y)
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

# D1: Domain Adversarial + Linear Curriculum
run_experiment "D1_domain_adversarial" "plackett_luce" "dream_rnn_domain_adversarial" "linear" "--temperature 1.0 --n_domains 10"

# D2: Bias Factorized + Linear Curriculum
run_experiment "D2_bias_factorized" "plackett_luce" "dream_rnn_bias_factorized" "linear" "--temperature 1.0"

# D3: Full Advanced (Domain Adversarial + Bias Factorized) + Linear Curriculum
run_experiment "D3_full_advanced" "plackett_luce" "dream_rnn_full_advanced" "linear" "--temperature 1.0 --n_domains 10"

echo ""
echo "=============================================="
echo "All advanced experiments completed!"
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
