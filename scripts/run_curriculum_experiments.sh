#!/bin/bash
#
# Run curriculum and advanced experiments: C1, C2, C3, D1, D2, D3
# These test curriculum learning and advanced noise modeling
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
echo "Curriculum & Advanced Experiments"
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

# C1: Plackett-Luce + Linear Curriculum
echo -n "Testing C1 (plackett_luce + linear curriculum)... "
if python -c "
import sys
sys.path.insert(0, '.')
from src.losses import plackett_luce_loss
from src.models import DREAM_RNN_SingleOutput
from src.data import TierBasedCurriculumSampler, assign_tiers
import torch

model = DREAM_RNN_SingleOutput()
x = torch.randn(4, 4, 230)
y = torch.randn(4)

# Test curriculum
tiers = assign_tiers(y)
sampler = TierBasedCurriculumSampler(tiers, num_samples=4, total_epochs=80, schedule='linear')
sampler.set_epoch(0)
indices = list(sampler)

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

# C2: Plackett-Luce + Stepped Curriculum
echo -n "Testing C2 (plackett_luce + stepped curriculum)... "
if python -c "
import sys
sys.path.insert(0, '.')
from src.data import TierBasedCurriculumSampler, assign_tiers
import torch

y = torch.randn(100)
tiers = assign_tiers(y)
sampler = TierBasedCurriculumSampler(tiers, num_samples=100, total_epochs=80, schedule='stepped')
sampler.set_epoch(40)
indices = list(sampler)
print('OK')
" 2>/dev/null; then
    echo "PASS"
else
    echo "FAIL"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
fi

# C3: Combined + Curriculum
echo -n "Testing C3 (combined + linear curriculum)... "
if python -c "
import sys
sys.path.insert(0, '.')
from src.losses import CombinedLoss
from src.models import DREAM_RNN_SingleOutput
from src.data import CurriculumScheduler
import torch

model = DREAM_RNN_SingleOutput()
criterion = CombinedLoss(alpha=0.5, ranking_loss_fn='plackett_luce')
scheduler = CurriculumScheduler(strategy='tier_based', total_epochs=80)
scheduler.set_epoch(10)

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

# C1: Plackett-Luce + Linear Curriculum
run_experiment "C1_pl_linear_curriculum" "plackett_luce" "dream_rnn_single" "linear" "--temperature 1.0"

# C2: Plackett-Luce + Stepped Curriculum
run_experiment "C2_pl_stepped_curriculum" "plackett_luce" "dream_rnn_single" "stepped" "--temperature 1.0"

# C3: Combined MSE + PL + Linear Curriculum
run_experiment "C3_combined_curriculum" "combined" "dream_rnn_single" "linear" "--alpha 0.5 --ranking_loss plackett_luce"

echo ""
echo "=============================================="
echo "Curriculum experiments completed!"
echo "=============================================="

# ============================================
# ADVANCED EXPERIMENTS (D1-D3)
# ============================================
echo ""
echo "=============================================="
echo "Advanced Experiments (D1-D3)"
echo "=============================================="
echo ""
echo "NOTE: D1-D3 (Domain Adversarial, Bias Factorization) are NOT YET IMPLEMENTED."
echo ""
echo "To implement these, add to src/models/:"
echo "  - D1: ExperimentInvariantModel with gradient reversal layer"
echo "  - D2: BiasFactorizedModel with pre-trained bias model"
echo "  - D3: Combined D1 + D2"
echo ""
echo "See IMPLEMENTATION_PLAN.md Phase 4 for details."
echo ""

# Placeholder commands (will fail until implemented)
# run_experiment "D1_domain_adversarial" "plackett_luce" "dream_rnn_domain_adversarial" "linear" ""
# run_experiment "D2_bias_factorization" "plackett_luce" "dream_rnn_bias_factorized" "linear" ""
# run_experiment "D3_combined_advanced" "plackett_luce" "dream_rnn_full_advanced" "linear" ""

echo ""
echo "=============================================="
echo "All implemented experiments completed!"
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
