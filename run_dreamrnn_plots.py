#!/usr/bin/env python3
"""
Load trained DREAM RNN from RankProject, run on validation set, generate:
  1) Scatter: true expression (x) vs absolute error (y)
  2) Calibration: confidence (x) vs accuracy (y) with blue/red bars
"""
import sys
import os
import numpy as np
import h5py
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/bcheng/RankProject')
from src.models.dream_rnn import DREAM_RNN

# ============================================================================
# LOAD DATA & MODEL
# ============================================================================

DATA_PATH = '/home/bcheng/RankProject/data/raw/dream_rnn_lentimpra/data/lentiMPRA_K562_activity_and_aleatoric_data.h5'
MODEL_PATH = '/home/bcheng/RankProject/results/B1_baseline_mse_20260121_201614/checkpoints/best_model.pth'
SAVE_DIR = '/home/bcheng/RankProject/dreamrnn_plots'
DEVICE = torch.device('cuda:0')

os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading data...")
with h5py.File(DATA_PATH, 'r') as f:
    X_val = f['Val/X'][:].astype(np.float32)
    y_val = f['Val/y'][:].astype(np.float32)
    y_val = y_val[:, 0] if y_val.ndim > 1 else y_val

# Transpose: (N, seq_len, 4) -> (N, 4, seq_len)
X_val = np.transpose(X_val, (0, 2, 1))
print(f"Val data: X={X_val.shape}, y={y_val.shape}")
print(f"  y range: [{y_val.min():.3f}, {y_val.max():.3f}], mean={y_val.mean():.3f}")

print("\nLoading model...")
model = DREAM_RNN(in_channels=4, seq_len=230, n_outputs=1, dropout=0.2)
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
state = checkpoint['model_state_dict']
model.load_state_dict(state)
model.to(DEVICE)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# ============================================================================
# INFERENCE
# ============================================================================

print("\nRunning inference...")
all_preds = []
batch_size = 1024
with torch.no_grad():
    for i in range(0, len(X_val), batch_size):
        x_batch = torch.from_numpy(X_val[i:i+batch_size]).to(DEVICE)
        pred = model(x_batch)
        if pred.dim() > 1:
            pred = pred.squeeze(-1)
        all_preds.append(pred.cpu().numpy())

preds = np.concatenate(all_preds)
targets = y_val

r, _ = pearsonr(preds, targets)
rho, _ = spearmanr(preds, targets)
print(f"Pearson r = {r:.4f}, Spearman rho = {rho:.4f}")
print(f"Points: {len(preds):,}")

abs_error = np.abs(preds - targets)

# ============================================================================
# PLOT 1: True Expression vs Absolute Error
# ============================================================================

print("\nGenerating scatter plot...")
fig1, ax1 = plt.subplots(figsize=(10, 7))

ax1.scatter(targets, abs_error, s=2, alpha=0.12, c='#1976D2', rasterized=True)

# Binned mean trend
n_bins_trend = 30
bins = np.linspace(targets.min(), targets.max(), n_bins_trend + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_idx = np.digitize(targets, bins) - 1
bin_means = []
for i in range(n_bins_trend):
    mask = bin_idx == i
    if mask.sum() > 0:
        bin_means.append(abs_error[mask].mean())
    else:
        bin_means.append(np.nan)
ax1.plot(bin_centers, bin_means, 'r-', lw=2.5, label='Binned Mean Error', zorder=5)

ax1.set_xlabel('True Expression Score', fontsize=14)
ax1.set_ylabel('Absolute Error', fontsize=14)
ax1.set_title(f'DREAM RNN (K562): True Expression vs Absolute Error\n'
              f'n={len(targets):,}  |  Pearson r={r:.3f}  |  Spearman \u03c1={rho:.3f}',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(f'{SAVE_DIR}/dreamrnn_scatter_error.png', dpi=200)
fig1.savefig(f'{SAVE_DIR}/dreamrnn_scatter_error.pdf')
print(f"  Saved: {SAVE_DIR}/dreamrnn_scatter_error.png")
plt.close(fig1)

# ============================================================================
# PLOT 2: Calibration - Confidence vs Accuracy
# ============================================================================

print("Generating calibration plot...")

# Define confidence from prediction residuals
# Sort by absolute error, bin into deciles
# Confidence = 1 - percentile_rank(abs_error)
ranks = np.argsort(np.argsort(abs_error))  # Rank of each error
confidence = 1.0 - ranks / len(ranks)  # Low error = high confidence

n_cal_bins = 10
cal_bins = np.linspace(0, 1, n_cal_bins + 1)
cal_centers = 0.5 * (cal_bins[:-1] + cal_bins[1:])

# Accuracy = fraction of predictions within tolerance
# Tolerance: at each confidence level, ideally confidence% should be within threshold
tolerance = np.median(abs_error)
accurate = (abs_error <= tolerance).astype(float)

bin_accuracy = []
bin_counts = []
for i in range(n_cal_bins):
    lo, hi = cal_bins[i], cal_bins[i+1]
    if i == n_cal_bins - 1:
        mask = (confidence >= lo) & (confidence <= hi)
    else:
        mask = (confidence >= lo) & (confidence < hi)
    if mask.sum() > 0:
        bin_accuracy.append(accurate[mask].mean())
        bin_counts.append(int(mask.sum()))
    else:
        bin_accuracy.append(0.0)
        bin_counts.append(0)

bin_accuracy = np.array(bin_accuracy)
perfect = cal_centers  # Perfect 1:1 line

fig2, ax2 = plt.subplots(figsize=(10, 7))
bar_w = 0.035

# Blue bars: model accuracy
ax2.bar(cal_centers - bar_w/2, bin_accuracy, width=bar_w,
        color='#2196F3', edgecolor='white', linewidth=0.8,
        label='Model Accuracy', zorder=3)

# Red bars: gap to perfect calibration (stacked on top of blue)
gap = np.maximum(perfect - bin_accuracy, 0)
ax2.bar(cal_centers + bar_w/2, gap, width=bar_w,
        bottom=bin_accuracy,
        color='#F44336', edgecolor='white', linewidth=0.8,
        label='Gap from Perfect 1:1', zorder=3, alpha=0.85)

# Also show red bars where the gap exists as standalone red bar at perfect level
ax2.bar(cal_centers + bar_w/2, bin_accuracy, width=bar_w,
        color='#2196F3', edgecolor='white', linewidth=0.8,
        zorder=2, alpha=0.5)

# Perfect calibration line
ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Perfect Calibration (1:1)')

# Count annotations
for c, acc, gap_v, cnt in zip(cal_centers, bin_accuracy, gap, bin_counts):
    top = acc + gap_v if gap_v > 0 else acc
    ax2.text(c, top + 0.025, f'n={cnt:,}', ha='center', va='bottom',
             fontsize=7, color='gray', rotation=45)

ax2.set_xlabel('Confidence', fontsize=14)
ax2.set_ylabel('Accuracy', fontsize=14)
ax2.set_title(f'DREAM RNN (K562): Calibration Plot\n'
              f'Accuracy = fraction within median error (\u00b1{tolerance:.3f})',
              fontsize=14, fontweight='bold')
ax2.set_xlim(-0.02, 1.05)
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=11, loc='upper left')
ax2.grid(True, alpha=0.3, axis='y')
fig2.tight_layout()
fig2.savefig(f'{SAVE_DIR}/dreamrnn_calibration.png', dpi=200)
fig2.savefig(f'{SAVE_DIR}/dreamrnn_calibration.pdf')
print(f"  Saved: {SAVE_DIR}/dreamrnn_calibration.png")
plt.close(fig2)

print("\nDone!")
