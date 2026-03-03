#!/usr/bin/env python3
"""
Load trained DREAM RNN from RankProject, run on validation set, generate:
  Scatter: true expression (x) vs absolute error (y)
"""
import sys
import os
import numpy as np
import h5py
import torch
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Style setup ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 250,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

sys.path.insert(0, '/home/bcheng/RankProject')
from src.models.dream_rnn import DREAM_RNN

# ============================================================================
# LOAD DATA & MODEL
# ============================================================================

DATA_PATH = '/home/bcheng/RankProject/data/raw/dream_rnn_lentimpra/data/lentiMPRA_K562_activity_and_aleatoric_data.h5'
MODEL_PATH = '/home/bcheng/RankProject/results/B1_baseline_mse_20260202_235710/checkpoints/best_model.pth'
SAVE_DIR = '/home/bcheng/RankProject/dreamrnn_plots'
DEVICE = torch.device('cuda:0')

os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading data...")
with h5py.File(DATA_PATH, 'r') as f:
    X_val = f['Val/X'][:].astype(np.float32)
    y_val = f['Val/y'][:].astype(np.float32)
    y_val = y_val[:, 0] if y_val.ndim > 1 else y_val

X_val = np.transpose(X_val, (0, 2, 1))  # (N, 4, seq_len)
print(f"Val data: X={X_val.shape}, y={y_val.shape}")
print(f"  y range: [{y_val.min():.3f}, {y_val.max():.3f}], mean={y_val.mean():.3f}")

print("\nLoading model...")
model = DREAM_RNN(in_channels=4, seq_len=230, n_outputs=1, dropout=0.2)
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# INFERENCE
# ============================================================================

print("\nRunning inference...")
all_preds = []
with torch.no_grad():
    for i in range(0, len(X_val), 1024):
        x_batch = torch.from_numpy(X_val[i:i+1024]).to(DEVICE)
        pred = model(x_batch)
        if pred.dim() > 1:
            pred = pred.squeeze(-1)
        all_preds.append(pred.cpu().numpy())

preds = np.concatenate(all_preds)
targets = y_val
abs_error = np.abs(preds - targets)

r, _ = pearsonr(preds, targets)
rho, _ = spearmanr(preds, targets)
print(f"Pearson r = {r:.4f}, Spearman rho = {rho:.4f}")
print(f"Points: {len(preds):,}")

# ============================================================================
# PLOT 1: True Expression vs Absolute Error (scatter)
# ============================================================================

print("\nGenerating scatter plot...")
fig1, ax1 = plt.subplots(figsize=(10, 7))

# Scatter with larger dots
ax1.scatter(targets, abs_error, s=35, alpha=0.7, c='#4A90D9', edgecolors='none',
            rasterized=True, zorder=2)

# LOESS-style smoothed trend via binned means (exclude sparse tail bins)
n_trend_bins = 25
# Use percentile-based bins to avoid sparse bins at extremes
bin_edges = np.percentile(targets, np.linspace(2, 98, n_trend_bins + 1))
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_idx = np.digitize(targets, bin_edges) - 1
bin_means = []
bin_stds = []
valid_centers = []
for i in range(n_trend_bins):
    mask = bin_idx == i
    if mask.sum() >= 20:  # Only use bins with enough points
        bin_means.append(abs_error[mask].mean())
        bin_stds.append(abs_error[mask].std() / np.sqrt(mask.sum()))
        valid_centers.append(bin_centers[i])

ax1.plot(valid_centers, bin_means, color='#D32F2F', lw=2.5,
         label='Binned Mean Error', zorder=5)
ax1.fill_between(valid_centers,
                 np.array(bin_means) - np.array(bin_stds),
                 np.array(bin_means) + np.array(bin_stds),
                 color='#D32F2F', alpha=0.15, zorder=4)

ax1.set_xlabel('True Expression Score')
ax1.set_ylabel('Absolute Error')
ax1.set_title(f'DREAM RNN (K562): True Expression vs Absolute Error\n'
              f'n = {len(targets):,}   |   Pearson r = {r:.3f}   |   '
              f'Spearman \u03c1 = {rho:.3f}', fontweight='bold')
ax1.legend(loc='upper left', frameon=True, fancybox=True,
           edgecolor='#cccccc', framealpha=0.95)
ax1.grid(True, alpha=0.2, linewidth=0.5)
sns.despine(ax=ax1)
fig1.tight_layout()
fig1.savefig(f'{SAVE_DIR}/dreamrnn_scatter_error.png', dpi=250, bbox_inches='tight')
fig1.savefig(f'{SAVE_DIR}/dreamrnn_scatter_error.pdf', bbox_inches='tight')
print(f"  Saved: {SAVE_DIR}/dreamrnn_scatter_error.png")
plt.close(fig1)

print("\nDone!")
