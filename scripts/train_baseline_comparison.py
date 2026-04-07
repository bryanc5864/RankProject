#!/usr/bin/env python3
"""
Train baseline models using both implementations:
1. de-Boer-Lab (official DREAM challenge) - 1 output, Flatten+Dense final
2. trchristensen-99 (lentiMPRA) - 2 outputs, GlobalAvgPool final

Both trained on lentiMPRA K562 data for fair comparison.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
import h5py
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr


# ============================================================================
# de-Boer-Lab Architecture (Official DREAM)
# ============================================================================

class ConvBlock(nn.Module):
    """Conv block used in de-Boer implementation."""
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x


class DeBoerDREAM(nn.Module):
    """
    Official de-Boer-Lab DREAM architecture adapted for lentiMPRA (230bp).

    Key differences from trchristensen:
    - Core block outputs 320 channels (not 512)
    - Final block uses Flatten + Dense (not GlobalAvgPool)
    - Single output (activity only)
    - Uses HuberLoss
    """
    def __init__(self, seqsize=230, in_channels=4):
        super().__init__()

        # First block: 512 channels
        self.first_conv1 = ConvBlock(in_channels, 256, kernel_size=9, dropout=0.2)
        self.first_conv2 = ConvBlock(in_channels, 256, kernel_size=15, dropout=0.2)

        # Core block: BiLSTM + Conv, outputs 320 channels
        self.lstm = nn.LSTM(input_size=512, hidden_size=320, batch_first=True, bidirectional=True)
        self.core_conv1 = ConvBlock(640, 160, kernel_size=9, dropout=0.2)
        self.core_conv2 = ConvBlock(640, 160, kernel_size=15, dropout=0.2)
        self.core_dropout = nn.Dropout(0.5)

        # Final block: Flatten + Dense (de-Boer style)
        self.flatten = nn.Flatten()
        self.final = nn.Sequential(
            nn.Linear(320 * seqsize, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # First block
        c1 = self.first_conv1(x)
        c2 = self.first_conv2(x)
        x = torch.cat([c1, c2], dim=1)  # (batch, 512, seq)

        # Core block
        x = x.permute(0, 2, 1)  # (batch, seq, 512)
        x, _ = self.lstm(x)  # (batch, seq, 640)
        x = x.permute(0, 2, 1)  # (batch, 640, seq)

        c1 = self.core_conv1(x)
        c2 = self.core_conv2(x)
        x = torch.cat([c1, c2], dim=1)  # (batch, 320, seq)
        x = self.core_dropout(x)

        # Final block
        x = self.flatten(x)
        x = self.final(x)
        return x.squeeze(-1)


# ============================================================================
# trchristensen-99 Architecture (lentiMPRA specific)
# ============================================================================

class TrchristensenDREAM(nn.Module):
    """
    trchristensen-99 lentiMPRA architecture.

    Key differences from de-Boer:
    - Core block outputs 512 channels (not 320)
    - Final block uses GlobalAvgPool + Dense (not Flatten)
    - Two outputs (activity + aleatoric)
    - Uses MSELoss on both outputs
    """
    def __init__(self, seqsize=230, in_channels=4, n_outputs=2):
        super().__init__()
        self.n_outputs = n_outputs

        # First block: 512 channels
        self.first_conv1 = ConvBlock(in_channels, 256, kernel_size=9, dropout=0.2)
        self.first_conv2 = ConvBlock(in_channels, 256, kernel_size=15, dropout=0.2)

        # Core block: BiLSTM + Conv, outputs 512 channels
        self.lstm = nn.LSTM(input_size=512, hidden_size=320, batch_first=True, bidirectional=True)
        self.core_conv1 = ConvBlock(640, 256, kernel_size=9, dropout=0.2)
        self.core_conv2 = ConvBlock(640, 256, kernel_size=15, dropout=0.2)
        self.core_dropout = nn.Dropout(0.5)

        # Final block: GlobalAvgPool + Dense (trchristensen style)
        self.pointwise = nn.Conv1d(512, 256, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final = nn.Linear(256, n_outputs)

    def forward(self, x):
        # First block
        c1 = self.first_conv1(x)
        c2 = self.first_conv2(x)
        x = torch.cat([c1, c2], dim=1)  # (batch, 512, seq)

        # Core block
        x = x.permute(0, 2, 1)  # (batch, seq, 512)
        x, _ = self.lstm(x)  # (batch, seq, 640)
        x = x.permute(0, 2, 1)  # (batch, 640, seq)

        c1 = self.core_conv1(x)
        c2 = self.core_conv2(x)
        x = torch.cat([c1, c2], dim=1)  # (batch, 512, seq)
        x = self.core_dropout(x)

        # Final block
        x = self.pointwise(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.final(x)
        return x


# ============================================================================
# Training Functions
# ============================================================================

def load_data(data_path, use_aleatoric=False):
    """Load lentiMPRA data."""
    print(f"Loading data from {data_path}")

    with h5py.File(data_path, 'r') as f:
        X_train = f['Train/X'][:].astype(np.float32)
        y_train = f['Train/y'][:].astype(np.float32)
        X_val = f['Val/X'][:].astype(np.float32)
        y_val = f['Val/y'][:].astype(np.float32)
        X_test = f['Test/X'][:].astype(np.float32)
        y_test = f['Test/y'][:].astype(np.float32)

    # Transpose to (batch, channels, seq_len)
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    if not use_aleatoric:
        # Extract only activity (column 0)
        y_train = y_train[:, 0] if y_train.ndim > 1 else y_train
        y_val = y_val[:, 0] if y_val.ndim > 1 else y_val
        y_test = y_test[:, 0] if y_test.ndim > 1 else y_test
    else:
        # Keep both columns
        y_train = y_train[:, :2]
        y_val = y_val[:, :2]
        y_test = y_test[:, :2]

    print(f"Data shapes: X_train {X_train.shape}, y_train {y_train.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(model, train_loader, val_loader, device, epochs=80, lr=0.005,
                use_huber=False, model_name="model"):
    """Train a model."""

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    if use_huber:
        criterion = nn.HuberLoss()
    else:
        criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    history = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        # Compute metrics
        if val_preds.ndim > 1:
            # Multi-output: compute for activity (col 0)
            spearman = spearmanr(val_preds[:, 0], val_targets[:, 0])[0]
            pearson = pearsonr(val_preds[:, 0], val_targets[:, 0])[0]
            # Also compute average
            pearson_aleatoric = pearsonr(val_preds[:, 1], val_targets[:, 1])[0]
            avg_pearson = (pearson + pearson_aleatoric) / 2
        else:
            spearman = spearmanr(val_preds, val_targets)[0]
            pearson = pearsonr(val_preds, val_targets)[0]
            avg_pearson = pearson

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_spearman': spearman,
            'val_pearson': pearson,
            'avg_pearson': avg_pearson
        })

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}: Val Loss={val_loss:.4f}, Spearman={spearman:.4f}, Pearson={pearson:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

    # Load best model
    model.load_state_dict(best_state)
    return model, history


def evaluate_model(model, test_loader, device):
    """Evaluate on test set."""
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds.extend(outputs.cpu().numpy())
            targets.extend(batch_y.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    results = {}

    if preds.ndim > 1:
        # Multi-output
        results['activity_spearman'] = spearmanr(preds[:, 0], targets[:, 0])[0]
        results['activity_pearson'] = pearsonr(preds[:, 0], targets[:, 0])[0]
        results['aleatoric_spearman'] = spearmanr(preds[:, 1], targets[:, 1])[0]
        results['aleatoric_pearson'] = pearsonr(preds[:, 1], targets[:, 1])[0]
        results['avg_spearman'] = (results['activity_spearman'] + results['aleatoric_spearman']) / 2
        results['avg_pearson'] = (results['activity_pearson'] + results['aleatoric_pearson']) / 2
    else:
        results['activity_spearman'] = spearmanr(preds, targets)[0]
        results['activity_pearson'] = pearsonr(preds, targets)[0]

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/dream_rnn_lentimpra/data/lentiMPRA_K562_activity_and_aleatoric_data.h5")
    parser.add_argument("--output_dir", default="results/baseline_comparison")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 1. Train de-Boer-Lab model (1 output, activity only)
    # ========================================================================
    print("\n" + "="*80)
    print("Training de-Boer-Lab DREAM (1 output, HuberLoss)")
    print("="*80)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data, use_aleatoric=False)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    deboer_model = DeBoerDREAM(seqsize=230).to(device)
    print(f"de-Boer model parameters: {sum(p.numel() for p in deboer_model.parameters()):,}")

    deboer_model, deboer_history = train_model(
        deboer_model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, use_huber=True,
        model_name="de-Boer"
    )

    deboer_results = evaluate_model(deboer_model, test_loader, device)
    print(f"\nde-Boer Test Results:")
    print(f"  Activity Spearman: {deboer_results['activity_spearman']:.4f}")
    print(f"  Activity Pearson:  {deboer_results['activity_pearson']:.4f}")

    torch.save(deboer_model.state_dict(), output_dir / "deboer_model.pth")

    # ========================================================================
    # 2. Train trchristensen model (2 outputs, activity + aleatoric)
    # ========================================================================
    print("\n" + "="*80)
    print("Training trchristensen-99 DREAM (2 outputs, MSELoss)")
    print("="*80)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data, use_aleatoric=True)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    trchristensen_model = TrchristensenDREAM(seqsize=230, n_outputs=2).to(device)
    print(f"trchristensen model parameters: {sum(p.numel() for p in trchristensen_model.parameters()):,}")

    trchristensen_model, trchristensen_history = train_model(
        trchristensen_model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, use_huber=False,
        model_name="trchristensen"
    )

    trchristensen_results = evaluate_model(trchristensen_model, test_loader, device)
    print(f"\ntrchristensen Test Results:")
    print(f"  Activity Spearman: {trchristensen_results['activity_spearman']:.4f}")
    print(f"  Activity Pearson:  {trchristensen_results['activity_pearson']:.4f}")
    print(f"  Aleatoric Spearman: {trchristensen_results['aleatoric_spearman']:.4f}")
    print(f"  Aleatoric Pearson:  {trchristensen_results['aleatoric_pearson']:.4f}")
    print(f"  Avg Spearman: {trchristensen_results['avg_spearman']:.4f}")
    print(f"  Avg Pearson:  {trchristensen_results['avg_pearson']:.4f}")

    torch.save(trchristensen_model.state_dict(), output_dir / "trchristensen_model.pth")

    # ========================================================================
    # 3. Save comparison
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    comparison = {
        'de-Boer-Lab': {
            'architecture': '1 output, Flatten+Dense, HuberLoss',
            'activity_spearman': deboer_results['activity_spearman'],
            'activity_pearson': deboer_results['activity_pearson'],
        },
        'trchristensen-99': {
            'architecture': '2 outputs, GlobalAvgPool, MSELoss',
            'activity_spearman': trchristensen_results['activity_spearman'],
            'activity_pearson': trchristensen_results['activity_pearson'],
            'aleatoric_spearman': trchristensen_results['aleatoric_spearman'],
            'aleatoric_pearson': trchristensen_results['aleatoric_pearson'],
            'avg_spearman': trchristensen_results['avg_spearman'],
            'avg_pearson': trchristensen_results['avg_pearson'],
        }
    }

    print(f"\n{'Model':<20} {'Activity Sp':<12} {'Activity Pe':<12} {'Avg Sp':<12} {'Avg Pe':<12}")
    print("-" * 70)
    print(f"{'de-Boer-Lab':<20} {deboer_results['activity_spearman']:<12.4f} {deboer_results['activity_pearson']:<12.4f} {'N/A':<12} {'N/A':<12}")
    print(f"{'trchristensen-99':<20} {trchristensen_results['activity_spearman']:<12.4f} {trchristensen_results['activity_pearson']:<12.4f} {trchristensen_results['avg_spearman']:<12.4f} {trchristensen_results['avg_pearson']:<12.4f}")

    with open(output_dir / "comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
