"""
Noise Avoidance Evaluation Metrics

Metrics to evaluate whether a model properly handles noise:
1. Noise prediction accuracy (for distributional models)
2. Residual-noise correlation (errors should NOT correlate with noise)
3. Stratified performance (consistent across noise levels)
4. Effective sample contribution (for weighted models)
"""

import torch
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader


class NoiseAvoidanceEvaluator:
    """
    Comprehensive evaluation of noise-aware model behavior.
    """

    def __init__(self, n_quantiles: int = 5):
        """
        Args:
            n_quantiles: Number of noise quantiles for stratified analysis
        """
        self.n_quantiles = n_quantiles

    def noise_prediction_accuracy(self, pred_variance: torch.Tensor,
                                   true_variance: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate how well the model predicts aleatoric uncertainty.

        For distributional models that output (μ, σ²).

        Args:
            pred_variance: Model's predicted variance [n_samples]
            true_variance: True aleatoric variance [n_samples]

        Returns:
            Dict with correlation metrics
        """
        pred_var = pred_variance.view(-1).cpu().numpy()
        true_var = true_variance.view(-1).cpu().numpy()

        # Pearson and Spearman correlations
        pearson_r, pearson_p = stats.pearsonr(pred_var, true_var)
        spearman_r, spearman_p = stats.spearmanr(pred_var, true_var)

        # MSE between predicted and true variance
        mse = np.mean((pred_var - true_var) ** 2)

        # Calibration: predicted variance should match residual magnitude
        # (Requires predictions and targets - handled separately)

        return {
            'variance_pearson_r': float(pearson_r),
            'variance_pearson_p': float(pearson_p),
            'variance_spearman_r': float(spearman_r),
            'variance_spearman_p': float(spearman_p),
            'variance_mse': float(mse),
            'mean_pred_var': float(pred_var.mean()),
            'mean_true_var': float(true_var.mean())
        }

    def residual_noise_correlation(self, predictions: torch.Tensor,
                                    targets: torch.Tensor,
                                    aleatoric_uncertainty: torch.Tensor) -> Dict[str, float]:
        """
        Analyze correlation between prediction errors and noise levels.

        A good noise-resistant model should have LOW correlation:
        errors should not be driven by noise.

        Args:
            predictions: Model predictions [n_samples]
            targets: Ground truth [n_samples]
            aleatoric_uncertainty: Noise proxy [n_samples]

        Returns:
            Dict with correlation metrics
        """
        preds = predictions.view(-1).cpu().numpy()
        targs = targets.view(-1).cpu().numpy()
        noise = aleatoric_uncertainty.view(-1).cpu().numpy()

        # Absolute residuals
        residuals = np.abs(preds - targs)

        # Correlation between residual magnitude and noise
        pearson_r, pearson_p = stats.pearsonr(residuals, noise)
        spearman_r, spearman_p = stats.spearmanr(residuals, noise)

        # Ideal: correlation close to 0 (errors not driven by noise)
        # Baseline: correlation ~ 0.5-0.7 (errors follow noise)

        return {
            'residual_noise_pearson': float(pearson_r),
            'residual_noise_pearson_p': float(pearson_p),
            'residual_noise_spearman': float(spearman_r),
            'residual_noise_spearman_p': float(spearman_p),
            'interpretation': 'good' if abs(pearson_r) < 0.3 else 'moderate' if abs(pearson_r) < 0.5 else 'poor'
        }

    def stratified_performance(self, predictions: torch.Tensor,
                                targets: torch.Tensor,
                                aleatoric_uncertainty: torch.Tensor) -> Dict[str, any]:
        """
        Compute ranking performance stratified by noise level.

        A good model should:
        - Perform better on low-noise samples
        - Maintain reasonable performance on high-noise samples
        - Show smaller performance gap than baseline

        Args:
            predictions: Model predictions [n_samples]
            targets: Ground truth [n_samples]
            aleatoric_uncertainty: Noise proxy [n_samples]

        Returns:
            Dict with per-quantile metrics and summary
        """
        preds = predictions.view(-1).cpu().numpy()
        targs = targets.view(-1).cpu().numpy()
        noise = aleatoric_uncertainty.view(-1).cpu().numpy()

        # Compute noise quantile edges
        quantile_edges = np.percentile(noise, np.linspace(0, 100, self.n_quantiles + 1))
        quantile_assignments = np.digitize(noise, quantile_edges[1:-1])

        results = {
            'per_quantile': {},
            'quantile_edges': quantile_edges.tolist()
        }

        spearman_by_quantile = []

        for q in range(self.n_quantiles):
            mask = quantile_assignments == q
            if mask.sum() < 10:  # Need enough samples
                continue

            q_preds = preds[mask]
            q_targs = targs[mask]

            spearman_r, _ = stats.spearmanr(q_preds, q_targs)
            pearson_r, _ = stats.pearsonr(q_preds, q_targs)
            mse = np.mean((q_preds - q_targs) ** 2)

            results['per_quantile'][f'Q{q}'] = {
                'spearman': float(spearman_r),
                'pearson': float(pearson_r),
                'mse': float(mse),
                'n_samples': int(mask.sum()),
                'noise_range': (float(quantile_edges[q]), float(quantile_edges[q + 1]))
            }
            spearman_by_quantile.append(spearman_r)

        # Summary statistics
        if spearman_by_quantile:
            results['summary'] = {
                'low_noise_spearman': spearman_by_quantile[0],  # Lowest noise quantile
                'high_noise_spearman': spearman_by_quantile[-1],  # Highest noise quantile
                'performance_gap': spearman_by_quantile[0] - spearman_by_quantile[-1],
                'mean_spearman': float(np.mean(spearman_by_quantile)),
                'std_spearman': float(np.std(spearman_by_quantile))
            }

        return results

    def effective_sample_contribution(self, weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute effective sample size (ESS) for heteroscedastic/weighted models.

        ESS = (Σw_i)² / Σw_i²

        Lower ESS indicates more aggressive downweighting of noisy samples.

        Args:
            weights: Per-sample weights used in loss [n_samples]

        Returns:
            Dict with ESS metrics
        """
        w = weights.view(-1).cpu().numpy()
        n = len(w)

        # Normalize weights
        w = w / w.sum()

        # Effective sample size
        ess = 1.0 / np.sum(w ** 2)
        ess_ratio = ess / n  # Ratio to actual sample size

        # Entropy of weight distribution (higher = more uniform)
        entropy = -np.sum(w * np.log(w + 1e-10))
        max_entropy = np.log(n)
        normalized_entropy = entropy / max_entropy

        return {
            'effective_sample_size': float(ess),
            'ess_ratio': float(ess_ratio),
            'weight_entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'min_weight': float(w.min()),
            'max_weight': float(w.max()),
            'weight_std': float(w.std())
        }

    def cross_noise_transfer(self, predictions_low: torch.Tensor,
                              targets_low: torch.Tensor,
                              predictions_high: torch.Tensor,
                              targets_high: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate zero-shot transfer from low-noise to high-noise regime.

        Model trained on low-noise data should still perform on high-noise data.

        Args:
            predictions_low: Predictions on low-noise test set
            targets_low: Targets for low-noise test set
            predictions_high: Predictions on high-noise test set
            targets_high: Targets for high-noise test set

        Returns:
            Dict with transfer metrics
        """
        pred_low = predictions_low.view(-1).cpu().numpy()
        targ_low = targets_low.view(-1).cpu().numpy()
        pred_high = predictions_high.view(-1).cpu().numpy()
        targ_high = targets_high.view(-1).cpu().numpy()

        # Performance on each regime
        spearman_low, _ = stats.spearmanr(pred_low, targ_low)
        spearman_high, _ = stats.spearmanr(pred_high, targ_high)

        # Transfer gap
        transfer_gap = spearman_low - spearman_high

        # Transfer ratio (ideally close to 1)
        transfer_ratio = spearman_high / (spearman_low + 1e-8)

        return {
            'spearman_low_noise': float(spearman_low),
            'spearman_high_noise': float(spearman_high),
            'transfer_gap': float(transfer_gap),
            'transfer_ratio': float(transfer_ratio),
            'transfer_quality': 'good' if transfer_ratio > 0.8 else 'moderate' if transfer_ratio > 0.6 else 'poor'
        }

    def noise_stratified_cv(self, all_predictions: torch.Tensor,
                            all_targets: torch.Tensor,
                            all_noise: torch.Tensor,
                            n_folds: int = 4) -> Dict[str, any]:
        """
        Cross-validation where each fold is a noise stratum.

        Evaluates model's ability to generalize across noise levels.

        Args:
            all_predictions: All predictions [n_samples]
            all_targets: All targets [n_samples]
            all_noise: All noise values [n_samples]
            n_folds: Number of noise strata (folds)

        Returns:
            Dict with per-fold and summary metrics
        """
        preds = all_predictions.view(-1).cpu().numpy()
        targs = all_targets.view(-1).cpu().numpy()
        noise = all_noise.view(-1).cpu().numpy()

        # Assign to folds based on noise
        quantile_edges = np.percentile(noise, np.linspace(0, 100, n_folds + 1))
        fold_assignments = np.digitize(noise, quantile_edges[1:-1])

        results = {'per_fold': {}, 'cross_fold': []}

        for test_fold in range(n_folds):
            test_mask = fold_assignments == test_fold
            train_mask = ~test_mask

            # This simulates training on other folds, testing on this fold
            # Since we have predictions, we just evaluate
            test_preds = preds[test_mask]
            test_targs = targs[test_mask]

            if len(test_preds) < 10:
                continue

            spearman_r, _ = stats.spearmanr(test_preds, test_targs)

            results['per_fold'][f'fold_{test_fold}'] = {
                'spearman': float(spearman_r),
                'n_samples': int(test_mask.sum()),
                'noise_range': (float(quantile_edges[test_fold]),
                               float(quantile_edges[test_fold + 1]))
            }
            results['cross_fold'].append(spearman_r)

        if results['cross_fold']:
            results['summary'] = {
                'mean_spearman': float(np.mean(results['cross_fold'])),
                'std_spearman': float(np.std(results['cross_fold'])),
                'min_spearman': float(np.min(results['cross_fold'])),
                'max_spearman': float(np.max(results['cross_fold']))
            }

        return results

    def comprehensive_evaluation(self, model, dataloader: DataLoader,
                                  device: torch.device,
                                  is_distributional: bool = False) -> Dict[str, any]:
        """
        Run comprehensive noise avoidance evaluation.

        Args:
            model: Trained model
            dataloader: DataLoader with (X, y, noise) tuples
            device: Device for inference
            is_distributional: If True, model outputs (mu, log_var)

        Returns:
            Dict with all evaluation metrics
        """
        model.eval()

        all_preds = []
        all_targets = []
        all_noise = []
        all_pred_var = []

        with torch.no_grad():
            for batch in dataloader:
                X, y = batch[0].to(device), batch[1].to(device)

                # Handle different batch formats
                if len(batch) > 2:
                    noise = batch[2].to(device)
                elif y.dim() > 1 and y.shape[1] > 1:
                    # y[:, 0] = activity, y[:, 1] = noise
                    noise = y[:, 1]
                    y = y[:, 0]
                else:
                    noise = torch.zeros_like(y)

                if is_distributional:
                    mu, log_var = model(X)
                    pred_var = torch.exp(log_var)
                    all_preds.append(mu.cpu())
                    all_pred_var.append(pred_var.cpu())
                else:
                    preds = model(X)
                    all_preds.append(preds.cpu())

                all_targets.append(y.cpu())
                all_noise.append(noise.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_noise = torch.cat(all_noise)

        results = {}

        # Distributional metrics
        if is_distributional and all_pred_var:
            all_pred_var = torch.cat(all_pred_var)
            true_var = all_noise ** 2
            results['variance_prediction'] = self.noise_prediction_accuracy(
                all_pred_var, true_var
            )

        # Residual-noise correlation
        results['residual_noise'] = self.residual_noise_correlation(
            all_preds, all_targets, all_noise
        )

        # Stratified performance
        results['stratified'] = self.stratified_performance(
            all_preds, all_targets, all_noise
        )

        # Cross-noise evaluation
        results['noise_cv'] = self.noise_stratified_cv(
            all_preds, all_targets, all_noise
        )

        return results
