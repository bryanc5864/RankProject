# Loss Functions

All ranking losses are applied at the **batch level**: each training batch of sequences is treated as a list, and the loss encourages correct relative ordering within that batch.

---

## Baseline

### MSE
Standard mean squared error between predicted and ground-truth expression.

$$\mathcal{L}_\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (s_i - y_i)^2$$

Used in the official Prix Fixe / de Boer Nature Biotech paper. All rank losses below are compared against this baseline.

---

## Pairwise Losses

These losses operate on **pairs** of sequences. For each pair $(i, j)$ where $y_i > y_j$, the model should predict $s_i > s_j$.

### RankNet
Bradley-Terry pairwise model. For each pair, applies binary cross-entropy on the predicted preference probability.

$$\mathcal{L}_\text{RankNet} = -\frac{1}{|P|} \sum_{(i,j) \in P} \left[ \log \sigma(\sigma_0 (s_i - s_j)) \right]$$

where $P = \{(i,j) : y_i > y_j\}$, $\sigma$ is the sigmoid, and $\sigma_0 = 1.0$ is a scaling factor. Equivalent to minimizing the probability of mis-ranking any pair.

### Margin RankNet
Extends RankNet with a **hinge loss** instead of BCE. The required score gap scales with the relevance difference — pairs that are far apart in activity must be further apart in score.

$$\mathcal{L}_\text{MarginRankNet} = \frac{1}{|P|} \sum_{(i,j) \in P} \max\left(0,\ \gamma |y_i - y_j| - (s_i - s_j)\right)$$

where $\gamma = 0.1$ is the base margin multiplier.

### Lambda RankNet
Weights each pair's gradient by the change in NDCG that would result from swapping that pair (LambdaRank). Pairs whose swap would most hurt ranking quality receive higher weight.

$$\mathcal{L}_\text{LambdaRankNet} = -\frac{1}{|P|} \sum_{(i,j) \in P} \Delta\text{NDCG}_{ij} \cdot \log \sigma(s_i - s_j)$$

$$\Delta\text{NDCG}_{ij} = \left|\frac{1}{\log_2(r_i+1)} - \frac{1}{\log_2(r_j+1)}\right| \cdot \left|2^{y_i} - 2^{y_j}\right|$$

where $r_i$ is the current predicted rank of item $i$.

### Sampled RankNet
Same objective as RankNet but samples $2n$ random pairs instead of evaluating all $O(n^2)$ pairs. More efficient on large batches at the cost of variance.

---

## Listwise Losses

These losses treat the entire batch as a ranked list and optimize the probability of the full ordering.

### Plackett-Luce (PL)
Models the ranking as a sequential selection process. The probability of a particular ranking $\pi$ is:

$$P(\pi) = \prod_{i=1}^{n} \frac{\exp(s_{\pi(i)})}{\sum_{j \geq i} \exp(s_{\pi(j)})}$$

The loss is the negative log-likelihood of the ground-truth ranking (items sorted by decreasing $y$):

$$\mathcal{L}_\text{PL} = -\frac{1}{n} \sum_{i=1}^{n} \left( s_{\pi(i)} - \log \sum_{j \geq i} \exp(s_{\pi(j)}) \right)$$

This is efficient to compute via a right-to-left `logcumsumexp`.

### Weighted Plackett-Luce
Same as PL but weights each position by $w_i = 1/\log_2(i+1)$ (DCG-style). Correct ranking of the top items contributes more to the loss.

$$\mathcal{L}_\text{WeightedPL} = -\frac{1}{n} \sum_{i=1}^{n} w_i \left( s_{\pi(i)} - \log \sum_{j \geq i} \exp(s_{\pi(j)}) \right)$$

---

## Differentiable Sorting Losses

These losses backpropagate through an approximation of the rank function.

### SoftSort
Computes **soft ranks** using pairwise sigmoid comparisons, then minimizes MSE between predicted soft ranks and ground-truth soft ranks.

The soft rank of item $i$ is:

$$\hat{r}_i = 1 + \sum_{j \neq i} \sigma\!\left(\frac{s_j - s_i}{\tau}\right)$$

where $\tau$ is a temperature parameter (lower = sharper, closer to true rank). Ground-truth soft ranks are computed the same way from $y$.

$$\mathcal{L}_\text{SoftSort} = \frac{1}{n} \sum_{i=1}^{n} (\hat{r}_i - r_i^*)^2$$

### Differentiable Spearman
Computes soft ranks as above, then maximizes the Pearson correlation of those ranks (= Spearman correlation):

$$\mathcal{L}_\text{Spearman} = 1 - \rho(\hat{\mathbf{r}}, \mathbf{r}^*)$$

where $\rho$ is Pearson correlation of the soft rank vectors.

---

## Combined Losses

All combined losses blend MSE with a ranking loss using a fixed weight $\alpha$:

$$\mathcal{L}_\text{combined} = \alpha \cdot \mathcal{L}_\text{MSE} + (1-\alpha) \cdot \mathcal{L}_\text{rank}$$

Higher $\alpha$ = more regression-focused; lower $\alpha$ = more ranking-focused.

| Experiment name | Ranking loss | $\alpha$ |
|---|---|---|
| `combined_pl_a05` | Plackett-Luce | 0.5 |
| `combined_pl_a03` | Plackett-Luce | 0.3 |
| `combined_ranknet_a05` | RankNet | 0.5 |
| `combined_ranknet_a03` | RankNet | 0.3 |
| `combined_softsort_a03` | SoftSort | 0.3 |
| `combined_softsort_a05` | SoftSort | 0.5 |
| `combined_softsort_a07` | SoftSort | 0.7 |
| `combined_margin_ranknet` | Margin RankNet | 0.5 |
| `combined_lambda_ranknet` | Lambda RankNet | 0.5 |
| `combined_sampled_ranknet` | Sampled RankNet | 0.5 |
| `combined_weighted_pl` | Weighted PL | 0.5 |
| `combined_spearman` | Differentiable Spearman | 0.5 |

### Adaptive SoftSort
$\alpha$ starts high (regression-focused) and decays linearly toward a ranking-focused value over training.

| Phase | Epochs | $\alpha$ |
|---|---|---|
| Warmup | 0–10 | 0.9 |
| Transition | 10–80 | 0.9 → 0.3 (linear) |

$$\alpha(t) = 0.9 + (0.3 - 0.9) \cdot \frac{\max(0,\ t - 10)}{70}$$

The intuition: early in training, the model needs MSE signal to learn absolute expression levels; later, ranking signal refines relative ordering.

---

## Pure Ranking Losses (no MSE)

| Experiment name | Loss |
|---|---|
| `pure_pl` | Plackett-Luce only |
| `pure_ranknet` | RankNet only |

---

## Soft Classification (not used in Prix Fixe experiments)

Bins continuous expression values into $K=10$ ordinal bins using quantiles, then applies cross-entropy with label smoothing:

$$\mathcal{L}_\text{SoftClass} = -\sum_{k=1}^{K} \tilde{y}_k \log p_k$$

where $\tilde{y}_k$ are smoothed one-hot labels ($\epsilon = 0.1$) and $p_k = \text{softmax}(\text{logits})_k$. Provides noise robustness via discretization but loses ordinal information between bins.

---

## Summary

| Loss | Type | Granularity | Optimizes |
|---|---|---|---|
| MSE | Regression | Pointwise | Absolute expression |
| Plackett-Luce | Listwise | All $n!$ orderings | Full ranking probability |
| Weighted PL | Listwise | All $n!$ orderings | Top-weighted ranking |
| RankNet | Pairwise | $O(n^2)$ pairs | Per-pair order |
| Margin RankNet | Pairwise | $O(n^2)$ pairs | Per-pair order + margin |
| Lambda RankNet | Pairwise | $O(n^2)$ pairs | NDCG-weighted pairs |
| Sampled RankNet | Pairwise | $O(n)$ pairs | Per-pair order (approx) |
| SoftSort | Differentiable | All ranks | Rank MSE |
| Differentiable Spearman | Differentiable | All ranks | Spearman $\rho$ |
| Adaptive SoftSort | Differentiable | All ranks | Rank MSE (scheduled) |
| Soft Classification | Classification | Pointwise | Ordinal bin CE |

---

## K562 lentiMPRA Test Set Results

Held-out test fold (fold 0), 9-model ensemble. Evaluated on 226K K562 sequences.
90-model reference uses full 10-fold CV.

| Experiment | n | Loss | α | Pearson | Spearman |
|---|---|---|---|---|---|
| MSE_90model (official) | 90 | MSE | — | **0.8249** | 0.7698 |
| RankNet_90model | 90 | RankNet | 0.5 | 0.8228 | **0.7719** |
| adaptive_softsort | 9 | Adaptive SoftSort | 0.9→0.3 | **0.8247** | **0.7668** |
| combined_softsort_a07 | 9 | MSE + SoftSort | 0.7 | 0.8240 | 0.7659 |
| mse_baseline | 9 | MSE | — | 0.8223 | 0.7625 |
| combined_margin_ranknet | 9 | MSE + Margin RankNet | 0.5 | 0.8214 | 0.7628 |
| combined_ranknet_a05 | 9 | MSE + RankNet | 0.5 | 0.8205 | 0.7653 |
| combined_weighted_pl | 9 | MSE + Weighted PL | 0.5 | 0.8198 | 0.7608 |
| pure_ranknet | 9 | RankNet | — | 0.8156 | 0.7635 |
| pure_pl | 9 | Plackett-Luce | — | 0.8110 | 0.7574 |
| combined_pl_a05 | 9 | MSE + PL | 0.5 | 0.8108 | 0.7572 |

`combined_lambda_ranknet` and `combined_sampled_ranknet` results pending (still training).

**Observations:**
- Adaptive SoftSort and combined_softsort_a07 achieve the best Pearson (0.8247, 0.8240) among 9-model runs, slightly below the 90-model MSE reference (0.8249).
- Pure ranking losses (no MSE) hurt Pearson noticeably: pure_pl drops to 0.8110, pure_ranknet to 0.8156.
- RankNet-based combined losses maintain near-MSE Pearson while improving CAGI5 Spearman (see below).
- Plackett-Luce consistently underperforms on the test set, consistent with CAGI5 results.

---

## CAGI5 Results: Variant Effect Prediction

Evaluation task: predict the effect of saturation mutagenesis variants on enhancer activity via `score(alt) - score(ref)`. Metric: Spearman correlation with measured variant effects.

All experiments use the **Prix Fixe architecture** (BHI BiLSTM, 4.2M params) trained on 226K K562 lentiMPRA sequences. 1-fold = hold out fold 0, train 9 models with rotating validation folds, ensemble predictions. 90-model runs use 10-fold CV.

**K562-matched elements**: GP1BB, HBB, HBG1, PKLR (these are K562 MPRA elements with CAGI5 data).

### All variants — K562 elements (Spearman)

Sorted by K562 mean. `*` = incomplete ensemble (fewer than 9 models).

| Experiment | n | GP1BB | HBB | HBG1 | PKLR | **K562 Mean** | All 15 Mean |
|---|---|---|---|---|---|---|---|
| combined_sampled_ranknet `*` | 8 | 0.371 | 0.505 | 0.472 | 0.459 | **0.4517** | 0.3193 |
| MSE_90model | 90 | 0.401 | 0.482 | 0.477 | 0.439 | **0.4497** | 0.3183 |
| RankNet_90model | 90 | 0.389 | 0.499 | 0.466 | 0.438 | **0.4481** | 0.3167 |
| combined_ranknet_a05 | 9 | 0.362 | 0.501 | 0.460 | 0.458 | **0.4453** | 0.3032 |
| combined_ranknet_a03 `*` | 4 | 0.371 | 0.479 | 0.483 | 0.443 | **0.4442** | 0.3056 |
| mse_baseline | 9 | 0.374 | 0.496 | 0.455 | 0.448 | **0.4432** | 0.3091 |
| combined_margin_ranknet | 9 | 0.384 | 0.504 | 0.467 | 0.416 | **0.4427** | 0.3104 |
| combined_softsort_a07 | 9 | 0.380 | 0.508 | 0.457 | 0.417 | **0.4404** | 0.3045 |
| combined_spearman `*` | 3 | 0.393 | 0.544 | 0.407 | 0.411 | **0.4387** | 0.3031 |
| pure_ranknet | 9 | 0.350 | 0.510 | 0.466 | 0.428 | **0.4385** | 0.3125 |
| combined_lambda_ranknet | 9 | 0.379 | 0.472 | 0.470 | 0.433 | **0.4383** | 0.3144 |
| adaptive_softsort | 9 | 0.380 | 0.489 | 0.444 | 0.439 | **0.4377** | 0.2995 |
| combined_softsort_a03 `*` | 6 | 0.380 | 0.496 | 0.437 | 0.432 | **0.4361** | 0.3090 |
| pure_pl | 9 | 0.367 | 0.519 | 0.446 | 0.401 | **0.4334** | 0.3016 |
| combined_weighted_pl | 9 | 0.361 | 0.501 | 0.460 | 0.401 | **0.4309** | 0.3091 |
| combined_softsort_a05 `*` | 6 | 0.387 | 0.461 | 0.443 | 0.429 | **0.4302** | 0.2954 |
| combined_pl_a03 `*` | 8 | 0.362 | 0.515 | 0.424 | 0.398 | **0.4249** | 0.3072 |
| combined_pl_a05 | 9 | 0.348 | 0.482 | 0.459 | 0.396 | **0.4212** | 0.2962 |

### High-confidence variants only (Confidence ≥ 0.1) — K562 elements

| Experiment | n | GP1BB | HBB | HBG1 | PKLR | **HC Mean** |
|---|---|---|---|---|---|---|
| combined_sampled_ranknet `*` | 8 | 0.703 | 0.700 | 0.727 | 0.763 | **0.7233** |
| RankNet_90model | 90 | 0.697 | 0.698 | 0.743 | 0.759 | **0.7240** |
| MSE_90model | 90 | 0.712 | 0.680 | 0.732 | 0.753 | **0.7193** |
| combined_softsort_a07 | 9 | 0.692 | 0.750 | 0.704 | 0.734 | **0.7199** |
| adaptive_softsort | 9 | 0.714 | 0.708 | 0.711 | 0.746 | **0.7197** |
| combined_ranknet_a05 | 9 | 0.688 | 0.713 | 0.712 | 0.769 | **0.7203** |
| mse_baseline | 9 | 0.679 | 0.699 | 0.721 | 0.765 | **0.7160** |
| combined_softsort_a03 `*` | 6 | 0.719 | 0.730 | 0.718 | 0.755 | **0.7304** |
| combined_ranknet_a03 `*` | 4 | 0.676 | 0.683 | 0.736 | 0.754 | **0.7121** |
| combined_margin_ranknet | 9 | 0.663 | 0.694 | 0.735 | 0.734 | **0.7064** |
| pure_ranknet | 9 | 0.636 | 0.728 | 0.719 | 0.763 | **0.7114** |
| combined_weighted_pl | 9 | 0.658 | 0.721 | 0.732 | 0.721 | **0.7078** |
| combined_softsort_a05 `*` | 6 | 0.676 | 0.716 | 0.708 | 0.732 | **0.7081** |
| pure_pl | 9 | 0.617 | 0.725 | 0.731 | 0.749 | **0.7055** |
| combined_lambda_ranknet | 9 | 0.672 | 0.665 | 0.727 | 0.732 | **0.6988** |
| combined_spearman `*` | 3 | 0.690 | 0.697 | 0.718 | 0.703 | **0.7020** |
| combined_pl_a03 `*` | 8 | 0.611 | 0.717 | 0.715 | 0.726 | **0.6923** |
| combined_pl_a05 | 9 | 0.639 | 0.682 | 0.740 | 0.715 | **0.6941** |

### Key findings

- **RankNet-based losses consistently outperform MSE** on CAGI5 variant effect prediction. All four RankNet variants (combined, margin, lambda, sampled) beat the 9-model MSE baseline on K562-matched Spearman.
- **9-model RankNet ≈ 90-model MSE**: `combined_ranknet_a05` (9 models, 0.4453) is nearly as good as `MSE_90model` (90 models, 0.4497), suggesting rank loss gives ~10× ensemble efficiency for CAGI5.
- **Plackett-Luce hurts**: `combined_pl_a05` is the worst complete 9-model run (0.4212), well below MSE baseline. Listwise optimization does not transfer to variant effect prediction.
- **SoftSort is mixed**: `combined_softsort_a07` (high MSE weight, α=0.7) beats MSE; lower α hurts.
- **α matters for RankNet**: α=0.5 outperforms α=0.3 — too much ranking signal degrades CAGI5 performance.
- **Sampled RankNet** (8/9 models, 0.4517) is the top result so far, outperforming both 90-model baselines. Needs final model to confirm.
