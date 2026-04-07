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
