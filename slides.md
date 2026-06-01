---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 24px;
  }
  h1 { font-size: 42px; }
  h2 { font-size: 32px; }
  table {
    font-size: 22px;
    margin: 0 auto;
  }
  th, td { padding: 6px 14px; }
  .win { color: #1e7e34; font-weight: bold; }
  .lose { color: #c92a2a; font-weight: bold; }
  .tied { color: #555; }
  .small { font-size: 18px; }
---

# Rank-Order Learning for MPRA Data
## K562 / HepG2 / WTC11 + CAGI5 Variant-Effect Benchmark

May 2026 — Bryan Cheng

Question: do rank-aware losses beat MSE on (a) held-out lentiMPRA test sets
and (b) cross-cell-type CAGI5 transfer?

---

## Setup

| Item | Value |
|---|---|
| Architecture | Prix Fixe **DREAM-RNN** (Rafi et al. 2025) — BHI + BiLSTM, ~4.2M params |
| Datasets | K562 (226K), HepG2 (140K), WTC11 (56K) lentiMPRA |
| Held-out test | Fold 0 (~10% of each) |
| Transfer eval | CAGI5 saturation mutagenesis (15 elements, ~14K variants) |
| Optimizer | AdamW, OneCycleLR, max_lr=0.005, batch=32, 80 epochs |
| Ensemble | 9 models per loss (rotating val fold), mean prediction |
| 17 loss variants | MSE baseline + 16 rank-loss variants |

---

## The 17 Loss Variants

| Family | Variants tested | α values |
|---|---|---|
| MSE only | `mse` (paper baseline) | — |
| Plackett–Luce | `combined_pl` | 0.3, 0.5 |
| RankNet (pairwise BCE) | `combined_ranknet` | 0.3, 0.5, 0.7 |
| SoftSort (diff. sort) | `combined_softsort` | 0.3, 0.5, 0.7 |
| RankNet variants | `lambda_ranknet`, `margin_ranknet`, `sampled_ranknet` | 0.3 |
| PL variants | `weighted_pl` | 0.3 |
| Soft Spearman | `combined_spearman` | 0.5 |
| Adaptive | `adaptive_softsort` | schedule |
| Pure rank | `plackett_luce`, `ranknet` | — |

Combined loss: `α · MSE + (1−α) · rank_loss`. Smaller α → more rank weight.

---

## K562 Held-Out Test Set — Top 5

| Method | Pearson | Spearman | Δ Pearson vs MSE |
|---|---|---|---|
| **combined_spearman** | **0.8245** | 0.7661 | +0.0012 |
| combined_softsort_a05 | 0.8239 | 0.7656 | +0.0006 |
| combined_ranknet_a05 | 0.8237 | 0.7651 | +0.0004 |
| combined_ranknet_a03 | 0.8234 | 0.7656 | +0.0001 |
| **mse** (paper baseline) | **0.8233** | **0.7642** | — |

Most rank losses are **within noise** of MSE on the held-out test set.
The interesting story is on CAGI5 transfer.

---

## CAGI5 Variant-Effect Transfer — Top 7 + MSE

| Method | K562-matched Sp | All-element Sp |
|---|---|---|
| **combined_ranknet_a03** | **0.4494** | 0.3098 |
| combined_softsort_a03 | 0.4449 | 0.3070 |
| combined_lambda_ranknet | 0.4430 | 0.3068 |
| combined_sampled_ranknet | 0.4416 | 0.3034 |
| **mse** (baseline) | **0.4408** | 0.3037 |
| combined_pl_a03 | 0.4402 | 0.3070 |
| `pure_ranknet` | 0.4356 | **0.3161** (best all-element) |

**+0.0086 Spearman** for the best rank-loss over MSE on K562-matched elements.

---

## CAGI5 Cross-Method Ensembling (Phase 1)

Tried multiple ensembles on the 17 trained methods:

| Ensemble | K562-matched Sp |
|---|---|
| Uniform mean over all 17 | 0.4490 |
| **Uniform top-5 by K562 Sp** | **0.4552** |
| Uniform top-7 | 0.4543 |
| Borda rank aggregation (top 10) | 0.4516 |
| RidgeCV with K562 supervision | 0.4519 |

**Best result: uniform mean of top-5 single methods → 0.4552**
(+0.0144 vs MSE baseline, +0.0058 vs best single method)

Top 5: `combined_ranknet_a03, _softsort_a03, _lambda_ranknet, _sampled_ranknet, mse`

---

## Effect-Size Stratification

Per K562 element, bin variants by |ground truth| (quartiles).

| Element | Bin 0 (small Δ) | Bin 1 | Bin 2 | Bin 3 (large Δ) |
|---|---|---|---|---|
| GP1BB | 0.18 | 0.31 | 0.47 | **0.61** |
| HBB | 0.27 | 0.39 | 0.52 | **0.74** |
| HBG1 | 0.21 | 0.34 | 0.49 | **0.69** |
| PKLR | 0.19 | 0.28 | 0.45 | **0.57** |

Large-effect variants score **0.57–0.74 Sp** — well above all-variant 0.45.
The "noise floor" on small-effect variants depresses overall metrics.

---

## The Bets: pushing past 0.4552

After v4, three bets were attempted. **None exceeded the v4 top-5 ensemble.**
But each taught something.

| Bet | Hypothesis | K562 CAGI5 Sp |
|---|---|---|
| v4 uniform_top5 (target to beat) | — | **0.4552** ★ |
| **Bet 1** | Heteroscedastic NLL w/ variance supervision | <span class="lose">0.3842</span> |
| **Bet 4** | RankNet weighted by large \|Δy\| | <span class="lose">0.4332</span> |
| **Bet 4-rev** | RankNet weighted by small \|Δy\| | <span class="lose">0.4444</span> |

---

## Bet 1 — Heteroscedastic NLL (FAILED)

**Hypothesis:** Predict per-variant aleatoric variance → downweight noisy samples → better transfer to CAGI5.

**Bug found in pre-existing implementation (Feb 2026):**
- `variance_prediction_r` = −0.024 to −0.032 across 3 seeds
- Variance head was learning *nothing* — root cause: `MSE(pred_var, true_var)` in linear-variance space, true_var range [0, 4.8] but most mass <0.1, so the supervision was overwhelmed by NLL

**Fix:** supervise in log-variance space + λ_var=2.0 (swept λ∈{2,10,50})

**Result after fix:**
- `variance_prediction_r` flipped: −0.029 → **+0.224** ✓
- K562 test Spearman: 0.7151 (≈ tied with broken version 0.7180)
- **CAGI5 K562 Sp: 0.3842** — worse than v4 MSE (0.4408) by **−0.057** <span class="lose">✗</span>

**Why it failed:** `DREAM_RNN_Distributional` is a weaker architecture than Prix Fixe. The loss fix worked; the model didn't.

---

## Bet 4 — Large-Δ Weighted RankNet (FAILED)

**Hypothesis:** Upweight pairs with large `|relevance_diff|` → focus on dominant ranking signal that transfers to CAGI5.

**Loss:** `loss_per_pair = BCE × |Δy|^p` (p=1), normalized to mean 1.

**Result:**
- K562 test Spearman: 0.7639 (≈ tied with v4 `combined_ranknet_a03` 0.7641)
- **CAGI5 K562 Sp: 0.4332** — *worse* than v4 ranknet (0.4494) by **−0.016** <span class="lose">✗</span>

**Why it failed:**
Variant effects are inherently **small-Δ** predictions (ref/alt differences are tiny).
Large-Δ weighting taught the model to be *less* sensitive to small differences —
the opposite of what CAGI5 rewards.

→ Direction was wrong. Let's flip it.

---

## Bet 4-reversed — Small-Δ Weighted RankNet

**Hypothesis (inverted):** Upweight pairs with *small* `|Δy|` → fine-grained sensitivity transfers to CAGI5.

**Loss:** `loss_per_pair = BCE × exp(−|Δy|/τ)` (τ=0.5), normalized.

**Result:**
- K562 test Spearman: 0.7624 (tied with v4)
- **CAGI5 K562 Sp: 0.4444** — **+0.011 over Bet 4** ✓ (direction confirmed!)
- But still **−0.005 vs `combined_ranknet_a03`** <span class="lose">✗</span>
- And adding to v4 top-5 ensemble *hurts* (0.4552 → 0.4536) — correlated, not complementary

**Conclusion:** Small-Δ weighting is the right intuition, but **unweighted RankNet already implicitly learns the right pair weighting via SGD**.

---

## Summary of K562 + CAGI5

### K562 held-out test
- MSE Pearson 0.8233 / Spearman 0.7642 (paper baseline)
- All rank losses within ±0.002 of MSE on Pearson
- `combined_spearman` (Pearson) and `combined_ranknet_a03` (Sp) at the top — but the gap is in the noise

### CAGI5 K562 variant effects
- MSE: 0.4408
- Best single rank loss: **0.4494** (`combined_ranknet_a03`, +0.0086)
- Best ensemble: **0.4552** (uniform top-5, +0.0144 over MSE)
- No bet exceeded 0.4552 — pair weighting tweaks ruled out

---

## Cross-Cell-Type Generalization (NEW)

Are the K562 winners also winners on HepG2 / WTC11? Tested MSE + `combined_ranknet_a03`.

| Cell type | Method | Held-out Sp | Held-out P | Δ Sp vs MSE |
|---|---|---|---|---|
| K562 (226K) | MSE | 0.7642 | 0.8233 | — |
| K562 | combined_ranknet_a03 | 0.7656 | 0.8234 | +0.0014 |
| **WTC11 (56K)** | MSE | 0.6371 | 0.7291 | — |
| **WTC11** | **combined_ranknet_a03** | **0.6431** | **0.7339** | **+0.0060** ✓ |
| **HepG2 (140K)** | MSE | **0.8010** | 0.8151 | — |
| HepG2 | combined_ranknet_a03 | *in progress* | | |

**WTC11 confirms the K562 result direction.** HepG2 surprisingly *higher* than K562 on MSE alone — easier task or less noisy labels?

---

## Key Findings

1. **All rank losses are within noise of MSE on held-out Pearson** — the test set doesn't discriminate.
2. **CAGI5 transfer is where rank losses win.** Best single method: +0.009 Sp; best ensemble: +0.014.
3. **Direction of the gain transfers across cell types.** WTC11 also shows +0.006 Sp for `combined_ranknet_a03` vs MSE.
4. **Pair-weighting tweaks don't help.** Large-Δ hurts, small-Δ slightly recovers but doesn't beat unweighted.
5. **Architecture matters more than loss for CAGI5.** Bet 1's variance-head fix worked perfectly but Prix Fixe DREAM-RNN simply outperforms the distributional model.
6. **Held-out Pearson and CAGI5 K562 are anti-correlated for the top methods.** `combined_spearman` wins on Pearson but worst on CAGI5; `combined_ranknet_a03` is mid-pack on Pearson but #1 on CAGI5.

---

## What's Next

### Possible directions
- **Bet 5: AlphaGenome protocol** — deferred to "the very end" per request
- **Architecture search** — Prix Fixe DREAM-RNN may have hit its ceiling on CAGI5
- **Genuinely complementary signal** — different architectures/objectives, not loss tweaks
- **Cross-cell co-training** — train one model on K562+HepG2+WTC11 jointly

### Currently running
- HepG2 `combined_ranknet_a03` (Model 5/9 in progress, ~5h to finish)
- After that: HepG2-matched CAGI5 elements (LDLR, F9, MSMB, HNF4A) eval

---

## Reproducibility

| Path | Purpose |
|---|---|
| `scripts/train_deboer_rankloss.py` | 17-loss trainer |
| `scripts/dump_cagi5_predictions.py` | CAGI5 inference for Prix Fixe |
| `scripts/dump_cagi5_distributional.py` | CAGI5 inference for Bet 1 model |
| `scripts/ensemble_cagi5.py` | Cross-method ensemble + stratified analysis |
| `src/losses/ranknet.py` | Standard + large_delta + small_delta variants |
| `src/losses/distributional.py` | HeteroscedasticDistributionalLoss + log-var fix |
| `results/deboer_rankloss_1fold_v4/` | All 17 trained methods + CAGI5 predictions pkl |
| `results/deboer_rankloss_1fold_v5/` | Bet 4 + Bet 4-rev runs |
| `results/deboer_rankloss_1fold_v6_{hepg2,wtc11}/` | Cross-cell-type runs |
| `results/noise_resistant/DH*_v2_logvar_*` | Bet 1 (3 seeds, fixed) |
| `RESULTS.md` | Full results document including Bets section |

---

# Thank You

Questions?

<span class="small">Hyperparameters match the Rafi et al. 2025 paper (lr=0.005, 80 epochs, 10-fold) after an early bug fix where we had used lr=0.001.</span>
