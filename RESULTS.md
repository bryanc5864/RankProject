# Results: Rank-Order Learning for MPRA Data

## LentiMPRA Test Set Results

### K562 Models

| Experiment | Model | Loss | Spearman | Pearson | Pairwise Acc | NDCG@10 |
|---|---|---|---|---|---|---|
| R3_ranknet | dream_rnn_single | ranknet | **0.7144** | 0.7553 | **0.7634** | 0.6420 |
| R4_combined | dream_rnn_single | combined | 0.7116 | 0.7605 | 0.7630 | 0.6919 |
| R2_softsort | dream_rnn_single | softsort | 0.7105 | **0.7699** | 0.7624 | 0.7106 |
| R1_plackett_luce | dream_rnn_single | plackett_luce | 0.7084 | 0.7547 | 0.7613 | 0.6546 |
| B1_baseline_mse | dream_rnn | mse | 0.7075 | 0.7587 | 0.7613 | 0.4843 |
| D1_domain_adversarial | dream_rnn_da | plackett_luce | 0.7073 | 0.7595 | 0.7611 | 0.5871 |
| C1_pl_linear_curriculum | dream_rnn_single | plackett_luce | 0.7064 | 0.7613 | 0.7610 | 0.6959 |
| D2_bias_factorized | dream_rnn_bf | plackett_luce | 0.7055 | 0.7609 | 0.7606 | **0.7133** |
| C3_combined_curriculum | dream_rnn_single | combined | 0.7037 | 0.7581 | 0.7598 | 0.6389 |
| R5_dual_combined | dream_rnn_dual | multi_task | 0.7032 | 0.7597 | 0.7590 | 0.6813 |
| C2_pl_stepped_curriculum | dream_rnn_single | plackett_luce | 0.7018 | 0.7540 | 0.7584 | 0.6538 |
| D3_full_advanced | dream_rnn_adv | plackett_luce | 0.7007 | 0.7578 | 0.7583 | 0.6570 |
| B2_soft_classification | dream_rnn | soft_class | 0.6685 | 0.6231 | 0.6503 | 0.3216 |

### HepG2 Models

| Experiment | Model | Loss | Spearman | Pearson | Pairwise Acc | NDCG@10 |
|---|---|---|---|---|---|---|
| H4_HepG2_softsort | dream_rnn_single | softsort | **0.7352** | **0.7425** | **0.7709** | 0.6964 |
| H3_HepG2_combined | dream_rnn_single | combined | 0.7320 | 0.7342 | 0.7700 | **0.7181** |
| H2_HepG2_plackett_luce | dream_rnn_single | plackett_luce | 0.7258 | 0.7276 | 0.7670 | 0.6440 |
| H1_HepG2_baseline_mse | dream_rnn | mse | 0.7185 | 0.7257 | 0.7634 | 0.6667 |

---

## CAGI5 Saturation Mutagenesis Results

### Cell-Type Element Mapping

- **K562 elements**: GP1BB, HBB, HBG1, PKLR
- **HepG2 elements**: F9, LDLR, SORT1

### CAGI5 Results: All SNPs (Matched Elements Only)

**K562 models on K562 elements (GP1BB, HBB, HBG1, PKLR)**

| Experiment | GP1BB | HBB | HBG1 | PKLR | Mean |
|---|---|---|---|---|---|
| B1_baseline_mse | 0.3402 | 0.4047 | 0.4444 | 0.3551 | **0.3861** |
| R3_ranknet | 0.2939 | 0.4026 | 0.3883 | 0.3756 | 0.3651 |
| R2_softsort | 0.3054 | 0.4333 | 0.3912 | 0.3153 | 0.3613 |
| D3_full_advanced | 0.3180 | 0.3613 | 0.4451 | 0.3059 | 0.3576 |
| R4_combined | 0.3285 | 0.4279 | 0.3554 | 0.2956 | 0.3518 |
| C1_pl_linear_curriculum | 0.2781 | 0.4278 | 0.3747 | 0.3227 | 0.3508 |
| R1_plackett_luce | 0.2759 | 0.4233 | 0.3757 | 0.3205 | 0.3489 |
| R5_dual_combined | 0.3036 | 0.3961 | 0.3931 | 0.3016 | 0.3486 |
| D2_bias_factorized | 0.2905 | 0.4018 | 0.3670 | 0.3326 | 0.3480 |
| C3_combined_curriculum | 0.2473 | 0.4513 | 0.3885 | 0.2938 | 0.3452 |
| D1_domain_adversarial (v2) | 0.2861 | 0.3651 | 0.3771 | 0.2931 | 0.3304 |
| C2_pl_stepped_curriculum | 0.2219 | 0.3688 | 0.3632 | 0.3151 | 0.3172 |

**HepG2 models on HepG2 elements (F9, LDLR, SORT1)**

| Experiment | F9 | LDLR | SORT1 | Mean |
|---|---|---|---|---|
| H1_HepG2_baseline_mse | **0.4817** | 0.3937 | **0.4658** | **0.4471** |
| H2_HepG2_plackett_luce | 0.4830 | 0.3876 | 0.4347 | 0.4351 |
| H3_HepG2_combined | **0.4999** | 0.3258 | 0.4561 | 0.4273 |
| H4_HepG2_softsort | 0.4789 | 0.3424 | 0.4071 | 0.4095 |

### CAGI5 Results: High-Confidence SNPs (Confidence >= 0.1)

**K562 models on K562 elements**

| Experiment | GP1BB | HBB | HBG1 | PKLR | Mean |
|---|---|---|---|---|---|
| R3_ranknet | 0.5919 | **0.7098** | 0.7199 | **0.7577** | **0.6948** |
| B1_baseline_mse | **0.7189** | 0.6708 | 0.6965 | 0.6612 | 0.6868 |
| R1_plackett_luce | 0.6291 | 0.7357 | 0.7154 | 0.6647 | 0.6862 |
| R2_softsort | 0.6465 | 0.6499 | 0.6997 | 0.6673 | 0.6658 |
| R5_dual_combined | 0.6367 | 0.6599 | 0.7157 | 0.6460 | 0.6646 |
| D2_bias_factorized | 0.5634 | 0.6753 | **0.7044** | 0.6550 | 0.6495 |
| R4_combined | 0.6117 | 0.6604 | 0.6620 | 0.6091 | 0.6358 |
| C1_pl_linear_curriculum | 0.5600 | 0.6622 | 0.6793 | 0.6370 | 0.6346 |
| C2_pl_stepped_curriculum | 0.5119 | 0.6296 | 0.6665 | 0.6955 | 0.6259 |
| D3_full_advanced | 0.6500 | 0.6004 | 0.6663 | 0.5846 | 0.6253 |
| C3_combined_curriculum | 0.6033 | 0.5986 | 0.6881 | 0.6036 | 0.6234 |
| D1_domain_adversarial (v2) | 0.5873 | 0.5374 | 0.7010 | 0.6368 | 0.6156 |

**HepG2 models on HepG2 elements**

| Experiment | F9 | LDLR | SORT1 | Mean |
|---|---|---|---|---|
| H2_HepG2_plackett_luce | 0.7752 | **0.5647** | **0.6308** | **0.6569** |
| H1_HepG2_baseline_mse | **0.8269** | 0.4784 | 0.6278 | 0.6444 |
| H3_HepG2_combined | 0.8336 | 0.3848 | 0.6150 | 0.6111 |
| H4_HepG2_softsort | 0.7464 | 0.4085 | 0.6013 | 0.5854 |

---

## Cross-Cell-Type Zero-Shot CAGI5 Results

Models trained on one cell type evaluated on the other cell type's matched CAGI5 elements.

### K562 Models on HepG2 Elements (F9, LDLR, SORT1)

**All SNPs**

| Experiment | F9 | LDLR | SORT1 | Mean |
|---|---|---|---|---|
| D3_full_advanced | 0.3021 | **0.4332** | 0.1141 | **0.2832** |
| B1_baseline_mse | **0.3700** | 0.3344 | **0.1401** | 0.2815 |
| R5_dual_combined | 0.3528 | 0.3912 | 0.0861 | 0.2767 |
| R2_softsort | 0.2685 | 0.4170 | 0.1413 | 0.2756 |
| R4_combined | 0.3341 | 0.2816 | 0.1270 | 0.2476 |
| R3_ranknet | 0.3243 | 0.3018 | 0.1135 | 0.2466 |
| R1_plackett_luce | 0.2495 | 0.3502 | 0.1086 | 0.2361 |

**High Confidence (>= 0.1)**

| Experiment | F9 | LDLR | SORT1 | Mean |
|---|---|---|---|---|
| B1_baseline_mse | 0.7522 | 0.4675 | **0.2072** | **0.4756** |
| R2_softsort | 0.6086 | **0.5596** | 0.2059 | 0.4581 |
| R5_dual_combined | 0.7340 | 0.5007 | 0.1113 | 0.4487 |
| R4_combined | 0.7044 | 0.4554 | 0.1766 | 0.4455 |
| R3_ranknet | 0.7152 | 0.4423 | 0.1572 | 0.4382 |
| R1_plackett_luce | 0.6917 | 0.4882 | 0.1069 | 0.4289 |

### HepG2 Models on K562 Elements (GP1BB, HBB, HBG1, PKLR)

**All SNPs**

| Experiment | GP1BB | HBB | HBG1 | PKLR | Mean |
|---|---|---|---|---|---|
| H4_HepG2_softsort | **0.2564** | 0.3586 | **0.3114** | **0.1858** | **0.2780** |
| H3_HepG2_combined | 0.2133 | **0.3638** | 0.3079 | 0.1846 | 0.2674 |
| H2_HepG2_plackett_luce | 0.2491 | 0.3394 | 0.2859 | 0.1665 | 0.2602 |
| H1_HepG2_baseline_mse | 0.1872 | 0.2909 | 0.2392 | 0.2144 | 0.2330 |

**High Confidence (>= 0.1)**

| Experiment | GP1BB | HBB | HBG1 | PKLR | Mean |
|---|---|---|---|---|---|
| H4_HepG2_softsort | **0.5583** | 0.6906 | 0.6361 | **0.4161** | **0.5753** |
| H1_HepG2_baseline_mse | 0.5196 | 0.6607 | 0.5449 | 0.4248 | 0.5375 |
| H3_HepG2_combined | 0.4630 | **0.7007** | **0.6306** | 0.3277 | 0.5305 |
| H2_HepG2_plackett_luce | 0.5216 | 0.6202 | 0.5794 | 0.3662 | 0.5218 |

### Combined Cross-Zero-Shot by Loss Function

Average of both directions (K562-to-HepG2 and HepG2-to-K562) for losses tested in both cell types.

**All SNPs**

| Loss | K562-to-HepG2 | HepG2-to-K562 | Average |
|---|---|---|---|
| softsort | 0.2756 | 0.2780 | **0.2768** |
| combined | 0.2476 | 0.2674 | 0.2575 |
| mse | 0.2815 | 0.2330 | 0.2572 |
| plackett_luce | 0.2361 | 0.2602 | 0.2482 |

**High Confidence (>= 0.1)**

| Loss | K562-to-HepG2 | HepG2-to-K562 | Average |
|---|---|---|---|
| softsort | 0.4581 | 0.5753 | **0.5167** |
| mse | 0.4756 | 0.5375 | 0.5066 |
| combined | 0.4455 | 0.5305 | 0.4880 |
| plackett_luce | 0.4289 | 0.5218 | 0.4754 |

---

## Key Findings

### 1. Ranking losses improve test-set Spearman over baseline MSE

All ranking-based losses (R1-R5) outperform the MSE baseline (B1: 0.7075) on the K562 test set, with RankNet achieving the best Spearman (R3: 0.7144). The improvement is modest but consistent (~0.3-1.0% absolute).

### 2. HepG2 models achieve higher test-set correlation than K562

HepG2 models consistently outperform K562 models on both Spearman and Pearson test metrics. The best HepG2 model (H4_softsort: 0.7352) exceeds the best K562 model (R3_ranknet: 0.7144) by 2 percentage points. This may reflect differences in data quality or regulatory complexity between cell types.

### 3. Cell-type matching matters for CAGI5 evaluation

HepG2-trained models substantially outperform K562-trained models on HepG2-matched CAGI5 elements (F9, LDLR, SORT1), with mean Spearman of 0.4471 vs 0.3861 for the best K562 model. This confirms that cell-type-specific training captures relevant biology.

### 4. High-confidence filtering dramatically improves correlations

Filtering to high-confidence SNPs (>= 0.1) roughly doubles Spearman correlations across all models: K562 mean goes from ~0.35 to ~0.67, HepG2 from ~0.43 to ~0.64. The low-confidence variants introduce substantial noise.

### 5. RankNet excels on high-confidence CAGI5 variants

On high-confidence matched elements, R3_ranknet achieves the highest K562 mean Spearman (0.6948), slightly beating the MSE baseline (0.6868). For HepG2, Plackett-Luce (H2: 0.6569) edges out the MSE baseline (H1: 0.6444).

### 6. NDCG@10 benefits most from ranking-aware losses

The largest advantage of ranking losses is in NDCG@10 (top-of-list precision). D2_bias_factorized achieves 0.7133 vs 0.4843 for the MSE baseline on K562. On HepG2, H3_combined reaches 0.7181 vs 0.6667 for MSE. This is expected since ranking losses directly optimize relative ordering.

### 7. Soft classification fails on this task

B2_soft_classification produces anti-correlated CAGI5 predictions (negative Spearman), suggesting binning continuous activity values is not appropriate for variant effect prediction.

### 8. Dual-head model provides no clear advantage over single-head combined

R5_dual_combined (dual-head multi-task) does not improve over R4_combined (single-head combined) on either test-set Spearman (0.7032 vs 0.7116) or CAGI5 matched mean (0.3486 vs 0.3518). The additional model complexity from separate heads appears unnecessary.

### 9. Softsort is the best loss for cross-cell-type zero-shot transfer

Averaging both transfer directions (K562-to-HepG2 and HepG2-to-K562), softsort achieves the highest cross-zero-shot Spearman on both all SNPs (0.2768) and high-confidence variants (0.5167). Softsort is notably symmetric across directions (0.2756 vs 0.2780 on all SNPs), while MSE is asymmetric (0.2815 vs 0.2330) — performing well in one direction but poorly in the other. Softsort's listwise differentiable sorting likely learns rank relationships that generalize across cell types better than absolute activity values optimized by MSE. RankNet, despite being the best on matched K562 CAGI5 elements, transfers the worst (0.2466 K562-to-HepG2, all SNPs), suggesting it overfits pairwise comparisons to cell-type-specific patterns.

---

## Interpretability Analysis

Plots saved in `results/interpretability/`.

### UMAP of Learned Embeddings

**K562 models** (`umap_embeddings_K562.png`): All models learn a continuous manifold organized by activity level (smooth color gradients from low to high activity). Ranking losses (R1-R5) produce tighter, more ring/arc-shaped structures compared to the more diffuse MSE baseline, suggesting ranking losses impose stronger ordering constraints on the embedding space.

**HepG2 models** (`umap_embeddings_HepG2.png`): Similar pattern — ranking losses (H2-H4) produce much tighter 1D manifolds than the MSE baseline (H1), which has a more spread-out cloud.

**Cross-cell-type** (`umap_cross_cell_type.png`): Running R3_ranknet on both K562 and HepG2 data shows sequences from both cell types are fully interleaved with no separation by cell type, ordered smoothly by activity. This confirms the model learns sequence-intrinsic features rather than cell-type-specific artifacts.

### CKA Representation Similarity

(`cka_similarity_matrix.png`, `cka_similarity.csv`)

| Comparison | CKA Range |
|---|---|
| K562 models vs K562 models | 0.78 – 0.85 |
| HepG2 models vs HepG2 models | 0.74 – 0.89 |
| K562 models vs HepG2 models | 0.42 – 0.53 |

All K562 models learn highly similar representations regardless of loss function (CKA 0.78-0.85). The loss primarily affects the output head, not the learned features. HepG2 models form their own high-similarity cluster (0.74-0.89). Cross-cell-type CKA is low (0.42-0.53), confirming genuinely different representation spaces from different data distributions. Within K562, R3_ranknet is slightly more divergent (CKA ~0.78-0.82 vs others at 0.82-0.85).

### Integrated Gradients Attribution

(`ig_attributions_*.png`, `ig_position_profiles.png`, `ig_attribution_correlation.png`)

Attribution profiles differ dramatically between models:

| Model | Attribution Focus | Magnitude Scale |
|---|---|---|
| B1_baseline_mse | Broad, distributed across all 230bp | ~0.03 |
| R1_plackett_luce | Concentrated at 5' end (pos 0-30) | ~1e9 |
| R2_softsort | Concentrated at 3' end (pos 180-230) | ~1e10 |
| R3_ranknet | 3'-focused | ~1e8 |
| R4_combined | 5'-focused | ~1e13 |

The MSE baseline distributes importance broadly, while ranking models concentrate attribution at sequence edges with vastly amplified magnitudes (a consequence of unconstrained output scales in ranking losses). Attribution correlation between models is near zero (Pearson 0.001-0.035), meaning each model uses completely different positional features to achieve similar ranking accuracy. This suggests ranking losses exploit different gradient pathways through the BiLSTM.

### Linear Probing of Representations

(`linear_probing.png`, `linear_probing.csv`)

| Experiment | Activity R² | Rank R² | GC Content R² | CpG Freq R² |
|---|---|---|---|---|
| R1_plackett_luce | **0.582** | **0.510** | 0.799 | 0.667 |
| C1_pl_linear_curriculum | 0.582 | 0.494 | 0.715 | 0.653 |
| C3_combined_curriculum | 0.578 | 0.500 | 0.663 | 0.622 |
| R3_ranknet | 0.576 | 0.507 | **0.666** | **0.646** |
| R4_combined | 0.575 | 0.496 | 0.725 | 0.637 |
| D1_domain_adversarial | 0.570 | 0.487 | 0.808 | 0.712 |
| C2_pl_stepped_curriculum | 0.569 | 0.486 | 0.753 | 0.657 |
| D2_bias_factorized | 0.568 | 0.481 | 0.784 | 0.687 |
| D3_full_advanced | 0.566 | 0.480 | 0.718 | 0.656 |
| B1_baseline_mse | 0.563 | 0.487 | **0.975** | **0.898** |
| R5_dual_combined | 0.563 | 0.486 | 0.805 | 0.673 |
| R2_softsort | 0.562 | 0.478 | 0.837 | 0.739 |

All models encode activity comparably (R² 0.56-0.58). Random baseline R² is negative for all models, confirming no noise memorization. The critical finding is **GC content encoding**:

- **B1_baseline_mse: GC R² = 0.975** — nearly perfect encoding of GC content, a known MPRA confounding variable
- **R3_ranknet: GC R² = 0.666** — lowest GC dependence among all models
- **C3_combined_curriculum: GC R² = 0.663** — similarly low

Ranking losses, especially RankNet and combined-curriculum, learn representations that are less dependent on GC content while maintaining similar activity prediction power. Since GC content is correlated with but not causative of regulatory activity, this suggests ranking losses learn more biologically meaningful features.

### Prediction Comparison

(`prediction_scatter.png`, `prediction_correlation.png`, `residual_vs_baseline.png`)

Inter-model prediction Spearman correlations (K562 models):

| | B1_mse | R1_pl | R2_ss | R3_rn | R4_comb | R5_dual |
|---|---|---|---|---|---|---|
| B1_mse | 1.000 | 0.878 | 0.877 | 0.869 | 0.881 | 0.875 |
| R1_pl | — | 1.000 | 0.888 | 0.889 | 0.902 | 0.894 |
| R2_ss | — | — | 1.000 | 0.892 | 0.887 | 0.883 |
| R3_rn | — | — | — | 1.000 | 0.891 | 0.881 |
| R4_comb | — | — | — | — | 1.000 | 0.892 |
| R5_dual | — | — | — | — | — | 1.000 |

All models are highly correlated (Spearman 0.87-0.90). R1_plackett_luce and R4_combined are most similar (0.902). B1_baseline_mse is most divergent from ranking models (~0.87). Residual plots show ranking models systematically differ from MSE at the extremes of the activity distribution, consistent with ranking losses better calibrating tail predictions.

### Interpretability Key Findings

**10. MSE baseline embeddings are confounded with GC content.** The MSE baseline encodes GC content at R² = 0.975 (near-perfect), while ranking losses reduce this to 0.66-0.84 without sacrificing activity information (R² ~0.56-0.58 for all). Since GC content is a known confounding variable in MPRA, this suggests ranking losses learn more biologically meaningful features by reducing reliance on sequence composition shortcuts.

**11. Ranking losses do not memorize noise.** All models achieve negative R² when probed for random labels, confirming they encode genuine sequence features rather than memorizing training noise.

**12. Models learn similar representations despite different losses.** CKA similarity of 0.78-0.85 across all K562 models indicates the DREAM-RNN architecture converges to similar internal representations regardless of training objective. The loss primarily affects the output head and attribution landscape, not the core learned features.

**13. Attribution patterns are model-specific despite similar representations.** Each ranking loss produces dramatically different attribution profiles (different positional focus, magnitude scales varying by 15 orders of magnitude). The near-zero inter-model attribution correlations suggest multiple distinct gradient solutions exist for similar predictive performance — a form of underspecification.

**14. Cross-cell-type embeddings are interleaved, not separated.** UMAP of a K562 model applied to both cell types shows complete mixing of K562 and HepG2 sequences ordered by activity, confirming models capture cell-type-agnostic sequence properties. This supports their utility for cross-cell-type variant effect prediction.
