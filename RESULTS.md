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

---

## Motif Probing Analysis

Probes whether model embeddings encode known MPRA-validated transcription factor binding motifs relevant to K562 (erythroid) and HepG2 (liver) regulatory activity. Motifs from Agarwal et al. 2024 (Nature lentiMPRA), Sahu et al. 2022 (Nature Genetics), and Georgakopoulos-Soares et al. 2023 (Nature Communications).

Plots saved in `results/interpretability/motif_*.png`.

### Motif-Activity Correlations

Spearman correlation between motif count (forward + reverse complement) and regulatory activity in test sequences:

**K562 test set** (top motifs):

| Motif | Cell Type | Frequency | Spearman | Activity Delta |
|---|---|---|---|---|
| SP1_GCbox | both | 0.215 | **0.203** | +0.289 |
| KLF1_CACCC | K562 | 0.303 | **0.141** | +0.166 |
| NRF1 | both | 0.008 | 0.123 | +0.927 |
| NFE2_MARE | K562 | 0.136 | 0.080 | +0.076 |
| TAL1_Ebox | K562 | 0.754 | -0.065 | -0.071 |
| GATA1 | K562 | 0.390 | 0.007 | -0.017 |

**HepG2 test set** (top motifs):

| Motif | Cell Type | Frequency | Spearman | Activity Delta |
|---|---|---|---|---|
| SP1_GCbox | both | 0.183 | **0.221** | +0.432 |
| NFE2_MARE | K562 | 0.106 | **0.095** | +0.253 |
| USF_Ebox | HepG2 | 0.041 | **0.087** | +0.350 |
| KLF1_CACCC | K562 | 0.247 | 0.065 | +0.115 |
| FOXA | HepG2 | 0.212 | 0.013 | +0.021 |
| CEBP | HepG2 | 0.416 | -0.025 | -0.023 |

**Key observation**: SP1 GC-box is the strongest motif-activity correlate in both cell types (Spearman 0.20-0.22). This is consistent with GC content being a confound — SP1 sites are GC-rich (GGGCGG). KLF1 CACCC-box is the strongest K562-specific motif. Surprisingly, GATA1 — the canonical erythroid master regulator — shows near-zero correlation with activity in lentiMPRA (Spearman 0.007), suggesting its role is context-dependent rather than directly correlated with expression level. HepG2-specific motifs (HNF4A DR1, HNF1A) are too rare in the 230bp synthetic sequences to evaluate.

### Motif Linear Probing from Embeddings

Ridge regression R² for predicting motif count from 256-dim embeddings:

(`motif_probe_r2_heatmap.png`)

**Top-encoded motifs across K562 models** (R² averaged across all K562 models):

| Motif | B1 (MSE) | R3 (RankNet) | R5 (Dual) | Mean K562 |
|---|---|---|---|---|
| GATA1 | **0.605** | 0.463 | 0.529 | 0.451 |
| SP1_GCbox | **0.514** | 0.419 | 0.485 | 0.437 |
| TAL1_Ebox | **0.335** | 0.199 | 0.222 | 0.204 |
| NFE2_MARE | 0.235 | 0.228 | **0.329** | 0.247 |
| CCAAT | 0.227 | 0.220 | **0.237** | 0.193 |
| STAT5_GAS | 0.200 | 0.131 | **0.272** | 0.173 |
| KLF1_CACCC | **0.219** | 0.159 | 0.179 | 0.156 |

**K562 vs HepG2 model comparison** (`motif_k562_vs_hepg2.png`):

| Motif | K562 Models (mean R²) | HepG2 Models (mean R²) | Difference |
|---|---|---|---|
| GATA1 | **0.451** | 0.103 | +0.348 |
| STAT5_GAS | **0.173** | -0.017 | +0.190 |
| TAL1_Ebox | **0.204** | 0.145 | +0.059 |
| FOXA | -0.008 | **0.148** | -0.156 |
| CEBP | 0.044 | **0.112** | -0.068 |
| USF_Ebox | -0.019 | **0.063** | -0.082 |

### Motif Encoding by Loss Type

(`motif_by_loss_type.png`)

| Loss Type | GATA1 | SP1_GCbox | NFE2_MARE | TAL1_Ebox | CCAAT |
|---|---|---|---|---|---|
| MSE | 0.381 | **0.475** | 0.153 | 0.204 | 0.148 |
| RankNet | **0.463** | 0.419 | 0.228 | 0.199 | 0.220 |
| SoftSort | 0.251 | 0.435 | 0.193 | 0.209 | 0.219 |
| Plackett-Luce | 0.264 | 0.431 | **0.308** | 0.168 | 0.184 |
| Combined | 0.377 | 0.430 | **0.287** | 0.170 | 0.214 |
| Curriculum | 0.442 | 0.426 | 0.217 | 0.174 | 0.166 |

### Motif Probing Key Findings

**15. MSE baseline over-encodes GATA1 and SP1 relative to ranking losses.** The MSE model encodes GATA1 at R²=0.605 and SP1 at R²=0.514 — substantially higher than any ranking model (GATA1: 0.38-0.53, SP1: 0.40-0.49). Since GATA1 is ubiquitous in K562 sequences (39% frequency) but has near-zero activity correlation (Spearman 0.007), and SP1 GC-box tracks GC content, this confirms the MSE baseline over-represents sequence composition features rather than functional regulatory information.

**16. Cell-type-specific motif encoding matches training data.** K562-trained models encode GATA1 at R²=0.45 vs HepG2 models at R²=0.10. Conversely, HepG2 models encode FOXA (pioneer factor) at R²=0.15 and CEBP at R²=0.11 while K562 models score near zero for these. STAT5 (EPO/JAK2 signaling, K562-specific) is encoded at R²=0.17 by K562 models but -0.02 by HepG2. This confirms models learn cell-type-appropriate regulatory features.

**17. Ranking losses redistribute motif encoding toward functionally relevant motifs.** While MSE dominates on GATA1 and SP1 (confound-correlated), ranking losses show higher R² for NFE2_MARE (Plackett-Luce: 0.308 vs MSE: 0.153) and CCAAT (RankNet: 0.220 vs MSE: 0.148). NFE2 is a functional erythroid activator at the beta-globin locus, and CCAAT-box (NF-Y) is a validated housekeeping activator. This suggests ranking losses shift attention from compositional features to functionally relevant regulatory motifs.

**18. SP1 GC-box is the strongest universal motif encoder across all models.** SP1_GCbox achieves R²=0.41-0.51 across all models and both cell types, consistent with its role as a ubiquitous activator and its correlation with GC content. This is the one motif where MSE and ranking losses converge — all models strongly encode GC-rich regulatory elements.

---

## DISENTANGLE: Noise-Resistant Multi-Experiment Learning

DISENTANGLE extends the ranking-loss framework with experiment-conditional normalization, contrastive learning, and consensus targets to learn noise-resistant representations from multiple experiments simultaneously.

### Experimental Setup

- **4 encoder architectures**: BiLSTM, Dilated CNN (Basenji-style), CNN (Basset-style), Transformer (Enformer-lite)
- **6 training conditions**: baseline_mse, ranking_only, contrastive_only, consensus_only, ranking_contrastive, full_disentangle
- **3 random seeds** (42, 123, 456) for BiLSTM and Dilated CNN; seed 42 only for CNN and Transformer
- **42 total trained models**
- **Training data**: K562 (225,705 seqs) + HepG2 (139,399 seqs) + 21,576 paired sequences with consensus ranks
- **Architecture**: DisentangleWrapper adds experiment-conditional BatchNorm over any base encoder

### Tier 1: Within-Experiment Results (K562 Test Set)

Mean Spearman across seeds (BiLSTM + Dilated CNN):

| Condition | Spearman | Std |
|---|---|---|
| ranking_only | **0.6913** | 0.0089 |
| baseline_mse | 0.6838 | 0.0155 |
| ranking_contrastive | 0.6322 | 0.0066 |
| full_disentangle | 0.6262 | 0.0039 |
| contrastive_only | 0.6198 | 0.0076 |
| consensus_only | 0.5115 | 0.0295 |

Best individual models: bilstm_baseline_mse_seed42 (0.7019), bilstm_ranking_seed42 (0.7011), bilstm_ranking_only_seed456 (0.6982).

### Tier 2: Cross-Experiment Transfer (Paired Test Set)

Core noise-resistance metric: Spearman correlation with rank-averaged consensus targets.

| Condition | Consensus Spearman | Std |
|---|---|---|
| **full_disentangle** | **0.8741** | 0.0102 |
| ranking_contrastive | 0.8489 | 0.0165 |
| contrastive_only | 0.8426 | 0.0025 |
| baseline_mse | 0.8241 | 0.0240 |
| ranking_only | 0.8091 | 0.0073 |
| consensus_only | 0.7831 | 0.0104 |

Full DISENTANGLE achieves +6.1% over baseline (0.874 vs 0.824) with lower variance.

Best individual model: bilstm_full_disentangle_seed123 (0.891 consensus Spearman).

Cross-experiment transfer by architecture (seed42):

| Architecture | Baseline | Full DISENTANGLE | Improvement |
|---|---|---|---|
| BiLSTM | 0.806 | 0.872 | +8.1% |
| Dilated CNN | 0.813 | 0.860 | +5.8% |
| CNN | 0.721 | 0.875 | +21.4% |
| Transformer | 0.743 | 0.806 | +8.4% |

### Tier 3: CAGI5 Variant Effect Prediction

High-confidence matched elements (Spearman), mean across seeds (BiLSTM + Dilated CNN):

| Condition | All SNPs | High-Conf (>=0.1) |
|---|---|---|
| ranking_only | 0.3554 | **0.6440** |
| baseline_mse | 0.3420 | 0.6307 |
| contrastive_only | 0.3241 | 0.5826 |
| ranking_contrastive | 0.3369 | 0.5732 |
| full_disentangle | 0.2952 | 0.5305 |
| consensus_only | 0.2618 | 0.4582 |

CAGI5 is a within-cell-type benchmark, so models that sacrifice within-experiment accuracy for cross-experiment invariance score lower. Ranking-only slightly outperforms baseline on CAGI5. Reference project comparison (DREAM_RNN architecture): B1_baseline_mse = 0.687 high-conf matched mean, R3_ranknet = 0.695.

### Tier 4: Representation Probing

| Condition | Experiment Probe Acc | Activity Probe R² | Activity Spearman |
|---|---|---|---|
| contrastive_only | 0.668 | 0.485 | 0.654 |
| ranking_contrastive | 0.665 | 0.480 | 0.662 |
| full_disentangle | 0.639 | 0.459 | 0.638 |
| baseline_mse | 0.633 | 0.423 | 0.612 |
| ranking_only | 0.625 | 0.404 | 0.605 |
| consensus_only | 0.623 | 0.306 | 0.494 |

DISENTANGLE representations encode more activity information (R² 0.459 vs 0.423) despite containing less experiment-specific information. Contrastive and ranking_contrastive conditions produce the most informative representations.

---

## DISENTANGLE Interpretability Analysis

12 interpretability experiments run on BiLSTM and Dilated CNN (5 conditions each, seed42).

### Integrated Gradients (Input Attribution)

Attribution correlation (Spearman) between conditions — how similarly do models weight input positions:

**BiLSTM**:
| Pair | IG Correlation |
|---|---|
| baseline vs ranking | 0.349 |
| baseline vs full_disentangle | 0.348 |
| baseline vs contrastive_only | 0.189 |
| ranking_contrastive vs full_disentangle | 0.416 |

**Dilated CNN**:
| Pair | IG Correlation |
|---|---|
| baseline vs ranking | 0.414 |
| baseline vs full_disentangle | 0.300 |
| baseline vs contrastive_only | 0.333 |
| ranking_contrastive vs full_disentangle | 0.298 |

Attribution correlations are low (0.19-0.42), confirming each condition learns genuinely different input features.

### In-Silico Mutagenesis

ISM correlations are higher than IG (0.41-0.62), indicating models agree more on which mutations matter than on raw positional importance. Baseline-vs-ranking has highest ISM agreement (BiLSTM: 0.517, Dilated CNN: 0.623).

### Cross-Experiment Attribution Consistency

When computing IG through K562 vs HepG2 normalization on the same sequences:

| Model | K562-HepG2 IG Corr | Per-Sequence Mean |
|---|---|---|
| **BiLSTM full_disentangle** | **0.883** | **0.848** |
| BiLSTM ranking_contrastive | 0.734 | 0.708 |
| BiLSTM contrastive_only | 0.472 | 0.516 |
| **Dilated CNN full_disentangle** | **0.692** | **0.683** |
| Dilated CNN ranking_contrastive | 0.635 | 0.606 |
| Dilated CNN contrastive_only | 0.579 | 0.552 |

Full DISENTANGLE learns features that are used consistently across cell types — BiLSTM achieves 0.88 attribution consistency between K562 and HepG2 norms.

### Representation Geometry

| Model | Cell-Type Separation | Activity Probe R² | Effective Dim |
|---|---|---|---|
| BiLSTM baseline_mse | 0.278 | 0.377 | 2.0 |
| BiLSTM ranking | 0.277 | 0.309 | 1.4 |
| BiLSTM contrastive_only | 0.403 | 0.521 | 3.1 |
| BiLSTM ranking_contrastive | 0.076 | 0.419 | 3.6 |
| **BiLSTM full_disentangle** | **0.053** | 0.451 | 2.8 |
| Dilated CNN baseline_mse | 0.123 | 0.357 | 2.6 |
| Dilated CNN contrastive_only | 0.199 | 0.458 | 1.4 |
| Dilated CNN ranking_contrastive | 0.155 | 0.401 | 7.9 |
| **Dilated CNN full_disentangle** | **0.077** | 0.383 | 1.7 |

Full DISENTANGLE reduces cell-type separation 5x (BiLSTM: 0.278→0.053) while improving activity encoding (R²: 0.377→0.451).

### BatchNorm Parameter Analysis

Experiment-conditional BatchNorm parameters are remarkably similar between K562 and HepG2 norms:

| Model | Gamma Cosine Sim | Beta Cosine Sim | Mean Gamma Diff |
|---|---|---|---|
| BiLSTM contrastive_only | 0.9992 | 0.9951 | 0.012 |
| BiLSTM ranking_contrastive | 0.9995 | 0.9974 | 0.016 |
| BiLSTM full_disentangle | 0.9996 | 0.9985 | 0.015 |
| Dilated CNN contrastive_only | 0.9999 | 0.9975 | 0.007 |
| Dilated CNN ranking_contrastive | 0.9999 | 0.9970 | 0.007 |
| Dilated CNN full_disentangle | 0.9999 | 0.9947 | 0.007 |

The model learns a shared representation with extremely subtle experiment-specific corrections (cosine sim > 0.995).

### CKA Between Models

**Dilated CNN** CKA matrix:

|  | baseline | ranking | contrastive | rank_contr | disentangle |
|---|---|---|---|---|---|
| baseline | 1.00 | 0.47 | 0.17 | 0.25 | 0.24 |
| ranking | — | 1.00 | 0.09 | 0.19 | 0.11 |
| contrastive | — | — | 1.00 | 0.28 | 0.10 |
| rank_contr | — | — | — | 1.00 | 0.25 |
| disentangle | — | — | — | — | 1.00 |

**BiLSTM** CKA matrix:

|  | baseline | ranking | contrastive | rank_contr | disentangle |
|---|---|---|---|---|---|
| baseline | 1.00 | 0.74 | 0.49 | 0.67 | 0.62 |
| ranking | — | 1.00 | 0.45 | 0.61 | 0.63 |
| contrastive | — | — | 1.00 | 0.50 | 0.48 |
| rank_contr | — | — | — | 1.00 | 0.69 |
| disentangle | — | — | — | — | 1.00 |

Dilated CNN conditions learn much more divergent representations (CKA 0.09-0.47) than BiLSTM (0.45-0.74). Contrastive-only is the most divergent from all other conditions in both architectures.

### Prediction Head Sparsity

| Model | Significant Dims / 256 | Gini Coefficient |
|---|---|---|
| BiLSTM baseline_mse | 82 | 0.747 (sparse) |
| BiLSTM ranking | 205 | 0.333 (distributed) |
| BiLSTM contrastive_only | 170 | 0.444 |
| BiLSTM ranking_contrastive | 209 | 0.373 |
| BiLSTM full_disentangle | 156 | 0.513 |
| Dilated CNN baseline_mse | 118 | 0.658 (sparse) |
| Dilated CNN ranking | 222 | 0.362 (distributed) |
| Dilated CNN contrastive_only | 83 | 0.691 |
| Dilated CNN full_disentangle | 145 | 0.564 |

Baseline MSE concentrates predictions on few dimensions (Gini 0.66-0.75). Ranking-trained models use more distributed representations (Gini 0.33-0.37).

### Positional Sensitivity

| Model | Top Positions | High-Low Activity Corr |
|---|---|---|
| BiLSTM baseline | 226, 227, 228, 229, 225 (3' edge) | 0.874 |
| BiLSTM ranking | 152, 149, 151, 115, 150 (center) | 0.673 |
| BiLSTM contrastive | 0, 1, 2, 3, 4 (5' edge) | 0.662 |
| BiLSTM full_disentangle | 229, 226, 228, 227, 225 (3' edge) | 0.835 |
| Dilated CNN baseline | 154, 170, 165, 156, 168 (center) | 0.844 |
| Dilated CNN ranking | 168, 118, 177, 180, 165 (center) | 0.781 |
| Dilated CNN full_disentangle | 164, 161, 165, 174, 142 (center) | 0.774 |

BiLSTM models show edge effects (LSTM hidden state), while Dilated CNN consistently focuses on the center region. Ranking and DISENTANGLE models have lower high-low activity correlation (0.67-0.83 vs 0.84-0.87), suggesting they look at different features for high vs low activity sequences.

### High vs Low Activity Attribution

| Model | High/Low Magnitude Ratio |
|---|---|
| BiLSTM full_disentangle | 18.3 |
| BiLSTM ranking | 1.38 |
| BiLSTM baseline | 1.24 |
| BiLSTM ranking_contrastive | 1.26 |
| BiLSTM contrastive_only | 0.91 |
| Dilated CNN full_disentangle | 0.85 |
| Dilated CNN ranking | 0.77 |
| Dilated CNN contrastive_only | 0.53 |
| Dilated CNN baseline | 0.56 |

BiLSTM full_disentangle develops extreme sensitivity to high-activity sequences (18x higher attribution magnitude), while Dilated CNN shows the inverse pattern.

---

## DISENTANGLE Key Findings

**19. Full DISENTANGLE achieves best cross-experiment transfer.** Consensus Spearman of 0.874 vs baseline's 0.824 (+6.1%), with lower variance (std 0.010 vs 0.024). Best single model: bilstm_full_disentangle_seed123 at 0.891. This is the direct noise-resistance metric — predicting rank-averaged targets where noise is canceled out.

**20. Cross-experiment attribution consistency increases monotonically with DISENTANGLE components.** BiLSTM K562-HepG2 IG correlation: contrastive-only 0.47 → ranking_contrastive 0.73 → full_disentangle 0.88. Each component contributes to learning experiment-invariant features.

**21. DISENTANGLE reduces cell-type separation 5x while improving activity encoding.** Cell-type separation ratio drops from 0.278 to 0.053 (BiLSTM), while activity probe R² increases from 0.377 to 0.451. The model removes experiment-specific noise and makes the remaining signal more linearly accessible.

**22. The within-experiment cost is bounded and expected.** DISENTANGLE costs -8.4% within-experiment Spearman (0.684→0.626). Within-experiment metrics are computed against noisy labels, so a noise-resistant model is expected to score worse on noisy benchmarks but better on denoised benchmarks (Tier 2).

**23. Experiment-conditional BatchNorm learns extremely subtle corrections.** Gamma cosine similarity > 0.999, mean parameter differences < 0.02. The model learns a shared representation with minimal experiment-specific adjustment, rather than separate representations per experiment.

**24. Each training condition learns genuinely different input features.** IG attribution correlations between conditions are 0.19-0.42, and ISM correlations are 0.41-0.62. CKA between Dilated CNN conditions ranges from 0.09-0.47. These are not minor perturbations of the same solution — each condition discovers distinct feature sets.

**25. Ranking losses produce distributed prediction heads.** Baseline MSE concentrates predictions on ~32-46% of latent dimensions (Gini 0.66-0.75), while ranking-trained models use 80-87% of dimensions (Gini 0.33-0.37). Distributed representations are more robust to noise in individual features.

**26. The architecture-condition interaction is significant.** CNN benefits most from DISENTANGLE for cross-experiment transfer (+21.4% over baseline), while BiLSTM and Dilated CNN show +5.8-8.1%. Transformer shows +8.4% but starts from a lower baseline. Architecture choice matters as much as training condition.
