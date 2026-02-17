# Presentation Script: Noise-Resistant Training for Sequence-to-Expression Prediction

**Total Duration:** ~25-30 minutes (15 slides, ~2 minutes each)

---

## Slide 1: Title

*[No narration needed - title slide]*

**Key Points to Mention:**
- 74 models systematically evaluated
- 7 distinct noise-resistant training strategies
- Evaluated on CAGI5 saturation mutagenesis benchmark
- Focus on K562-matched regulatory elements

---

## Slide 2: Methods Overview

Good morning/afternoon. Today I'll present our systematic evaluation of noise-resistant training strategies for sequence-to-expression prediction models.

The core problem we're addressing is that lentiMPRA measurements—which we use to train these models—contain inherent experimental noise. This noise, which we call aleatoric uncertainty, varies substantially across sequences. Some sequences have highly reproducible measurements across replicates, while others show significant variance. Standard mean squared error training treats all samples equally, which means the model may overfit to noisy measurements and hurt generalization to clinical variants.

Our approach leverages the aleatoric uncertainty information directly available in the lentiMPRA data. The HDF5 files contain both activity measurements and replicate variance for each sequence. We developed seven complementary strategies to handle this noise.

The first category—Rank Stability—down-weights pairwise comparisons involving high-noise samples. Distributional methods jointly predict activity mean and variance, with explicit supervision using measured noise. Noise-Gated combines heteroscedastic loss with noise-weighted ranking. Contrastive learning uses noise levels to define similarity between samples. On the sampling side, Quantile Sampling constructs batches stratified by activity and noise levels. Curriculum learning progressively increases difficulty. Finally, Hard Negative mining focuses learning on distinguishing similar sequences with reliable labels.

We evaluate all methods on the CAGI5 benchmark, specifically the four K562-matched elements, reporting both Spearman and Pearson correlations stratified by variant confidence.

---

## Slide 3: Test vs CAGI5 Performance

This figure shows the relationship between held-out test performance and CAGI5 clinical variant prediction for all 74 models we trained. Each point represents one model, colored by method category.

The left panel shows Spearman correlations, and the right panel shows Pearson. The dashed red lines indicate baseline performance—a standard MSE-trained model.

The key observation here is that test and CAGI5 metrics are only moderately correlated. You can see substantial vertical spread at any given test performance level. This highlights a critical point: optimizing for test set performance doesn't guarantee good generalization to clinical variants.

Looking at the method categories, Distributional models—shown in purple—cluster in the upper portion of CAGI5 performance despite having similar test scores to other methods. Noise-Gated models in green also show strong CAGI5 performance. The Rank Stability models in blue show moderate CAGI5 scores but, as we'll see later, excel in a different metric.

This disconnect between test and CAGI5 performance justifies our focus on external benchmark evaluation. Models that learn to handle noise appropriately seem to generalize better to clinical variants, even when their held-out test performance is comparable to noise-naive approaches.

The baseline achieves 0.386 Spearman and 0.50 Pearson on CAGI5. Our best models improve both metrics, with Distributional methods showing the largest gains.

---

## Slide 4: Rank Stability (RS) Results

Now let's examine each method category in detail, starting with Rank Stability.

Rank Stability implements a simple but effective idea: when comparing pairs of samples to learn rankings, down-weight pairs where both samples have high measurement noise. The weight function is a sigmoid: weight equals sigmoid of negative k times the sum of squared noise for both samples. This means pairs involving two clean measurements dominate the learning signal.

Panel A shows all RS model variants ranked by CAGI5 Spearman. RS3 achieves the best performance at 0.367, just below the baseline of 0.386. The RS models don't dramatically improve CAGI5, but that's not their primary strength.

Panel B reveals what makes RS special: noise correlation. This metric measures the Pearson correlation between prediction errors and aleatoric uncertainty. Ideally, this should be zero—errors shouldn't track with measurement noise. The baseline has a positive correlation of 0.0155, meaning it makes larger errors on noisier samples.

RS3 achieves negative 0.088—the only model across all 74 with a negative noise correlation. This means RS3 actually makes smaller errors on high-noise samples. This is remarkable and suggests the model has learned representations that are genuinely robust to measurement noise.

Panel C shows the top three RS models stratified by confidence level. Performance is relatively balanced between high-confidence and low-confidence variants.

The parameter k controls how aggressively pairs are down-weighted. RS3 uses k equals 1.0, which appears optimal.

---

## Slide 5: Distributional (DH) Results

Distributional methods take a fundamentally different approach. Instead of predicting just activity, these models output both a mean prediction μ and a variance prediction σ-squared.

The loss function has two components. First, a heteroscedastic negative log-likelihood: one-half times log σ-squared plus the squared error divided by σ-squared. This automatically down-weights samples where the model predicts high uncertainty. Second, we add explicit variance supervision: lambda times the MSE between predicted variance and actual replicate variance from the data.

Panel A shows DH7—the distributional dual-head model—achieving the best CAGI5 Spearman at 0.391, a 1.3% improvement over baseline. The dual-head architecture has separate regression and distributional heads sharing the same encoder, giving it flexibility in combining predictions.

Panel B plots Spearman versus Pearson for all DH variants. Notice DH8 in the upper region—it achieves the highest Pearson correlation at 0.557, an impressive 11.4% improvement over baseline. DH8 uses lambda equals 2.0 for variance supervision, putting stronger weight on learning to predict noise accurately.

Panel C shows confidence-stratified results for the top three models. DH7 performs well across all confidence levels, with particularly strong low-confidence performance at 0.227 compared to baseline's 0.207.

The distributional approach works because teaching the model to recognize noisy samples—through variance prediction—helps it build better internal representations that separate signal from noise.

---

## Slide 6: Noise-Gated (NG) and Ablation Results

Noise-Gated methods combine multiple noise-aware components into a unified loss function. The formula is: L equals heteroscedastic loss plus alpha times rank stability loss plus beta times variance supervision loss.

This combines the automatic down-weighting from heteroscedastic regression, the pairwise noise awareness from rank stability, and explicit variance learning. The key hyperparameters alpha and beta control the balance between these components.

Panel A shows the top NG and ablation models. ABL4 and NG8 both achieve 0.375 Spearman, tied for third-best overall. The ablation experiments—labeled ABL—systematically vary hyperparameters to find optimal values.

Panel B shows the alpha parameter sweep from ABL1 through ABL6. Performance peaks around alpha equals 0.05, then decreases with higher values. This suggests a relatively light ranking component works best—too much ranking emphasis may destabilize training.

Panel C compares Spearman and Pearson for the top five models. All exceed baseline in both metrics, showing robust improvements.

The adaptive noise-gated variant dynamically adjusts alpha during training using cosine annealing. Starting with regression-heavy learning and gradually introducing ranking provides a curriculum effect. However, in our experiments, fixed weights performed comparably to adaptive scheduling.

NG7 uses a simplified version without explicit variance prediction, applying inverse-noise weighting directly to MSE loss. It achieves 0.344 Spearman—modest improvement showing that even simple noise weighting helps.

---

## Slide 7: Contrastive (CA) Results

Contrastive learning approaches noise resistance through representation learning. The idea is to learn embeddings where noise level influences similarity structure.

We define positive pairs as samples with similar activity where both have low noise—these are reliable comparisons. Negative pairs have similar activity but mixed noise levels—one clean, one noisy. The contrastive loss pushes these confounding pairs apart in embedding space while pulling reliable pairs together.

The implementation uses InfoNCE loss with noise-based pair construction. We use quantile thresholds: samples below median noise are considered "clean," and activity differences below the 10th percentile are considered "similar."

Panel A shows high variance across CA models. CA8 with temperature 0.5 achieves the best performance at 0.356, while CA4 using triplet loss performs poorly at only 0.245—actually worse than baseline.

Panel B directly compares contrastive versus triplet loss formulations. Standard contrastive clearly outperforms triplet. The triplet approach constrains the geometry too rigidly, while contrastive allows more flexible embedding structure.

Panel C shows the top three CA models stratified by confidence. Performance is reasonable but not exceptional compared to other methods.

The high variance in CA results suggests sensitivity to hyperparameters, particularly temperature and the noise/activity thresholds. Contrastive methods require careful tuning and may be less reliable as a standalone approach. However, the learned representations could potentially complement other methods.

---

## Slide 8: Sampling Strategy Results

Sampling strategies attack noise from a different angle—controlling batch construction rather than modifying the loss function.

Quantile Sampling stratifies batches by activity level, ensuring each batch contains samples from across the activity range. Within each stratum, we can optionally weight samples by inverse noise, favoring cleaner measurements. QS6 uses noise weight 0.5, meaning half uniform and half inverse-noise weighted.

Curriculum Learning starts with easy distinctions—extreme high or low activity values—and progressively adds harder samples near the median. The intuition is that extreme values are easier to rank correctly, building model confidence before tackling ambiguous cases.

Hard Negative Mining identifies informative training pairs: samples with small activity differences and low noise. These are the cases where the model can learn fine-grained distinctions without being confounded by measurement uncertainty.

Panel A shows all sampling models ranked. QS6 achieves the best at 0.373, a meaningful improvement over baseline. Curriculum and Hard Negative methods show more modest gains.

Panel B compares the three strategies via box plots. Quantile Sampling shows the most consistent improvement with tight variance. Curriculum has higher variance, and Hard Negative falls in between.

Panel C displays the best model from each strategy. QS6 outperforms QC1 and HN1, though all exceed baseline.

Sampling strategies are computationally cheap and can be combined with any loss function. QS6's success suggests that balanced activity representation combined with noise-aware weighting within strata is an effective approach.

---

## Slide 9: Comprehensive Method Comparison

This four-panel figure provides a systematic comparison across all seven method categories using different evaluation metrics.

Panel A shows CAGI5 Spearman by method. Distributional and Noise-Gated methods have the highest medians and show improvements over the baseline dashed line. Rank Stability and Quantile Sampling also consistently exceed baseline. Contrastive shows the highest variance, consistent with its sensitivity to hyperparameters.

Panel B shows CAGI5 Pearson, which measures linear correlation. The pattern is similar—DH and NG lead—but the relative improvement is larger for Pearson. DH8 achieving 0.557 represents an 11.4% improvement, while the best Spearman improvement is only 1.3%.

Panel C focuses on high-confidence variants, those with ground truth effect size above 0.1. Most methods match or slightly exceed baseline here. The gains are smaller because high-confidence variants are already well-predicted—there's less room for improvement.

Panel D shows low-confidence variants, which is where noise-resistant methods really shine. The baseline achieves only 0.207 on these challenging variants. DH methods show the largest gains, with DH7 reaching 0.227—nearly a 10% relative improvement. This makes sense: low-confidence variants have smaller effect sizes that are harder to distinguish from noise. Models that handle noise better should excel here.

Overall, this comparison validates that noise-resistant training provides consistent benefits, with Distributional and Noise-Gated methods offering the most robust improvements across metrics.

---

## Slide 10: Top 15 Models

Let's look at the top 15 models ranked by overall CAGI5 Spearman.

Panel A shows Spearman correlations broken down by confidence level. DH7 leads at 0.391 overall. Notice the consistent pattern: high-confidence performance around 0.65-0.68, low-confidence around 0.19-0.23, with overall reflecting the weighted average.

The top 15 includes representatives from multiple method categories: DH7, DH8, DH3 from Distributional; ABL4, NG8 from Noise-Gated; RS3, RS1, RS2 from Rank Stability; QS6 from Quantile Sampling; and HN1 from Hard Negative. This diversity suggests that multiple approaches to noise resistance are viable.

Panel B shows the same breakdown for Pearson correlations. DH8 stands out with significantly higher Pearson than other models, reaching 0.557. This suggests DH8 captures linear relationships particularly well, possibly because stronger variance supervision—lambda equals 2.0—leads to better-calibrated predictions.

Looking at the low-confidence bars specifically, DH7 achieves 0.227, DH8 achieves 0.192, and RS3 achieves 0.213. The variation here is meaningful—different methods trade off high-confidence versus low-confidence performance differently.

For practitioners, DH7 offers the best overall Spearman with balanced confidence performance. DH8 is preferable if Pearson correlation is the primary metric. RS3 is the choice if noise resistance—avoiding errors on uncertain measurements—is critical.

---

## Slide 11: Per-Element CAGI5 Performance

This heatmap breaks down performance on each of the four K562-matched CAGI5 elements: GP1BB, HBB, HBG1, and PKLR.

Panel A shows Spearman correlations. HBB—the beta-globin promoter—consistently shows the highest correlations across all models, reaching 0.45-0.47 for top performers. This suggests HBB's regulatory grammar is well-captured by our model architecture.

GP1BB—the platelet glycoprotein promoter—shows the largest model-dependent variation. The baseline achieves only 0.147 on GP1BB, while DH7 reaches 0.312. That's a 112% relative improvement, the largest gain on any element. GP1BB may have regulatory features that noise-naive models struggle with but noise-aware models capture.

HBG1 and PKLR show intermediate performance and improvement. HBG1 correlations range from 0.32 to 0.45, while PKLR ranges from 0.27 to 0.35.

Panel B shows Pearson correlations with a similar pattern. DH8's strength is apparent across elements, achieving the highest or near-highest Pearson on all four.

The element-level analysis reveals that noise-resistant training benefits all elements, but the magnitude varies. GP1BB benefits most dramatically, possibly because it has higher measurement noise in the training data or regulatory features that require more robust learning to capture.

For comprehensive variant effect prediction, using an ensemble or selecting models based on the target element may be beneficial.

---

## Slide 12: Noise Correlation Analysis

This figure examines noise correlation in depth—measuring whether prediction errors track with experimental uncertainty.

Panel A shows noise correlation by method category. Rank Stability models—the blue points—achieve negative values, meaning they make smaller errors on noisier samples. All other methods show positive correlations, though most are below the baseline value of 0.0155.

Panel B displays the distribution across all models. The histogram is centered around 0.02 with a long left tail. The baseline sits at the dashed red line. RS3 achieves -0.088, shown by the green dashed line—a clear outlier in the beneficial direction.

Panel C plots noise correlation against CAGI5 Spearman. There's no strong relationship, suggesting these capture different aspects of model quality. A model can have good CAGI5 performance with moderate noise correlation, or excellent noise resistance with moderate CAGI5. RS3 offers both reasonable CAGI5 and best noise resistance.

Panel D shows noise correlation versus test set Spearman. Again, no strong pattern—noise resistance is somewhat independent of standard performance metrics.

The interpretation of negative noise correlation is subtle. It doesn't mean the model ignores noisy samples—it means the model has learned to not rely on features that correlate with experimental noise. RS3 apparently builds representations that are invariant to noise-related confounds.

For applications where model reliability on uncertain measurements matters—such as clinical variant interpretation—RS3's unique property is valuable.

---

## Slide 13: Confidence Gap Analysis

The gap between high-confidence and low-confidence performance indicates how balanced a model's predictions are across the difficulty spectrum.

Panel A plots this HC-LC gap against overall CAGI5 Spearman. Each point is one model. The ideal region is upper-left: high overall performance with small gap, indicating balanced predictions.

Most models cluster with gaps between 0.4 and 0.5 and overall Spearman between 0.30 and 0.39. The top-performing models—DH7, ABL4, NG8—appear in the upper portion with gaps around 0.45.

I've labeled the top four models. DH7 achieves 0.391 overall with a 0.453 gap. RS3 has 0.367 overall with a 0.429 gap—one of the smaller gaps among good performers.

Panel B shows gap distributions by method category. Distributional methods show the smallest median gap, indicating more balanced performance. Noise-Gated also shows relatively small gaps. Rank Stability shows moderate gaps, while Contrastive and Curriculum show larger gaps.

Why does balanced performance matter? In clinical applications, we care about both clear pathogenic variants—high confidence—and variants of uncertain significance—low confidence. A model that excels only on clear cases but struggles on ambiguous ones provides less clinical utility.

DH methods' small gaps likely stem from explicit uncertainty modeling. By learning to predict variance, the model develops calibrated uncertainty that correlates with prediction difficulty, leading to more balanced performance across confidence levels.

---

## Slide 14: Combined Methods (BEST)

We tested whether combining multiple noise-resistant strategies would yield additive benefits. The BEST models combine Noise-Gated loss with various sampling strategies.

Panel A shows the four BEST model variants. BEST2 uses Noise-Gated loss with quantile-noise weighted sampling, achieving 0.325 Spearman. BEST1 uses Noise-Gated with standard quantile sampling at 0.317. BEST3 combines with Hard Negative sampling at 0.333.

Panel B compares BEST models against their component methods. The gray bars show individual components: NG8 at 0.375, QS6 at 0.373, HN1 at 0.362. The orange bars show combinations.

Surprisingly, combinations don't outperform the best individual methods. BEST2 achieves 0.325, below both NG8 and QS6 individually. This suggests the strategies may not be complementary—they might address overlapping aspects of the noise problem.

Panel C shows confidence-stratified results. BEST models perform respectably but not exceptionally across confidence levels.

Several explanations are possible. First, over-regularization: combining multiple noise-handling approaches may be overly conservative, limiting the model's capacity to fit real signal. Second, optimization difficulty: more complex training objectives may have more challenging loss landscapes. Third, redundancy: if noise-gated loss already handles noise well, additional sampling-based noise handling provides diminishing returns.

The practical implication is that practitioners should choose the single best approach—likely Distributional or Noise-Gated—rather than attempting to combine multiple strategies.

---

## Slide 15: Summary and Recommendations

Let me summarize our findings and provide practical recommendations.

Panel A shows the top 10 models overall. DH7 leads, followed by ABL4, NG8, QS6, and others from various method categories. The diversity confirms multiple viable approaches exist.

Panel B shows the best model from each method category. Distributional achieves the highest absolute performance. Rank Stability offers unique noise resistance. Noise-Gated provides a robust combined approach. Sampling strategies offer meaningful improvements with minimal implementation complexity.

Panel C presents key quantitative findings. DH7 achieves the best CAGI5 Spearman at 0.391, a 1.3% improvement. DH8 achieves the best CAGI5 Pearson at 0.557, an 11.4% improvement. DH7 also achieves the best low-confidence Spearman at 0.227, a 9.7% improvement. RS3 uniquely achieves negative noise correlation at -0.088.

Panel D provides practical recommendations. For best overall performance, use DH7—the distributional dual-head model. It offers the highest CAGI5 Spearman with balanced confidence performance. For applications requiring noise resistance—where prediction reliability on uncertain measurements is critical—use RS3. For best linear correlation, particularly if Pearson is your primary metric, use DH8 with strong variance supervision. For a robust all-around choice with good performance across metrics, ABL4 with optimized Noise-Gated parameters is reliable.

In conclusion, noise-resistant training provides meaningful improvements for clinical variant prediction. The choice of method depends on specific application requirements, but Distributional approaches offer the most consistent benefits.

Thank you. I'm happy to take questions.

---

## Appendix: Technical Details for Q&A

### Heteroscedastic Loss Formula
```
L = 0.5 * [log(σ²) + (y - μ)² / σ²] + λ * MSE(σ², σ²_true)
```

### Rank Stability Weight Function
```
w_ij = sigmoid(-k * (σ²_i + σ²_j))
```
where k=1.0 for RS3

### Model Architecture
- Backbone: DREAM_RNN (BiLSTM + parallel convolutions)
- Input: 230bp one-hot encoded sequences (4 × 230)
- Hidden dimension: 512 → 256
- Output: μ and log(σ²) for distributional models

### Training Details
- Epochs: 80
- Batch size: 1024
- Optimizer: AdamW with OneCycleLR scheduler
- Learning rate: 0.005 with 10-epoch warmup

### CAGI5 Evaluation
- Alt-Ref scoring: effect = model(Alt) - model(Ref)
- K562 elements: GP1BB (n=869), HBB (n=432), HBG1 (n=633), PKLR (n=1025)
- High-confidence: |ground_truth| ≥ 0.1
- Low-confidence: |ground_truth| < 0.1
