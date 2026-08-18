from .plackett_luce import (
    plackett_luce_loss,
    plackett_luce_loss_with_ties,
    weighted_plackett_luce_loss,
)
from .ranknet import (
    ranknet_loss,
    margin_ranknet_loss,
    lambda_ranknet_loss,
    sampled_ranknet_loss,
)
from .softsort import (
    softsort_loss,
    softsort_spearman_loss,
    soft_ndcg_loss,
    differentiable_rank_mse,
    TORCHSORT_AVAILABLE,
)
from .combined import (
    combined_loss,
    CombinedLoss,
    AdaptiveCombinedLoss,
    MultiTaskRankingLoss,
    UncertaintyWeightedLoss,
)
from .soft_classification import (
    SoftClassificationLoss,
    OrdinalRegressionLoss,
    soft_classification_loss,
)
from .rank_stability import (
    RankStabilityRankNet,
    SampledRankStabilityRankNet,
    rank_stability_ranknet_loss,
    rank_stability_weight,
)
from .distributional import (
    DistributionalLoss,
    HeteroscedasticDistributionalLoss,
    VarianceWeightedMSE,
    distributional_loss,
    heteroscedastic_distributional_loss,
)
from .noise_gated import (
    NoiseGatedRanking,
    AdaptiveNoiseGatedRanking,
    NoiseGatedMSERanking,
    noise_gated_ranking_loss,
)
from .contrastive_anchor import (
    ContrastiveNoiseAnchor,
    TripletNoiseAnchor,
    SoftContrastiveNoiseAnchor,
    contrastive_noise_anchor_loss,
)

__all__ = [
    # Plackett-Luce
    "plackett_luce_loss",
    "plackett_luce_loss_with_ties",
    "weighted_plackett_luce_loss",
    # RankNet
    "ranknet_loss",
    "margin_ranknet_loss",
    "lambda_ranknet_loss",
    "sampled_ranknet_loss",
    # SoftSort
    "softsort_loss",
    "softsort_spearman_loss",
    "soft_ndcg_loss",
    "differentiable_rank_mse",
    "TORCHSORT_AVAILABLE",
    # combined
    "combined_loss",
    "CombinedLoss",
    "AdaptiveCombinedLoss",
    "MultiTaskRankingLoss",
    "UncertaintyWeightedLoss",
    # soft classification
    "SoftClassificationLoss",
    "OrdinalRegressionLoss",
    "soft_classification_loss",
    # rank stability (noise-aware)
    "RankStabilityRankNet",
    "SampledRankStabilityRankNet",
    "rank_stability_ranknet_loss",
    "rank_stability_weight",
    # distributional (noise-aware)
    "DistributionalLoss",
    "HeteroscedasticDistributionalLoss",
    "VarianceWeightedMSE",
    "distributional_loss",
    "heteroscedastic_distributional_loss",
    # noise-gated (noise-aware)
    "NoiseGatedRanking",
    "AdaptiveNoiseGatedRanking",
    "NoiseGatedMSERanking",
    "noise_gated_ranking_loss",
    # contrastive anchor (noise-aware)
    "ContrastiveNoiseAnchor",
    "TripletNoiseAnchor",
    "SoftContrastiveNoiseAnchor",
    "contrastive_noise_anchor_loss",
]
