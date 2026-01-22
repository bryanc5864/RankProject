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
    # Combined
    "combined_loss",
    "CombinedLoss",
    "AdaptiveCombinedLoss",
    "MultiTaskRankingLoss",
    "UncertaintyWeightedLoss",
    # Soft Classification
    "SoftClassificationLoss",
    "OrdinalRegressionLoss",
    "soft_classification_loss",
]
