import polars as pl


def roc_auc(target: str, score: str) -> pl.Expr:
    """Return a Polars expression to compute ROC AUC for a binary target."""
    # Get total positives and negatives
    total_pos = (pl.col(target) == 1).cast(pl.Float64).sum()
    total_neg = (pl.col(target) == 0).cast(pl.Float64).sum()

    # Handle case when all scores are identical
    tie_cond = pl.col(score).max() == pl.col(score).min()

    # Sort by score (high to low) and create a rank column to handle ties
    score_sorted = pl.col(score).sort(descending=True)
    target_sorted = pl.col(target).sort_by(pl.col(score), descending=True)

    pos = (target_sorted == 1).cast(pl.Float64)
    neg = (target_sorted == 0).cast(pl.Float64)
    cum_pos = pos.cum_sum()
    cum_neg = neg.cum_sum()
    tpr = cum_pos / total_pos
    fpr = cum_neg / total_neg

    # For standard ROC AUC calculation (trapezoid method)
    delta_fpr = (fpr - fpr.shift(1)).fill_null(0)
    trapezoid_area = (delta_fpr * (tpr + tpr.shift(1)) / 2).sum()

    # Check for edge cases
    has_duplicates_cond = pl.col(score).is_duplicated().any()

    # When we have ties in scores, use sklearn-compatible calculation
    # to ensure we match the expected behavior
    sorted_scores_desc = pl.col(score).sort(descending=True)
    sorted_labels = pl.col(target).sort_by(pl.col(score), descending=True)

    # Group by distinct score values
    distinct_scores = sorted_scores_desc.is_first_distinct()

    # Calculate FPR and TPR at each distinct threshold
    group_tpr = sorted_labels.cum_sum().filter(distinct_scores) / total_pos
    group_fpr = (sorted_labels == 0).cast(pl.Float64).cum_sum().filter(
        distinct_scores
    ) / total_neg

    # Calculate AUC using trapezoidal rule on distinct thresholds
    width = group_fpr - group_fpr.shift(1).fill_null(0)
    height = (group_tpr + group_tpr.shift(1).fill_null(0)) / 2
    auc_tied = (width * height).sum()

    # Use the tie-handling version when duplicates exist
    trapezoid_area = (
        pl.when(has_duplicates_cond).then(auc_tied).otherwise(trapezoid_area)
    )

    return (
        pl.when(tie_cond)
        .then(pl.lit(0.5))
        .otherwise(trapezoid_area)
        .alias(f"roc_auc_{target}_{score}")
    )
