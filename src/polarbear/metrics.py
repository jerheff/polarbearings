"""Machine learning metrics implemented as Polars expressions."""

import polars as pl


def roc_auc(target: str, score: str) -> pl.Expr:
    """Compute ROC AUC score for binary classification as a Polars expression.

    This function returns a Polars expression that calculates the Area Under the
    Receiver Operating Characteristic Curve (ROC AUC) for binary classification tasks.
    The implementation matches scikit-learn's roc_auc_score behavior, including
    proper handling of tied scores.

    The ROC AUC metric measures the ability of a classifier to distinguish between
    classes. A score of 1.0 indicates perfect classification, 0.5 indicates random
    guessing, and 0.0 indicates perfectly inverted predictions.

    Args:
        target: Name of the column containing binary labels (0 or 1).
        score: Name of the column containing prediction scores (higher = more likely positive).

    Returns:
        A Polars expression that computes the ROC AUC score. The result is aliased
        as f"roc_auc_{target}_{score}".

    Examples:
        >>> import polars as pl
        >>> from polarbear import roc_auc
        >>>
        >>> # Perfect classification
        >>> df = pl.DataFrame({
        ...     "label": [0, 0, 1, 1],
        ...     "score": [0.1, 0.2, 0.8, 0.9]
        ... })
        >>> df.select(roc_auc("label", "score"))
        shape: (1, 1)
        ┌──────────────────────┐
        │ roc_auc_label_score  │
        │ ---                  │
        │ f64                  │
        ╞══════════════════════╡
        │ 1.0                  │
        └──────────────────────┘

        >>> # Random prediction
        >>> df = pl.DataFrame({
        ...     "label": [0, 1, 0, 1],
        ...     "score": [0.5, 0.5, 0.5, 0.5]
        ... })
        >>> df.select(roc_auc("label", "score"))
        shape: (1, 1)
        ┌──────────────────────┐
        │ roc_auc_label_score  │
        │ ---                  │
        │ f64                  │
        ╞══════════════════════╡
        │ 0.5                  │
        └──────────────────────┘

    Notes:
        - Handles tied scores correctly using sklearn-compatible logic
        - Returns 0.5 when all scores are identical
        - Requires at least one positive and one negative example
        - Uses the trapezoidal rule for AUC calculation

    See Also:
        - scikit-learn's roc_auc_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        - ROC curves: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """
    # Calculate total number of positive and negative examples
    total_pos = (pl.col(target) == 1).cast(pl.Float64).sum()
    total_neg = (pl.col(target) == 0).cast(pl.Float64).sum()

    # Edge case: all scores are identical (perfect ties)
    # In this case, the classifier provides no discrimination → AUC = 0.5
    tie_cond = pl.col(score).max() == pl.col(score).min()

    # Use Mann-Whitney U statistic formulation which handles ties correctly
    # AUC = (sum of ranks of positive class - min possible sum) / (n_pos * n_neg)
    # This is equivalent to the probability that a random positive example
    # scores higher than a random negative example, with ties counted as 0.5

    # Assign fractional ranks (average rank for ties) - this matches sklearn behavior
    # rank() with method='average' gives the average rank for tied values
    ranks = pl.col(score).rank(method="average")

    # Sum of ranks for positive examples
    pos_rank_sum = (ranks * (pl.col(target) == 1).cast(pl.Float64)).sum()

    # Min possible sum of ranks for positive examples is 1+2+...+n_pos = n_pos*(n_pos+1)/2
    min_pos_rank_sum = total_pos * (total_pos + 1) / 2

    # Mann-Whitney U statistic
    u_statistic = pos_rank_sum - min_pos_rank_sum

    # AUC = U / (n_pos * n_neg)
    auc = u_statistic / (total_pos * total_neg)

    # Return final AUC with special handling for perfect ties
    return (
        pl.when(tie_cond)
        .then(pl.lit(0.5))
        .otherwise(auc)
        .alias(f"roc_auc_{target}_{score}")
    )
