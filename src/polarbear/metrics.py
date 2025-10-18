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
    # Cast target to float once to avoid repeated conversions
    target_float = pl.col(target).cast(pl.Float64)

    # Calculate total number of positive examples
    total_pos = (target_float == 1).sum()
    # Total negatives = count - positives (more efficient than separate computation)
    total_neg = target_float.len() - total_pos

    # Edge case: all scores are identical (perfect ties)
    # Check using variance (0 variance means all values are the same)
    # This is faster than computing min and max separately
    tie_cond = pl.col(score).var() == 0

    # Use Mann-Whitney U statistic formulation which handles ties correctly
    # AUC = (sum of ranks of positive class - min possible sum) / (n_pos * n_neg)
    # This is equivalent to the probability that a random positive example
    # scores higher than a random negative example, with ties counted as 0.5

    # Assign fractional ranks (average rank for ties) - this matches sklearn behavior
    # rank() with method='average' gives the average rank for tied values
    ranks = pl.col(score).rank(method="average")

    # Sum of ranks for positive examples (reuse target_float to avoid casting)
    pos_rank_sum = (ranks * (target_float == 1)).sum()

    # Min possible sum of ranks for positive examples is 1+2+...+n_pos = n_pos*(n_pos+1)/2
    min_pos_rank_sum = total_pos * (total_pos + 1) / 2

    # Mann-Whitney U statistic
    u_statistic = pos_rank_sum - min_pos_rank_sum

    # AUC = U / (n_pos * n_neg)
    auc = u_statistic / (total_pos * total_neg)

    # Return final AUC with special handling for perfect ties
    return pl.when(tie_cond).then(pl.lit(0.5)).otherwise(auc).alias(f"roc_auc_{target}_{score}")


def log_loss(target: str, score: str, eps: float = 1e-15) -> pl.Expr:
    """Compute log loss (binary cross-entropy) for binary classification.

    Log loss measures the performance of a classification model where the
    prediction is a probability value between 0 and 1. It penalizes false
    classifications heavily, especially confident wrong predictions.

    Args:
        target: Name of the column containing binary labels (0 or 1).
        score: Name of the column containing predicted probabilities [0, 1].
        eps: Small constant to clip probabilities for numerical stability.

    Returns:
        A Polars expression that computes the log loss.

    Examples:
        >>> import polars as pl
        >>> from polarbear import log_loss
        >>>
        >>> df = pl.DataFrame({
        ...     "label": [0, 0, 1, 1],
        ...     "prob": [0.1, 0.2, 0.8, 0.9]
        ... })
        >>> df.select(log_loss("label", "prob"))

    Notes:
        - Lower is better (0 is perfect)
        - Heavily penalizes confident wrong predictions
        - Scores are clipped to [eps, 1-eps] for numerical stability
    """
    # Cast target to float once
    target_float = pl.col(target).cast(pl.Float64)

    # Clip probabilities to avoid log(0)
    prob_clipped = pl.col(score).clip(eps, 1 - eps)

    # Log loss = -1/N * Σ(y*log(p) + (1-y)*log(1-p))
    # Compute log once per probability to avoid redundant calculations
    log_prob = prob_clipped.log()
    log_1_minus_prob = (1 - prob_clipped).log()

    loss = -(target_float * log_prob + (1 - target_float) * log_1_minus_prob).mean()

    return loss.alias(f"log_loss_{target}_{score}")


def brier_score(target: str, score: str) -> pl.Expr:
    """Compute Brier score for binary classification.

    The Brier score measures the mean squared error between predicted
    probabilities and actual outcomes. It's a proper scoring rule that
    measures the accuracy of probabilistic predictions.

    Args:
        target: Name of the column containing binary labels (0 or 1).
        score: Name of the column containing predicted probabilities [0, 1].

    Returns:
        A Polars expression that computes the Brier score.

    Examples:
        >>> import polars as pl
        >>> from polarbear import brier_score
        >>>
        >>> df = pl.DataFrame({
        ...     "label": [0, 0, 1, 1],
        ...     "prob": [0.1, 0.2, 0.8, 0.9]
        ... })
        >>> df.select(brier_score("label", "prob"))

    Notes:
        - Lower is better (0 is perfect)
        - Brier score = mean((predicted_probability - actual_outcome)²)
        - Proper scoring rule (rewards calibrated probabilities)
    """
    # Brier score = 1/N * Σ(p - y)²
    target_float = pl.col(target).cast(pl.Float64)
    brier = ((pl.col(score) - target_float) ** 2).mean()

    return brier.alias(f"brier_score_{target}_{score}")
