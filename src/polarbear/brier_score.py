"""Brier score metric implemented as a Polars expression."""

import polars as pl


def brier_score(target: str, prob: str, weight: str | None = None) -> pl.Expr:
    """Compute Brier score for binary classification.

    Args:
        target: Name of the column containing binary labels (0 or 1).
        prob: Name of the column containing predicted probabilities [0, 1].
        weight: Optional name of the column containing sample weights.

    Returns:
        A Polars expression that computes the Brier score.

    Notes:
        - Lower is better (0 is perfect).
        - Brier score = mean((predicted_probability - actual_outcome)²).
        - Proper scoring rule (rewards calibrated probabilities).
    """
    target_float = pl.col(target).cast(pl.Float64)
    per_sample = (pl.col(prob) - target_float) ** 2

    if weight is not None:
        w = pl.col(weight).cast(pl.Float64)
        brier = (per_sample * w).sum() / w.sum()
    else:
        brier = per_sample.mean()

    alias = f"brier_score_{target}_{prob}"
    if weight is not None:
        alias += f"_{weight}"
    return brier.alias(alias)
