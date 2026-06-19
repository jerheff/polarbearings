"""Brier score metric implemented as a Polars expression."""

import polars as pl

from polarbearings._common import WeightInput, resolve_weight, weight_suffix


def brier_score(
    target: str, prob: str, weight: WeightInput = None, pos_label: int | float | str | bool = 1
) -> pl.Expr:
    """Compute Brier score for binary classification.

    Args:
        target: Name of the column containing class labels.
        prob: Name of the column containing predicted probabilities [0, 1].
        weight: Optional name of the column containing sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).

    Returns:
        A Polars expression that computes the Brier score.

    Notes:
        - Lower is better (0 is perfect).
        - Brier score = mean((predicted_probability - actual_outcome)²).
        - Proper scoring rule (rewards calibrated probabilities).
    """
    target_float = (pl.col(target) == pos_label).cast(pl.Float64)
    per_sample = (pl.col(prob) - target_float) ** 2

    w = resolve_weight(weight)
    brier = (per_sample * w).sum() / w.sum() if w is not None else per_sample.mean()

    alias = f"brier_score_{target}_{prob}"
    alias += weight_suffix(weight)
    if pos_label != 1:
        alias += f"_pos{pos_label}"
    return brier.alias(alias)
