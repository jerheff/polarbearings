"""Brier score metric implemented as a Polars expression."""

import polars as pl

from polarbearings._common import (
    IntoExpr,
    PosLabel,
    WeightInput,
    col_expr,
    col_name,
    guarded,
    resolve_weight,
    weight_suffix,
    weighted_mean,
)


def brier_score(
    target: IntoExpr,
    prob: IntoExpr,
    *,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
) -> pl.Expr:
    """Compute Brier score for binary classification.

    Args:
        target: Column name or expression containing class labels.
        prob: Column name or expression containing predicted probabilities [0, 1].
        weight: Optional name of the column containing sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).

    Returns:
        A Polars expression that computes the Brier score.

    Notes:
        - Lower is better (0 is perfect).
        - Brier score = mean((predicted_probability - actual_outcome)²).
        - Proper scoring rule (rewards calibrated probabilities).
    """
    target_float = (col_expr(target) == pos_label).cast(pl.Float64)
    per_sample = (col_expr(prob) - target_float) ** 2

    brier = weighted_mean(per_sample, resolve_weight(weight))

    alias = f"brier_score_{col_name(target)}_{col_name(prob)}"
    alias += weight_suffix(weight)
    if pos_label != 1:
        alias += f"_pos{pos_label}"
    return guarded(brier, values=[prob], labels=[target], weight=weight).alias(alias)
