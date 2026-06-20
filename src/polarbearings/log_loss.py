"""Log loss metric implemented as a Polars expression."""

import polars as pl

from polarbearings._common import (
    IntoExpr,
    WeightInput,
    col_expr,
    col_name,
    guarded,
    resolve_weight,
    weight_suffix,
)


def log_loss(
    target: IntoExpr,
    prob: IntoExpr,
    eps: float = 1e-15,
    weight: WeightInput = None,
    pos_label: int | float | str | bool = 1,
) -> pl.Expr:
    """Compute log loss (binary cross-entropy) for binary classification.

    Args:
        target: Column name or expression containing class labels.
        prob: Column name or expression containing predicted probabilities [0, 1].
        eps: Small constant to clip probabilities for numerical stability.
        weight: Optional name of the column containing sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).

    Returns:
        A Polars expression that computes the log loss.

    Notes:
        - Lower is better (0 is perfect).
        - Heavily penalizes confident wrong predictions.
        - Probabilities are clipped to [eps, 1-eps] for numerical stability.
    """
    target_float = (col_expr(target) == pos_label).cast(pl.Float64)

    prob_clipped = col_expr(prob).clip(eps, 1 - eps)
    log_prob = prob_clipped.log()
    log_1_minus_prob = (1 - prob_clipped).log()

    per_sample = -(target_float * log_prob + (1 - target_float) * log_1_minus_prob)

    w = resolve_weight(weight)
    loss = (per_sample * w).sum() / w.sum() if w is not None else per_sample.mean()

    alias = f"log_loss_{col_name(target)}_{col_name(prob)}"
    alias += weight_suffix(weight)
    if pos_label != 1:
        alias += f"_pos{pos_label}"
    return guarded(loss, values=[prob], labels=[target], weight=weight).alias(alias)
