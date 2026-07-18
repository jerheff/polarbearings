"""Log loss metric implemented as a Polars expression."""

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


def log_loss(
    target: IntoExpr,
    prob: IntoExpr,
    *,
    eps: float = 1e-15,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
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

    Examples:
        >>> import polars as pl
        >>> from polarbearings import log_loss
        >>>
        >>> df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.4, 0.6, 0.9]})
        >>> df.select(log_loss("label", "prob"))
        shape: (1, 1)
        ┌─────────────────────┐
        │ log_loss_label_prob │
        │ ---                 │
        │ f64                 │
        ╞═════════════════════╡
        │ 0.308093            │
        └─────────────────────┘
    """
    target_float = (col_expr(target) == pos_label).cast(pl.Float64)

    prob_clipped = col_expr(prob).clip(eps, 1 - eps)
    log_prob = prob_clipped.log()
    log_1_minus_prob = (1 - prob_clipped).log()

    per_sample = -(target_float * log_prob + (1 - target_float) * log_1_minus_prob)

    loss = weighted_mean(per_sample, resolve_weight(weight))

    alias = f"log_loss_{col_name(target)}_{col_name(prob)}"
    alias += weight_suffix(weight)
    if pos_label != 1:
        alias += f"_pos{pos_label}"
    return guarded(loss, values=[prob], labels=[target], weight=weight).alias(alias)
