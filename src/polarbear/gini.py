"""Normalized Gini coefficient implemented as a Polars expression.

The Gini coefficient measures how well a score ranks observations by a target
variable. It is commonly used in fraud and credit-risk work where large losses
should receive the highest scores.

The returned value is normalized so that:

* ``1.0`` means the score ordering is optimal for the given target distribution.
* ``0.0`` means the score is no better than random.
* Negative values mean the score ordering is worse than random.

Target values must be non-negative. Undefined cases return ``null``.
"""

import polars as pl


def gini_coefficient(target: str, score: str, weight: str | None = None) -> pl.Expr:
    """Compute the normalized Gini coefficient as a Polars expression.

    Args:
        target: Column name with the target values (e.g. fraud losses). Values
            must be non-negative.
        score: Column name with the score used to rank observations. Higher
            scores should correspond to larger target values.
        weight: Optional column name with sample weights. If provided, the
            Lorenz curve uses cumulative weight on the x-axis.

    Returns:
        A Polars expression evaluating to the normalized Gini coefficient.
    """
    alias = f"gini_{target}_{score}"
    if weight is not None:
        alias += f"_{weight}"

    if weight is not None:
        return _gini_weighted(target, score, weight, alias)
    return _gini_unweighted(target, score, alias)


def _gini_unweighted(target: str, score: str, alias: str) -> pl.Expr:
    """Rank-based normalized Gini for the unweighted case."""
    target_float = pl.col(target).cast(pl.Float64)
    score_col = pl.col(score)

    n = score_col.len().cast(pl.Float64)
    total = target_float.sum()

    # Standard Gini coefficient using ascending average ranks. This equals
    # ``2 * AUC - 1`` for binary targets.
    rank_score = score_col.rank(method="average")
    raw = (2 * (rank_score * target_float).sum() - (n + 1) * total) / (n * total)

    # Perfect ordering for the observed target values.
    rank_target = target_float.rank(method="average")
    perfect = (2 * (rank_target * target_float).sum() - (n + 1) * total) / (n * total)

    undefined = (n < 2) | (total == 0) | (perfect == 0)
    return pl.when(undefined).then(None).otherwise(raw / perfect).alias(alias)


def _gini_weighted(target: str, score: str, weight: str, alias: str) -> pl.Expr:
    """Weighted normalized Gini via Lorenz curve areas."""
    target_float = pl.col(target).cast(pl.Float64)
    weight_float = pl.col(weight).cast(pl.Float64)
    score_col = pl.col(score).cast(pl.Float64)

    n = score_col.len().cast(pl.Float64)
    total_target = target_float.sum()
    total_weight = weight_float.sum()

    def _lorenz_gini(target_sorted: pl.Expr, weight_sorted: pl.Expr) -> pl.Expr:
        cum_weight = weight_sorted.cum_sum() / total_weight
        cum_target = target_sorted.cum_sum() / total_target

        delta_weight = cum_weight - cum_weight.shift(1).fill_null(0)
        avg_target = (cum_target + cum_target.shift(1).fill_null(0)) / 2

        area = (delta_weight * avg_target).sum()
        return 2 * area - 1

    # Model ordering: sort by the provided score descending.
    weight_by_score = weight_float.sort_by(score_col, descending=True)
    target_by_score = target_float.sort_by(score_col, descending=True)
    raw = _lorenz_gini(target_by_score, weight_by_score)

    # Perfect ordering for weighted data ranks by target per unit weight.
    ratio = pl.when(weight_float == 0).then(float("inf")).otherwise(target_float / weight_float)
    weight_by_ratio = weight_float.sort_by(ratio, descending=True)
    target_by_ratio = target_float.sort_by(ratio, descending=True)
    perfect = _lorenz_gini(target_by_ratio, weight_by_ratio)

    undefined = (n < 2) | (total_target == 0) | (total_weight == 0) | (perfect == 0)
    return pl.when(undefined).then(None).otherwise(raw / perfect).alias(alias)
