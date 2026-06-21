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

from polarbearings._common import (
    IntoExpr,
    PosLabel,
    WeightInput,
    col_expr,
    col_name,
    guarded,
    weight_expr,
    weight_suffix,
)


def _gini_target(target: IntoExpr, pos_label: PosLabel | None) -> pl.Expr:
    """Resolve the target to the non-negative Float64 magnitude Gini ranks by.

    With ``pos_label=None`` the target is used directly (a continuous value, or an
    already-numeric 0/1 indicator). Otherwise it is mapped to a 0/1 indicator via
    ``== pos_label``, so a string/categorical/non-1 binary class label works too.
    """
    if pos_label is None:
        return col_expr(target).cast(pl.Float64)
    return (col_expr(target) == pos_label).cast(pl.Float64)


def gini_coefficient(
    target: IntoExpr,
    score: IntoExpr,
    *,
    weight: WeightInput = None,
    pos_label: PosLabel | None = None,
) -> pl.Expr:
    """Compute the normalized Gini coefficient as a Polars expression.

    Args:
        target: Column name or expression with the target. Either a non-negative
            magnitude (continuous, e.g. fraud losses) used directly, or a class
            label resolved with ``pos_label``.
        score: Column name or expression with the score used to rank
            observations. Higher scores should correspond to larger target
            values.
        weight: Optional column name with sample weights. If provided, the
            Lorenz curve uses cumulative weight on the x-axis.
        pos_label: When given, map ``target == pos_label`` to a 0/1 indicator
            before computing Gini (= ``2·AUC − 1`` for unweighted data — see the
            note below); needed for string/categorical or non-1 binary labels. When
            ``None`` (default), the target is used as a numeric magnitude directly.

    Returns:
        A Polars expression evaluating to the normalized Gini coefficient.

    Note:
        The ``2·AUC − 1`` identity for binary targets holds for the **unweighted**
        case only. The weighted normalized Gini uses a per-unit-weight
        perfect-ordering baseline (and a count-vs-weight Lorenz-axis asymmetry), so
        weighted binary Gini does **not** equal ``2·weighted_AUC − 1`` — the two can
        differ materially. Reach for :func:`roc_auc` if you specifically need a
        weighted AUC.
    """
    alias = f"gini_{col_name(target)}_{col_name(score)}"
    alias += weight_suffix(weight)
    if pos_label is not None:
        alias += f"_pos{pos_label}"

    if weight is not None:
        result = _gini_weighted(target, score, weight, pos_label, alias)
    else:
        result = _gini_unweighted(target, score, pos_label, alias)
    # With pos_label the target is a class label (compared via ==), so check it for
    # nulls only; with pos_label=None it is a numeric magnitude, so check NaN too.
    if pos_label is None:
        guard = guarded(result, values=[target, score], weight=weight)
    else:
        guard = guarded(result, values=[score], labels=[target], weight=weight)
    return guard.alias(alias)


def _gini_unweighted(
    target: IntoExpr, score: IntoExpr, pos_label: PosLabel | None, alias: str
) -> pl.Expr:
    """Rank-based normalized Gini for the unweighted case."""
    target_float = _gini_target(target, pos_label)
    score_col = col_expr(score)

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


def _gini_weighted(
    target: IntoExpr,
    score: IntoExpr,
    weight: str | pl.Expr,
    pos_label: PosLabel | None,
    alias: str,
) -> pl.Expr:
    """Weighted normalized Gini via Lorenz curve areas."""
    target_float = _gini_target(target, pos_label)
    weight_float = weight_expr(weight)
    score_col = col_expr(score).cast(pl.Float64)

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
