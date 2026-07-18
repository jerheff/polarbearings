"""ROC AUC metric implemented as a Polars expression."""

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


def roc_auc(
    target: IntoExpr, score: IntoExpr, *, weight: WeightInput = None, pos_label: PosLabel = 1
) -> pl.Expr:
    """Compute ROC AUC score for binary classification as a Polars expression.

    Uses the Mann-Whitney U statistic for unweighted data (fast, exact).
    Uses the trapezoidal rule on the weighted ROC curve when sample weights
    are provided.

    Args:
        target: Column name or expression containing class labels.
        score: Column name or expression containing prediction scores (higher = more likely positive).
        weight: Optional name of the column containing sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).

    Returns:
        A Polars expression that computes the ROC AUC score.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import roc_auc
        >>>
        >>> df = pl.DataFrame({
        ...     "label": [0, 0, 1, 1],
        ...     "score": [0.1, 0.2, 0.8, 0.9]
        ... })
        >>> df.select(roc_auc("label", "score"))
        shape: (1, 1)
        ┌─────────────────────┐
        │ roc_auc_label_score │
        │ ---                 │
        │ f64                 │
        ╞═════════════════════╡
        │ 1.0                 │
        └─────────────────────┘

    Notes:
        - Returns null when only one class is present (ROC AUC is undefined).
        - Returns 0.5 when all scores are identical.
        - Handles tied scores correctly.
    """
    alias = f"roc_auc_{col_name(target)}_{col_name(score)}"
    alias += weight_suffix(weight)
    if pos_label != 1:
        alias += f"_pos{pos_label}"

    if weight is not None:
        return _roc_auc_weighted(target, score, weight, pos_label, alias)
    return _roc_auc_unweighted(target, score, pos_label, alias)


def _roc_auc_unweighted(
    target: IntoExpr, score: IntoExpr, pos_label: PosLabel, alias: str
) -> pl.Expr:
    """Mann-Whitney U statistic approach (fast, no weights)."""
    is_pos = col_expr(target) == pos_label

    # Cast counts to UInt64 before any multiplication: a boolean `sum()` returns
    # UInt32, and the products below (``total_pos * total_neg`` and
    # ``total_pos * (total_pos + 1)``) overflow UInt32 once total_pos exceeds
    # ~65k rows (i.e. n > ~131k), silently corrupting the AUC.
    total_pos = is_pos.sum().cast(pl.UInt64)
    total_neg = col_expr(target).len() - total_pos
    single_class = (total_pos == 0) | (total_neg == 0)

    # All scores identical -> AUC is 0.5. Use max == min rather than var() == 0:
    # variance squares the deviations, which underflows to 0.0 for tiny-magnitude
    # but distinct scores (e.g. 5e-303 vs 0.0), falsely reporting a tie.
    tie_cond = col_expr(score).max() == col_expr(score).min()

    ranks = col_expr(score).rank(method="average")
    pos_rank_sum = (ranks * is_pos).sum()
    min_pos_rank_sum = total_pos * (total_pos + 1) / 2
    u_statistic = pos_rank_sum - min_pos_rank_sum
    auc = u_statistic / (total_pos * total_neg)

    result = pl.when(single_class).then(None).when(tie_cond).then(pl.lit(0.5)).otherwise(auc)
    return guarded(result, values=[score], labels=[target]).alias(alias)


def _roc_auc_weighted(
    target: IntoExpr, score: IntoExpr, weight: IntoExpr, pos_label: PosLabel, alias: str
) -> pl.Expr:
    """Trapezoidal rule on weighted ROC curve."""
    is_pos = (col_expr(target) == pos_label).cast(pl.Float64)
    neg_indicator = 1 - is_pos
    score_col = col_expr(score)
    weight_col = weight_expr(weight)

    total_wt_pos = (is_pos * weight_col).sum()
    total_wt_neg = (neg_indicator * weight_col).sum()
    single_class = (total_wt_pos == 0) | (total_wt_neg == 0)

    # Sort by score descending
    # NOTE: use explicit sort_by on each derived column rather than
    # arithmetic on sorted expressions because Polars does not propagate
    # sort_by through arithmetic in group_by context.
    sorted_score = score_col.sort(descending=True)

    # Weighted cumulative TP and FP
    cum_wt_tp = (is_pos * weight_col).sort_by(score_col, descending=True).cum_sum()
    cum_wt_fp = (neg_indicator * weight_col).sort_by(score_col, descending=True).cum_sum()

    tpr = cum_wt_tp / total_wt_pos
    fpr = cum_wt_fp / total_wt_neg

    # Handle ties: use boundary values (last occurrence of each unique score)
    is_boundary = sorted_score != sorted_score.shift(-1)
    is_boundary = is_boundary.fill_null(True)

    tpr_adj = pl.when(is_boundary).then(tpr).otherwise(None).backward_fill()
    fpr_adj = pl.when(is_boundary).then(fpr).otherwise(None).backward_fill()

    # Trapezoidal rule: AUC = Σ (Δfpr × average_tpr)
    delta_fpr = fpr_adj - fpr_adj.shift(1).fill_null(0)
    avg_tpr = (tpr_adj + tpr_adj.shift(1).fill_null(0)) / 2

    auc = (delta_fpr * avg_tpr).sum()

    result = pl.when(single_class).then(None).otherwise(auc)
    return guarded(result, values=[score], labels=[target], weight=weight).alias(alias)
