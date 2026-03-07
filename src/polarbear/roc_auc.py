"""ROC AUC metric implemented as a Polars expression."""

import polars as pl


def roc_auc(target: str, score: str, weight: str | None = None) -> pl.Expr:
    """Compute ROC AUC score for binary classification as a Polars expression.

    Uses the Mann-Whitney U statistic for unweighted data (fast, exact).
    Uses the trapezoidal rule on the weighted ROC curve when sample weights
    are provided.

    Args:
        target: Name of the column containing binary labels (0 or 1).
        score: Name of the column containing prediction scores (higher = more likely positive).
        weight: Optional name of the column containing sample weights.

    Returns:
        A Polars expression that computes the ROC AUC score.

    Examples:
        >>> import polars as pl
        >>> from polarbear import roc_auc
        >>>
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

    Notes:
        - Returns null when only one class is present (ROC AUC is undefined).
        - Returns 0.5 when all scores are identical.
        - Handles tied scores correctly.
    """
    alias = f"roc_auc_{target}_{score}"
    if weight is not None:
        alias += f"_{weight}"

    if weight is not None:
        return _roc_auc_weighted(target, score, weight, alias)
    return _roc_auc_unweighted(target, score, alias)


def _roc_auc_unweighted(target: str, score: str, alias: str) -> pl.Expr:
    """Mann-Whitney U statistic approach (fast, no weights)."""
    target_float = pl.col(target).cast(pl.Float64)

    total_pos = (target_float == 1).sum()
    total_neg = target_float.len() - total_pos
    single_class = (total_pos == 0) | (total_neg == 0)

    tie_cond = pl.col(score).var() == 0

    ranks = pl.col(score).rank(method="average")
    pos_rank_sum = (ranks * (target_float == 1)).sum()
    min_pos_rank_sum = total_pos * (total_pos + 1) / 2
    u_statistic = pos_rank_sum - min_pos_rank_sum
    auc = u_statistic / (total_pos * total_neg)

    return (
        pl.when(single_class)
        .then(None)
        .when(tie_cond)
        .then(pl.lit(0.5))
        .otherwise(auc)
        .alias(alias)
    )


def _roc_auc_weighted(target: str, score: str, weight: str, alias: str) -> pl.Expr:
    """Trapezoidal rule on weighted ROC curve."""
    target_float = pl.col(target).cast(pl.Float64)
    neg_target_float = 1 - target_float
    score_col = pl.col(score)
    weight_col = pl.col(weight).cast(pl.Float64)

    total_wt_pos = (target_float * weight_col).sum()
    total_wt_neg = (neg_target_float * weight_col).sum()
    single_class = (total_wt_pos == 0) | (total_wt_neg == 0)

    # Sort by score descending
    # NOTE: use explicit sort_by on each derived column rather than
    # arithmetic on sorted expressions because Polars does not propagate
    # sort_by through arithmetic in group_by context.
    sorted_score = score_col.sort(descending=True)

    # Weighted cumulative TP and FP
    cum_wt_tp = (target_float * weight_col).sort_by(score_col, descending=True).cum_sum()
    cum_wt_fp = (neg_target_float * weight_col).sort_by(score_col, descending=True).cum_sum()

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

    return pl.when(single_class).then(None).otherwise(auc).alias(alias)
