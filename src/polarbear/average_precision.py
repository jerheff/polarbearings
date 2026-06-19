"""Average precision metric implemented as a Polars expression."""

import polars as pl

from polarbear._common import WeightInput, resolve_weight, weight_suffix


def average_precision(
    target: str, score: str, weight: WeightInput = None, pos_label: int | float | str | bool = 1
) -> pl.Expr:
    """Compute average precision (non-interpolated) for binary classification.

    Uses the non-interpolated formula: AP = Σ (Rₙ - Rₙ₋₁) × Pₙ, matching
    scikit-learn's average_precision_score. This avoids the trapezoidal rule
    which uses linear interpolation and can be too optimistic.

    Args:
        target: Name of the column containing class labels.
        score: Name of the column containing prediction scores (higher = more likely positive).
        weight: Optional name of the column containing sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).

    Returns:
        A Polars expression that computes the average precision score.

    Examples:
        >>> import polars as pl
        >>> from polarbear import average_precision
        >>>
        >>> df = pl.DataFrame({
        ...     "label": [0, 0, 1, 1],
        ...     "score": [0.1, 0.4, 0.35, 0.8]
        ... })
        >>> df.select(average_precision("label", "score"))

    Notes:
        - Returns null when no positive examples exist (AP is undefined).
        - Handles tied scores correctly by grouping them at the same threshold.
        - Higher is better (1.0 is perfect).
    """
    target_float = (pl.col(target) == pos_label).cast(pl.Float64)
    score_col = pl.col(score)

    # Sort by score descending
    # NOTE: use explicit sort_by on each derived column rather than
    # arithmetic on sorted expressions (e.g. `1 - sorted_target`) because
    # Polars does not propagate sort_by through arithmetic in group_by context.
    sorted_target = target_float.sort_by(score_col, descending=True)
    sorted_neg_target = (1 - target_float).sort_by(score_col, descending=True)
    sorted_score = score_col.sort(descending=True)

    weight_col = resolve_weight(weight)
    if weight_col is not None:
        sorted_weight = weight_col.sort_by(score_col, descending=True)
        total_pos = (target_float * weight_col).sum()
        sorted_tp_weight = (target_float * weight_col).sort_by(score_col, descending=True)
        cum_tp = sorted_tp_weight.cum_sum()
        cum_total = sorted_weight.cum_sum()
        delta_recall = sorted_tp_weight / total_pos
    else:
        total_pos = target_float.sum()
        cum_tp = sorted_target.cum_sum()
        cum_fp = sorted_neg_target.cum_sum()
        cum_total = cum_tp + cum_fp
        delta_recall = sorted_target / total_pos

    no_positives = total_pos == 0

    # Handle ties: at each unique score threshold, use the precision computed
    # at the LAST occurrence of that score (all tied samples are predicted
    # positive together). Backward-fill assigns boundary values to all rows
    # in a tie group.
    is_boundary = sorted_score != sorted_score.shift(-1)
    is_boundary = is_boundary.fill_null(True)

    cum_tp_adj = pl.when(is_boundary).then(cum_tp).otherwise(None).backward_fill()
    cum_total_adj = pl.when(is_boundary).then(cum_total).otherwise(None).backward_fill()

    prec = cum_tp_adj / cum_total_adj

    # AP = Σ (precision × delta_recall)
    # delta_recall is nonzero only where target == 1, so the sum naturally
    # weights each threshold's precision by the recall gained there.
    ap = (prec * delta_recall).sum()

    alias = f"average_precision_{target}_{score}"
    alias += weight_suffix(weight)
    if pos_label != 1:
        alias += f"_pos{pos_label}"

    return pl.when(no_positives).then(None).otherwise(ap).alias(alias)
