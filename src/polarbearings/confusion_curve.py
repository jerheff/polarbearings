"""Exact confusion-matrix curve over every distinct score threshold.

Where ``threshold_sweep`` evaluates a *fixed grid* of thresholds (N independent
aggregations), ``confusion_curve`` returns the confusion cells at **every distinct
score** via a single sorted cumulative pass — ``O(n log n)``, one row per distinct
score. It is the exact step function underlying ROC and precision-recall (the same
computation scikit-learn's ``roc_curve`` does), and scales to millions of rows
where a per-threshold sweep cannot.

The output schema matches the tidy frame produced by
``threshold_sweep(confusion_matrix, ...)`` — ``threshold, tp, fp, fn, tn`` (plus any
``by`` columns) — so the grid and exact forms are interchangeable downstream.

It accepts a ``DataFrame`` or ``LazyFrame`` and **always returns a LazyFrame**:
call ``.collect()`` to materialize, or compose it directly in a larger lazy query
and let the optimizer plan it all at collect time.
"""

import polars as pl

from polarbearings._common import (
    IntoExpr,
    PosLabel,
    WeightInput,
    col_expr,
    col_name,
    resolve_weight,
    row_has_missing,
)


def confusion_curve(
    frame: pl.DataFrame | pl.LazyFrame,
    target: IntoExpr,
    score: IntoExpr,
    *,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
    by: IntoExpr | list[IntoExpr] | None = None,
    endpoints: bool = True,
) -> pl.LazyFrame:
    """Confusion-matrix cells at every distinct score threshold.

    For each distinct value ``t`` of ``score``, reports the cells obtained by
    predicting positive when ``score >= t`` — identical to ``confusion_matrix`` at
    ``threshold=t``, but computed for all thresholds at once.

    Rows with a missing (null/NaN) ``score``, ``target``, or ``weight`` are dropped
    and the curve is computed on the complete cases — unlike the scalar metrics,
    which return null for any missing input.

    Args:
        frame: ``DataFrame`` or ``LazyFrame`` holding the columns.
        target: Column with class labels.
        score: Column with predicted scores/probabilities (the thresholds are its
            distinct values).
        weight: Optional column with sample weights. When given, the cells are
            summed weights (``Float64``) instead of counts.
        pos_label: Value in ``target`` treated as the positive class (default 1).
        by: Optional column(s) to compute a separate curve per group, all in one
            pass (each group uses its own distinct scores).
        endpoints: When True (default), prepend a trivial ``threshold = +inf`` row
            with ``tp = fp = 0`` so a derived ROC/PR curve starts at the origin.
            Set False for exactly the distinct-score thresholds.

    Returns:
        A ``LazyFrame`` with columns ``[*by, threshold, tp, fp, fn, tn]``, sorted by
        ``by`` then descending ``threshold`` (the curve runs from predicting nothing
        positive to predicting everything positive). Cells are ``Int64`` counts, or
        ``Float64`` summed weights when ``weight`` is given. Call ``.collect()`` to
        materialize.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import confusion_curve
        >>> df = pl.DataFrame({"y": [0, 1, 1, 0], "score": [0.2, 0.9, 0.6, 0.4]})
        >>> confusion_curve(df, "y", "score", endpoints=False).collect().columns
        ['threshold', 'tp', 'fp', 'fn', 'tn']
    """
    # Rows with a missing score/target/weight are dropped — the curve is computed
    # on the complete cases (a NaN-score point has no place on a curve). This
    # differs from the scalar metrics, which return null for any missing input.
    lf = frame.lazy().filter(~row_has_missing(values=[score], labels=[target], weight=weight))
    by_list: list[IntoExpr] = (
        [] if by is None else [by] if isinstance(by, str | pl.Expr) else list(by)
    )
    by_names = [col_name(b) for b in by_list]
    descending = [False] * len(by_names)

    w = resolve_weight(weight)
    cell_dtype = pl.Float64 if w is not None else pl.Int64

    # Normalize every column reference (name or expression) to a stable internal
    # column up front, so the grouping/sorting below works by name either way. The
    # score lands directly as ``threshold``.
    projection: list[pl.Expr] = [
        *(col_expr(b).alias(name) for b, name in zip(by_list, by_names, strict=True)),
        col_expr(score).cast(pl.Float64).alias("threshold"),
        (col_expr(target) == pos_label).alias("_pos"),
    ]
    if w is not None:
        projection.append(w.alias("_w"))
    base = lf.select(projection)

    pos_expr = pl.col("_w").filter(pl.col("_pos")).sum() if w is not None else pl.col("_pos").sum()
    tot_expr = pl.col("_w").sum() if w is not None else pl.len()

    per = (
        base.group_by([*by_names, "threshold"])
        .agg(pos_expr.alias("_p"), tot_expr.alias("_t"))
        .sort([*by_names, "threshold"], descending=[*descending, True])
    )

    neg = pl.col("_t") - pl.col("_p")
    if by_names:
        tp, fp = pl.col("_p").cum_sum().over(by_names), neg.cum_sum().over(by_names)
        total_pos, total_all = (
            pl.col("_p").sum().over(by_names),
            pl.col("_t").sum().over(by_names),
        )
    else:
        tp, fp = pl.col("_p").cum_sum(), neg.cum_sum()
        total_pos, total_all = pl.col("_p").sum(), pl.col("_t").sum()

    # total_pos / total_all are group aggregations; each broadcasts over the
    # cumulative tp / fp in this select, so fn = P - tp and tn = N - fp need no
    # intermediate columns.
    curve = per.select(
        *by_names,
        "threshold",
        tp.cast(cell_dtype).alias("tp"),
        fp.cast(cell_dtype).alias("fp"),
        (total_pos - tp).cast(cell_dtype).alias("fn"),
        ((total_all - total_pos) - fp).cast(cell_dtype).alias("tn"),
    )

    if endpoints:
        totals = (
            base.group_by(by_names).agg(pos_expr.alias("_P"), tot_expr.alias("_All"))
            if by_names
            else base.select(pos_expr.alias("_P"), tot_expr.alias("_All"))
        )
        origin = totals.select(
            *by_names,
            pl.lit(float("inf")).alias("threshold"),
            pl.lit(0).cast(cell_dtype).alias("tp"),
            pl.lit(0).cast(cell_dtype).alias("fp"),
            pl.col("_P").cast(cell_dtype).alias("fn"),
            (pl.col("_All") - pl.col("_P")).cast(cell_dtype).alias("tn"),
        )
        curve = pl.concat([origin, curve])

    return curve.sort([*by_names, "threshold"], descending=[*descending, True])
