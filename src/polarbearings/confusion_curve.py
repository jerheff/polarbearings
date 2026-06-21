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
    by_columns,
    col_expr,
    col_name,
    resolve_weight,
    row_has_missing,
)
from polarbearings.classification import _confusion_components
from polarbearings.thresholds import ResolvedThreshold, ThresholdsLike, resolve_thresholds

# Grid size at/above which the whole-frame ``thresholds=`` path computes the exact
# curve once and samples it (``O(n log n + N)``) instead of running N independent
# aggregations (``O(n · N)``). PROVISIONAL: the per-aggregation form has a very
# small, well-parallelized constant, so the practical crossover is sensitive to
# machine state; the value here needs confirming on a thermally-baseline box (the
# laptop it was first tuned on was throttling). Whole-frame only — the grouped
# (``by``) ``join_asof`` emits an unsuppressable "sortedness cannot be checked"
# notice at collect time on newer Polars.
_GRID_EXACT_CUTOVER = 30


def confusion_curve(
    frame: pl.DataFrame | pl.LazyFrame,
    target: IntoExpr,
    score: IntoExpr,
    *,
    thresholds: ThresholdsLike | None = None,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
    by: IntoExpr | list[IntoExpr] | None = None,
    endpoints: bool = True,
) -> pl.LazyFrame:
    """Confusion-matrix cells across score thresholds.

    By default (``thresholds=None``) reports the cells at **every distinct value**
    ``t`` of ``score`` — the exact step function, via a single sorted cumulative
    pass. Pass a fixed ``thresholds`` grid (a ``list[float]`` or a spec such as
    :func:`~polarbearings.thresholds.quantiles`) to instead evaluate the cells at
    exactly those operating points, one aggregation per threshold. Either way each
    row is the confusion matrix obtained by predicting positive when
    ``score >= t`` — identical to ``confusion_matrix`` at ``threshold=t``.

    Rows with a missing (null/NaN) ``score``, ``target``, or ``weight`` are dropped
    and the curve is computed on the complete cases — unlike the scalar metrics,
    which return null for any missing input.

    Args:
        frame: ``DataFrame`` or ``LazyFrame`` holding the columns.
        target: Column with class labels.
        score: Column with predicted scores/probabilities (when ``thresholds`` is
            ``None`` the thresholds are its distinct values).
        thresholds: Optional fixed grid — an ``int`` ``N`` (``N`` data-driven score
            quantiles), a threshold spec (e.g. ``quantiles(100)``,
            ``equal_width(50)``, ``linspace(101)``), or a ``list[float]``. When
            given, the cells are evaluated at exactly these operating points (each
            group at its own, for an int or a data-derived spec) instead of at
            every distinct score, and the ``endpoints`` origin row is not added.
        weight: Optional column with sample weights. When given, the cells are
            summed weights (``Float64``) instead of counts.
        pos_label: Value in ``target`` treated as the positive class (default 1).
        by: Optional column(s) to compute a separate curve per group, all in one
            pass (each group uses its own distinct scores).
        endpoints: When True (default), prepend a trivial ``threshold = +inf`` row
            with ``tp = fp = 0`` so a derived ROC/PR curve starts at the origin.
            Set False for exactly the distinct-score thresholds. Ignored when an
            explicit ``thresholds`` grid is given.

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
    by_list = by_columns(by)
    by_names = [col_name(b) for b in by_list]

    if thresholds is not None:
        return _grid_curve(lf, target, score, thresholds, weight, pos_label, by_list, by_names)
    return _exact_curve(
        lf, target, score, weight, pos_label, by_list, by_names, endpoints=endpoints
    )


def _exact_curve(
    lf: pl.LazyFrame,
    target: IntoExpr,
    score: IntoExpr,
    weight: WeightInput,
    pos_label: PosLabel,
    by_list: list[IntoExpr],
    by_names: list[str],
    *,
    endpoints: bool,
) -> pl.LazyFrame:
    """Confusion cells at every distinct score via one sorted cumulative pass.

    The default (``thresholds=None``) path of :func:`confusion_curve`; also the
    backbone of the large-grid fast path in :func:`_grid_via_exact`.
    """
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


def _grid_curve(
    lf: pl.LazyFrame,
    target: IntoExpr,
    score: IntoExpr,
    thresholds: ThresholdsLike,
    weight: WeightInput,
    pos_label: PosLabel,
    by_list: list[IntoExpr],
    by_names: list[str],
) -> pl.LazyFrame:
    """Confusion cells at a fixed threshold grid (the ``thresholds=`` path).

    Stays fully lazy and scans the data **once**: each cell is aggregated into a
    length-``N`` list (one entry per threshold) in a single ``select`` / ``agg``,
    then a multi-column ``explode`` unpacks the four cell lists and the threshold
    list together into the tidy ``[*by, threshold, tp, fp, fn, tn]`` long form. A
    data-derived spec (e.g. ``quantiles``) resolves inside that one aggregation, so
    under ``by`` every group is thresholded at its own grid.

    The lists must be primitive, not a list of the ``confusion_matrix`` struct: the
    struct form duplicates the aggregation across an ``N``-branch ``UNION`` (a lazy
    ``concat`` of per-threshold sub-plans, re-run once per threshold by projection
    pushdown) and also trips the floor Polars' lazy schema inference for struct
    reductions under ``group_by``. Parallel primitive lists avoid both.
    """
    resolved = resolve_thresholds(thresholds, score)
    # Large whole-frame grids are cheaper computed off the exact curve (see
    # _GRID_EXACT_CUTOVER); small grids and any grouped grid stay on the direct form.
    if not by_names and len(resolved) >= _GRID_EXACT_CUTOVER:
        return _grid_via_exact(lf, target, score, resolved, weight, pos_label)

    w = resolve_weight(weight)
    cell_dtype = pl.Float64 if w is not None else pl.Int64

    thr, tp_l, fp_l, fn_l, tn_l = [], [], [], [], []
    for _label, value in resolved:
        tp, fp, fn, tn = _confusion_components(target, score, value, weight, pos_label)
        thr.append((value if isinstance(value, pl.Expr) else pl.lit(value)).cast(pl.Float64))
        tp_l.append(tp.cast(cell_dtype))
        fp_l.append(fp.cast(cell_dtype))
        fn_l.append(fn.cast(cell_dtype))
        tn_l.append(tn.cast(cell_dtype))

    long_cols = ["threshold", "tp", "fp", "fn", "tn"]
    aggs = [
        pl.concat_list(thr).alias("threshold"),
        pl.concat_list(tp_l).alias("tp"),
        pl.concat_list(fp_l).alias("fp"),
        pl.concat_list(fn_l).alias("fn"),
        pl.concat_list(tn_l).alias("tn"),
    ]
    if by_names:
        keyed = lf.with_columns(
            *(col_expr(b).alias(name) for b, name in zip(by_list, by_names, strict=True))
        )
        wide = keyed.group_by(by_names).agg(*aggs)
    else:
        wide = lf.select(*aggs)

    # Re-cast each unpacked column to its scalar dtype. The explode produces scalar
    # values at runtime, but the floor Polars' lazy schema keeps them ``List``-typed
    # (worse under ``group_by``), which makes a downstream ``.select`` — e.g. a curve
    # wrapper computing ``fp / (fp + tn)`` — panic with a list/scalar supertype error.
    # The cast is a no-op on the data and forces the schema to resolve.
    grid = wide.explode(long_cols).with_columns(
        pl.col("threshold").cast(pl.Float64),
        pl.col("tp").cast(cell_dtype),
        pl.col("fp").cast(cell_dtype),
        pl.col("fn").cast(cell_dtype),
        pl.col("tn").cast(cell_dtype),
    )
    return grid.sort([*by_names, "threshold"], descending=[*([False] * len(by_names)), True])


def _grid_via_exact(
    lf: pl.LazyFrame,
    target: IntoExpr,
    score: IntoExpr,
    resolved: list[ResolvedThreshold],
    weight: WeightInput,
    pos_label: PosLabel,
) -> pl.LazyFrame:
    """Whole-frame grid cells, sampled off the exact curve (the large-N fast path).

    Builds the exact step function once — with the ``+inf`` endpoint as the
    predict-nothing sentinel — then reads the cells at each grid threshold with a
    single forward ``join_asof``. For a threshold ``t``, the forward asof matches the
    smallest distinct score ``>= t``, whose cumulative cells are exactly the
    confusion matrix at ``t`` (predict positive when ``score >= t``); a ``t`` above
    every score matches the ``+inf`` row (predict nothing). Identical results to the
    direct per-threshold form; selected past :data:`_GRID_EXACT_CUTOVER`.

    Whole-frame only: see :data:`_GRID_EXACT_CUTOVER` for why the grouped path is
    excluded. Both operands are sorted by ``threshold`` and explicitly flagged sorted
    so ``join_asof`` neither re-checks nor warns across Polars versions.
    """
    w = resolve_weight(weight)
    cell_dtype = pl.Float64 if w is not None else pl.Int64

    exact = (
        _exact_curve(lf, target, score, weight, pos_label, [], [], endpoints=True)
        .select("threshold", "tp", "fp", "fn", "tn")
        .sort("threshold")
        .with_columns(pl.col("threshold").set_sorted())
    )
    thr_exprs = [(v if isinstance(v, pl.Expr) else pl.lit(v)).cast(pl.Float64) for _, v in resolved]
    grid = (
        lf.select(pl.concat_list(thr_exprs).alias("threshold"))
        .explode("threshold")
        .sort("threshold")
        .with_columns(pl.col("threshold").set_sorted())
    )
    sampled = grid.join_asof(exact, on="threshold", strategy="forward")
    return sampled.select(
        "threshold",
        pl.col("tp").cast(cell_dtype),
        pl.col("fp").cast(cell_dtype),
        pl.col("fn").cast(cell_dtype),
        pl.col("tn").cast(cell_dtype),
    ).sort("threshold", descending=True)
