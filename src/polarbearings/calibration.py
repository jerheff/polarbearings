"""Calibration curve (reliability-diagram) data as a tidy Polars frame.

A calibration curve bins predictions by their predicted probability and compares,
per bin, the mean predicted probability against the observed positive fraction. A
well-calibrated model sits on the diagonal. The returned frame is plot-ready: feed
``prob_pred`` (x) and ``prob_true`` (y) straight to Plotly/matplotlib, with
``count`` for bin weighting and ``bin_lower``/``bin_upper`` for tooltips.

Mirrors scikit-learn's ``calibration_curve`` for the ``"uniform"`` and
``"quantile"`` strategies, and additionally accepts explicit bin edges — useful
for fixed, comparable bins across models or for domain-specific score bands.

It accepts a ``DataFrame`` or ``LazyFrame`` and **always returns a LazyFrame**
(call ``.collect()`` to materialize). The only step that is not deferred is
computing ``"quantile"`` bin edges, which needs concrete edge values to build the
binning expression and so triggers a scoped collect of those quantiles (the
``"uniform"`` and explicit-``bins`` strategies add no collect of their own).
"""

from typing import Literal

import polars as pl

from polarbearings._common import (
    IntoExpr,
    PosLabel,
    WeightInput,
    by_columns,
    col_expr,
    col_name,
    guarded,
    resolve_weight,
    row_has_missing,
    weight_suffix,
)

BinStrategy = Literal["uniform", "quantile"]


def _bin_edges(
    lf: pl.LazyFrame,
    prob: IntoExpr,
    n_bins: int,
    strategy: BinStrategy,
    bins: list[float] | None,
) -> list[float]:
    """Resolve the monotonic bin edges (length ``n_bins + 1``)."""
    if bins is not None:
        edges = sorted(float(b) for b in bins)
        if len(edges) < 2:
            raise ValueError("`bins` must contain at least two edges.")
        return edges
    if n_bins < 1:
        raise ValueError("`n_bins` must be >= 1.")
    if strategy == "uniform":
        return [i / n_bins for i in range(n_bins + 1)]
    if strategy == "quantile":
        qs = [i / n_bins for i in range(n_bins + 1)]
        prob_f = col_expr(prob).cast(pl.Float64)
        row = (
            lf.select(
                [prob_f.quantile(q, interpolation="linear").alias(str(i)) for i, q in enumerate(qs)]
            )
            .collect()
            .row(0)
        )
        if any(v is None for v in row):
            raise ValueError("Cannot compute quantile bins on an empty column.")
        return [float(v) for v in row]
    raise ValueError(f"Unknown strategy {strategy!r}; use 'uniform' or 'quantile'.")


def calibration_curve(
    frame: pl.DataFrame | pl.LazyFrame,
    target: IntoExpr,
    prob: IntoExpr,
    *,
    n_bins: int = 5,
    strategy: BinStrategy = "uniform",
    bins: list[float] | None = None,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
    by: IntoExpr | list[IntoExpr] | None = None,
) -> pl.LazyFrame:
    """Compute calibration-curve data, one row per non-empty bin.

    Predictions are assigned to bins by ``prob`` (using the same left-edge
    convention as scikit-learn), then each bin reports its mean predicted
    probability and observed positive fraction. Rows with a missing (null/NaN)
    ``prob``, ``target``, or ``weight`` are dropped and the curve is computed on
    the complete cases — unlike the scalar metrics, which null on any missing.

    Args:
        frame: ``DataFrame`` or ``LazyFrame`` holding the columns.
        target: Column with class labels.
        prob: Column with predicted probabilities in ``[0, 1]``.
        n_bins: Number of bins (ignored when ``bins`` is given). Defaults to 5.
        strategy: ``"uniform"`` for equal-width bins over ``[0, 1]`` or
            ``"quantile"`` for equal-frequency bins. Ignored when ``bins`` is given.
        bins: Explicit, monotonic bin edges (length ``k + 1`` for ``k`` bins).
            Overrides ``n_bins``/``strategy`` — has no scikit-learn equivalent.
        weight: Optional column with sample weights. When given, ``count`` is the
            summed weight per bin and the means are weighted.
        pos_label: Value in ``target`` treated as the positive class (default 1).
        by: Optional column(s) for a separate curve per group, all in one pass. The
            bin edges are shared across groups (computed over the whole frame, so a
            ``"quantile"`` strategy uses whole-frame quantiles) — this keeps the
            bins comparable across groups, which is the point of segmenting.

    Returns:
        A ``LazyFrame`` with columns ``[*by, bin, bin_lower, bin_upper, count,
        prob_pred, prob_true]`` — ``bin`` 0-indexed, ``prob_pred`` the mean
        predicted probability, ``prob_true`` the observed positive fraction —
        sorted by ``by`` then ``bin``. Empty bins are omitted. Call ``.collect()``
        to materialize.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import calibration_curve
        >>> df = pl.DataFrame({
        ...     "y": [0, 0, 1, 0, 1, 1, 1, 1],
        ...     "p": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        ... })
        >>> calibration_curve(df, "y", "p", n_bins=2).collect().columns
        ['bin', 'bin_lower', 'bin_upper', 'count', 'prob_pred', 'prob_true']
    """
    # Drop rows with a missing prob/target/weight, then bin the complete cases
    # (so quantile edges are computed on clean data). Differs from the scalar
    # metrics, which null the whole result on any missing input.
    lf = frame.lazy().filter(~row_has_missing(values=[prob], labels=[target], weight=weight))
    edges = _bin_edges(lf, prob, n_bins, strategy, bins)
    n = len(edges) - 1
    interior = edges[1:-1]

    by_list = by_columns(by)
    by_names = [col_name(b) for b in by_list]
    by_proj = [col_expr(b).alias(name) for b, name in zip(by_list, by_names, strict=True)]

    prob_f = col_expr(prob).cast(pl.Float64)
    # bin id = number of interior edges strictly below the prediction; this
    # reproduces numpy's ``searchsorted(edges[1:-1], p, side="left")`` exactly.
    binid: pl.Expr = pl.lit(0, dtype=pl.Int64)
    for e in interior:
        binid = binid + (prob_f > e).cast(pl.Int64)
    is_pos = (col_expr(target) == pos_label).cast(pl.Float64)

    w = resolve_weight(weight)
    if w is None:
        agg = (
            lf.select(*by_proj, binid.alias("bin"), prob_f.alias("p"), is_pos.alias("t"))
            .group_by([*by_names, "bin"])
            .agg(
                pl.len().alias("count"),
                pl.col("p").mean().alias("prob_pred"),
                pl.col("t").mean().alias("prob_true"),
            )
        )
    else:
        agg = (
            lf.select(
                *by_proj, binid.alias("bin"), prob_f.alias("p"), is_pos.alias("t"), w.alias("w")
            )
            .group_by([*by_names, "bin"])
            .agg(
                pl.col("w").sum().alias("count"),
                (pl.col("p") * pl.col("w")).sum().alias("ps"),
                (pl.col("t") * pl.col("w")).sum().alias("ts"),
            )
            .with_columns(
                (pl.col("ps") / pl.col("count")).alias("prob_pred"),
                (pl.col("ts") / pl.col("count")).alias("prob_true"),
            )
            .drop("ps", "ts")
        )

    edge_df = pl.LazyFrame(
        {
            "bin": list(range(n)),
            "bin_lower": edges[:-1],
            "bin_upper": edges[1:],
        },
        schema={"bin": pl.Int64, "bin_lower": pl.Float64, "bin_upper": pl.Float64},
    )
    return (
        agg.join(edge_df, on="bin", how="left")
        .sort([*by_names, "bin"])
        .select(*by_names, "bin", "bin_lower", "bin_upper", "count", "prob_pred", "prob_true")
    )


def _calibration_gap_terms(
    target: IntoExpr,
    prob: IntoExpr,
    n_bins: int,
    strategy: BinStrategy,
    bins: list[float] | None,
    weight: WeightInput,
    pos_label: PosLabel,
) -> tuple[list[pl.Expr], list[pl.Expr], pl.Expr]:
    """Per-bin ``(effective count, |mean_pred - mean_true|)`` terms, plus the total.

    Bins ``prob`` with the same left-edge convention as :func:`calibration_curve`,
    then for each bin returns its summed weight (or count) and the absolute
    calibration gap. Uses per-bin ``filter`` aggregations rather than a window over
    the bin id, so the result composes inside ``group_by().agg()`` on every supported
    Polars version (windows-in-aggregation are rejected on the floor).
    """
    prob_f = col_expr(prob).cast(pl.Float64)
    is_pos = (col_expr(target) == pos_label).cast(pl.Float64)

    # Interior bin edges — fixed floats, or per-group quantile expressions.
    if bins is not None:
        edges = sorted(float(b) for b in bins)
        if len(edges) < 2:
            raise ValueError("`bins` must contain at least two edges.")
        n_used = len(edges) - 1
        interior: list[float | pl.Expr] = list(edges[1:-1])
    elif n_bins < 1:
        raise ValueError("`n_bins` must be >= 1.")
    elif strategy == "uniform":
        n_used = n_bins
        interior = [i / n_bins for i in range(1, n_bins)]
    elif strategy == "quantile":
        n_used = n_bins
        interior = [prob_f.quantile(i / n_bins, interpolation="linear") for i in range(1, n_bins)]
    else:
        raise ValueError(f"Unknown strategy {strategy!r}; use 'uniform' or 'quantile'.")

    # bin id = number of interior edges strictly below the prediction.
    binid: pl.Expr = pl.lit(0, dtype=pl.Int64)
    for edge in interior:
        binid = binid + (prob_f > edge).cast(pl.Int64)

    w = resolve_weight(weight)
    counts: list[pl.Expr] = []
    gaps: list[pl.Expr] = []
    for b in range(n_used):
        mask = binid == b
        if w is None:
            count = mask.sum()
            pred = prob_f.filter(mask).mean()
            true = is_pos.filter(mask).mean()
        else:
            count = w.filter(mask).sum()
            pred = (prob_f * w).filter(mask).sum() / count
            true = (is_pos * w).filter(mask).sum() / count
        counts.append(count)
        gaps.append((pred - true).abs())
    total = pl.len() if w is None else w.sum()
    return counts, gaps, total


def _calibration_alias(
    name: str, target: IntoExpr, prob: IntoExpr, weight: WeightInput, pos_label: PosLabel
) -> str:
    """Build the output-column alias for a calibration-error metric."""
    alias = f"{name}_{col_name(target)}_{col_name(prob)}{weight_suffix(weight)}"
    if pos_label != 1:
        alias += f"_pos{pos_label}"
    return alias


def expected_calibration_error(
    target: IntoExpr,
    prob: IntoExpr,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    bins: list[float] | None = None,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
) -> pl.Expr:
    """Expected Calibration Error (ECE) as a Polars expression.

    Bins predictions by ``prob`` and returns the count-weighted average absolute gap
    between each bin's mean predicted probability and its observed positive fraction:
    ``sum_b (n_b / N) * |pred_b - true_b|`` (0 is perfectly calibrated). It is a plain
    expression, so it drops into ``select`` and ``group_by().agg()`` alongside any
    other metric, on every supported Polars version.

    Rows with a missing (null/NaN) ``prob``, ``target``, or ``weight`` make the whole
    result null — like the other scalar metrics, and unlike :func:`calibration_curve`,
    which drops incomplete rows.

    Args:
        target: Column with class labels.
        prob: Column with predicted probabilities in ``[0, 1]``.
        n_bins: Number of bins (ignored when ``bins`` is given). Defaults to 10.
        strategy: ``"uniform"`` (equal-width over ``[0, 1]``) or ``"quantile"``
            (equal-frequency). Ignored when ``bins`` is given.
        bins: Explicit monotonic bin edges (length ``k + 1`` for ``k`` bins),
            overriding ``n_bins``/``strategy``.
        weight: Optional sample-weight column; the bins, per-bin means, and the
            average are all weighted.
        pos_label: Value in ``target`` treated as the positive class (default 1).

    Returns:
        A Polars expression yielding the ECE (null if any input is missing or the
        frame/group is empty).

    Raises:
        ValueError: For an unknown ``strategy``, ``n_bins < 1``, or fewer than two
            ``bins`` edges.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import expected_calibration_error
        >>> df = pl.DataFrame({"y": [0, 0, 1, 1], "p": [0.1, 0.4, 0.6, 0.9]})
        >>> df.select(expected_calibration_error("y", "p", n_bins=2))  # doctest: +SKIP
    """
    counts, gaps, total = _calibration_gap_terms(
        target, prob, n_bins, strategy, bins, weight, pos_label
    )
    terms = [pl.when(c > 0).then(c * g).otherwise(0.0) for c, g in zip(counts, gaps, strict=True)]
    ece = pl.when(total > 0).then(pl.sum_horizontal(terms) / total).otherwise(None)
    alias = _calibration_alias("expected_calibration_error", target, prob, weight, pos_label)
    return guarded(ece, values=[prob], labels=[target], weight=weight).alias(alias)


def maximum_calibration_error(
    target: IntoExpr,
    prob: IntoExpr,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    bins: list[float] | None = None,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
) -> pl.Expr:
    """Maximum Calibration Error (MCE) as a Polars expression.

    The worst bin's absolute calibration gap: ``max_b |pred_b - true_b|`` over the
    non-empty bins (0 is perfectly calibrated). Shares all binning options and
    semantics with :func:`expected_calibration_error`, and likewise composes in
    ``select`` and ``group_by().agg()``.

    Args:
        target: Column with class labels.
        prob: Column with predicted probabilities in ``[0, 1]``.
        n_bins: Number of bins (ignored when ``bins`` is given). Defaults to 10.
        strategy: ``"uniform"`` or ``"quantile"``. Ignored when ``bins`` is given.
        bins: Explicit monotonic bin edges, overriding ``n_bins``/``strategy``.
        weight: Optional sample-weight column.
        pos_label: Value in ``target`` treated as the positive class (default 1).

    Returns:
        A Polars expression yielding the MCE (null if any input is missing or the
        frame/group is empty).

    Raises:
        ValueError: For an unknown ``strategy``, ``n_bins < 1``, or fewer than two
            ``bins`` edges.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import maximum_calibration_error
        >>> df = pl.DataFrame({"y": [0, 0, 1, 1], "p": [0.1, 0.4, 0.6, 0.9]})
        >>> df.select(maximum_calibration_error("y", "p", n_bins=2))  # doctest: +SKIP
    """
    counts, gaps, _ = _calibration_gap_terms(
        target, prob, n_bins, strategy, bins, weight, pos_label
    )
    terms = [pl.when(c > 0).then(g).otherwise(None) for c, g in zip(counts, gaps, strict=True)]
    mce = pl.max_horizontal(terms)
    alias = _calibration_alias("maximum_calibration_error", target, prob, weight, pos_label)
    return guarded(mce, values=[prob], labels=[target], weight=weight).alias(alias)
