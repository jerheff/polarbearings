"""Diagnostic curves derived from the confusion-matrix cells.

ROC, precision-recall, and detection-error-tradeoff (DET) curves, plus cost
curves, are all column math over the ``threshold, tp, fp, fn, tn`` frame produced
by :func:`~polarbearings.confusion_curve.confusion_curve`. These helpers wrap that
primitive so the most common diagnostic plots are one call returning plot-ready,
tidy data — feed the rate columns straight to Plotly/matplotlib.

Each accepts a ``DataFrame`` or ``LazyFrame`` and **always returns a LazyFrame**
(call ``.collect()`` to materialize). The ``thresholds`` argument is forwarded to
``confusion_curve``: the default (``None``) gives the exact all-thresholds curve,
while a fixed grid — an ``int`` ``N`` (``N`` score quantiles), a spec such as
``quantiles(100)``, or a ``list[float]`` — evaluates the curve at comparable
operating points, useful for monitoring across models. Rows with a missing
score/target/weight are dropped on the curve, matching ``confusion_curve``.
"""

import polars as pl

from polarbearings._common import (
    IntoExpr,
    PosLabel,
    WeightInput,
    by_columns,
    col_name,
)
from polarbearings.confusion_curve import confusion_curve
from polarbearings.thresholds import ThresholdsLike

# Confusion-cell columns shared by every curve below.
_TP, _FP, _FN, _TN = pl.col("tp"), pl.col("fp"), pl.col("fn"), pl.col("tn")
# Cost-curve cell keys, in the column order used for validation messages.
_COST_CELLS = ("tp", "fp", "fn", "tn")


def _rate(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """A rate that is null where its denominator is zero (a degenerate class)."""
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


def roc_curve(
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
    """Receiver-operating-characteristic curve: false vs true positive rate.

    Reports ``fpr = fp / (fp + tn)`` and ``tpr = tp / (tp + fn)`` at each threshold
    of :func:`~polarbearings.confusion_curve.confusion_curve`, in increasing-``fpr``
    order. Mirrors ``sklearn.metrics.roc_curve`` (which keeps all points; this does
    not drop collinear ones). The scalar summary is :func:`~polarbearings.roc_auc`.

    Args:
        frame: ``DataFrame`` or ``LazyFrame`` holding the columns.
        target: Column with class labels.
        score: Column with predicted scores/probabilities.
        thresholds: Optional fixed grid forwarded to ``confusion_curve``; ``None``
            (default) gives the exact all-thresholds curve.
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
        by: Optional column(s) to compute a separate curve per group in one pass.
        endpoints: When True (default), include the trivial ``threshold = +inf``
            origin so the curve starts at ``(0, 0)``. Ignored for a fixed grid.

    Returns:
        A ``LazyFrame`` with columns ``[*by, threshold, fpr, tpr]``. A rate is null
        for a group with no negatives (``fpr``) or no positives (``tpr``). Call
        ``.collect()`` to materialize.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import roc_curve
        >>> df = pl.DataFrame({"y": [0, 1, 1, 0], "score": [0.2, 0.9, 0.6, 0.4]})
        >>> roc_curve(df, "y", "score").collect().columns
        ['threshold', 'fpr', 'tpr']
    """
    by_names = [col_name(b) for b in by_columns(by)]
    cells = confusion_curve(
        frame,
        target,
        score,
        thresholds=thresholds,
        weight=weight,
        pos_label=pos_label,
        by=by,
        endpoints=endpoints,
    )
    return cells.select(
        *by_names,
        "threshold",
        _rate(_FP, _FP + _TN).alias("fpr"),
        _rate(_TP, _TP + _FN).alias("tpr"),
    )


def pr_curve(
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
    """Precision-recall curve.

    Reports ``precision = tp / (tp + fp)`` and ``recall = tp / (tp + fn)`` at each
    threshold of :func:`~polarbearings.confusion_curve.confusion_curve`. Following
    scikit-learn's ``precision_recall_curve`` convention, precision is ``1.0`` where
    nothing is predicted positive (the ``threshold = +inf`` endpoint). The scalar
    summary is :func:`~polarbearings.average_precision`.

    Args:
        frame: ``DataFrame`` or ``LazyFrame`` holding the columns.
        target: Column with class labels.
        score: Column with predicted scores/probabilities.
        thresholds: Optional fixed grid forwarded to ``confusion_curve``; ``None``
            (default) gives the exact all-thresholds curve.
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
        by: Optional column(s) to compute a separate curve per group in one pass.
        endpoints: When True (default), include the ``threshold = +inf`` endpoint
            ``(recall = 0, precision = 1)``. Ignored for a fixed grid.

    Returns:
        A ``LazyFrame`` with columns ``[*by, threshold, precision, recall]``.
        ``recall`` is null for a group with no positives. Call ``.collect()`` to
        materialize.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import pr_curve
        >>> df = pl.DataFrame({"y": [0, 1, 1, 0], "score": [0.2, 0.9, 0.6, 0.4]})
        >>> pr_curve(df, "y", "score").collect().columns
        ['threshold', 'precision', 'recall']
    """
    by_names = [col_name(b) for b in by_columns(by)]
    cells = confusion_curve(
        frame,
        target,
        score,
        thresholds=thresholds,
        weight=weight,
        pos_label=pos_label,
        by=by,
        endpoints=endpoints,
    )
    precision = pl.when(_TP + _FP > 0).then(_TP / (_TP + _FP)).otherwise(1.0)
    return cells.select(
        *by_names,
        "threshold",
        precision.alias("precision"),
        _rate(_TP, _TP + _FN).alias("recall"),
    )


def det_curve(
    frame: pl.DataFrame | pl.LazyFrame,
    target: IntoExpr,
    score: IntoExpr,
    *,
    thresholds: ThresholdsLike | None = None,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
    by: IntoExpr | list[IntoExpr] | None = None,
    endpoints: bool = False,
) -> pl.LazyFrame:
    """Detection-error-tradeoff curve: false-positive vs false-negative rate.

    Reports ``fpr = fp / (fp + tn)`` and ``fnr = fn / (fn + tp)`` at each threshold
    of :func:`~polarbearings.confusion_curve.confusion_curve`. Mirrors
    ``sklearn.metrics.det_curve``, which omits the trivial ``(0, 1)`` / ``(1, 0)``
    endpoints (they sit at infinity on the usual probit axes) — hence ``endpoints``
    defaults to False here.

    Args:
        frame: ``DataFrame`` or ``LazyFrame`` holding the columns.
        target: Column with class labels.
        score: Column with predicted scores/probabilities.
        thresholds: Optional fixed grid forwarded to ``confusion_curve``; ``None``
            (default) gives the exact all-thresholds curve.
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
        by: Optional column(s) to compute a separate curve per group in one pass.
        endpoints: When True, include the ``threshold = +inf`` origin. Defaults to
            False to match scikit-learn. Ignored for a fixed grid.

    Returns:
        A ``LazyFrame`` with columns ``[*by, threshold, fpr, fnr]``. A rate is null
        for a group with no negatives (``fpr``) or no positives (``fnr``). Call
        ``.collect()`` to materialize.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import det_curve
        >>> df = pl.DataFrame({"y": [0, 1, 1, 0], "score": [0.2, 0.9, 0.6, 0.4]})
        >>> det_curve(df, "y", "score").collect().columns
        ['threshold', 'fpr', 'fnr']
    """
    by_names = [col_name(b) for b in by_columns(by)]
    cells = confusion_curve(
        frame,
        target,
        score,
        thresholds=thresholds,
        weight=weight,
        pos_label=pos_label,
        by=by,
        endpoints=endpoints,
    )
    return cells.select(
        *by_names,
        "threshold",
        _rate(_FP, _FP + _TN).alias("fpr"),
        _rate(_FN, _FN + _TP).alias("fnr"),
    )


def expected_cost(
    frame: pl.DataFrame | pl.LazyFrame,
    target: IntoExpr,
    score: IntoExpr,
    costs: dict[str, float],
    *,
    thresholds: ThresholdsLike | None = None,
    normalize: bool = False,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
    by: IntoExpr | list[IntoExpr] | None = None,
    endpoints: bool = True,
) -> pl.LazyFrame:
    """Total decision cost swept across thresholds.

    Applies a per-cell cost (or benefit) to the confusion cells from
    :func:`~polarbearings.confusion_curve.confusion_curve` and sums it at each
    threshold, so the cost-optimal operating point is the ``cost`` argmin. Cells
    absent from ``costs`` cost zero. There is no scikit-learn equivalent.

    Args:
        frame: ``DataFrame`` or ``LazyFrame`` holding the columns.
        target: Column with class labels.
        score: Column with predicted scores/probabilities.
        costs: Mapping of confusion cell (``"tp"``, ``"fp"``, ``"fn"``, ``"tn"``) to
            its per-instance cost. Missing cells default to ``0.0``.
        thresholds: Optional fixed grid forwarded to ``confusion_curve``; ``None``
            (default) gives the exact all-thresholds curve.
        normalize: When True, divide by the number of instances (summed weight when
            weighted) to report expected cost per decision instead of total cost.
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
        by: Optional column(s) to compute a separate curve per group in one pass.
        endpoints: When True (default), include the ``threshold = +inf`` endpoint.
            Ignored for a fixed grid.

    Returns:
        A ``LazyFrame`` with columns ``[*by, threshold, cost]``. Call ``.collect()``
        to materialize.

    Raises:
        ValueError: If ``costs`` contains a key outside ``{tp, fp, fn, tn}``.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import expected_cost
        >>> df = pl.DataFrame({"y": [0, 1, 1, 0], "score": [0.2, 0.9, 0.6, 0.4]})
        >>> expected_cost(df, "y", "score", {"fn": 5.0, "fp": 1.0}).collect().columns
        ['threshold', 'cost']
    """
    unknown = set(costs) - set(_COST_CELLS)
    if unknown:
        raise ValueError(
            f"Unknown cost cell(s) {sorted(unknown)}; use a subset of {list(_COST_CELLS)}."
        )
    by_names = [col_name(b) for b in by_columns(by)]
    cells = confusion_curve(
        frame,
        target,
        score,
        thresholds=thresholds,
        weight=weight,
        pos_label=pos_label,
        by=by,
        endpoints=endpoints,
    )
    total = pl.lit(0.0)
    for cell in _COST_CELLS:
        c = costs.get(cell, 0.0)
        if c:
            total = total + c * pl.col(cell)
    if normalize:
        total = total / (_TP + _FP + _FN + _TN)
    return cells.select(*by_names, "threshold", total.alias("cost"))
