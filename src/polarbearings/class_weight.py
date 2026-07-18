"""Balanced sample weights implemented as a Polars expression.

The "balanced" weighting scheme assigns each row a weight inversely proportional
to the frequency of its class, so that every class contributes equally to a
downstream loss or aggregate. It mirrors scikit-learn's
:func:`sklearn.utils.class_weight.compute_sample_weight` with ``class_weight="balanced"``.

The weight for a row belonging to class ``c`` is::

    n_samples / (n_classes * count(c))

where ``n_samples`` is the total number of rows, ``n_classes`` is the number of
distinct classes, and ``count(c)`` is the number of rows in class ``c``.
"""

import polars as pl

from polarbearings._common import IntoExpr, col_expr, col_name


def balanced_sample_weight(target: IntoExpr) -> pl.Expr:
    """Compute per-row balanced sample weights as a Polars expression.

    Returns a Float64 weight for every row matching
    ``sklearn.utils.class_weight.compute_sample_weight("balanced", y)``: a row of
    class ``c`` gets ``n_samples / (n_classes * count(c))``.

    The class frequencies are computed with window expressions over the *current
    context* — the whole frame in a ``select``/``with_columns``. Because the result
    is itself a window expression, Polars does not allow it to be nested directly
    inside ``group_by(...).agg(...)``; to get per-group weights, add it in a
    ``with_columns`` partitioned by the group (e.g. ``pl.len().over([group, target])``
    composed by the caller) or compute per filtered frame.

    **Null labels** are treated as missing, not as their own class: they are
    excluded from ``n_samples`` and ``n_classes`` (so they do not dilute the real
    classes' weights), and a null-label row gets a null weight. This matches the
    library's "missing input never silently changes a number" policy.

    Args:
        target: Column name or expression with the class labels. Integer, string,
            and boolean columns are all supported.

    Returns:
        A Polars expression evaluating to the per-row balanced weight (null for a
        null-label row), aliased ``"balanced_sample_weight_<target>"``.
    """
    label = col_expr(target)
    count_over_class = pl.len().over(label)
    n_classes = label.drop_nulls().n_unique()  # null is not a class
    n_samples = label.count()  # count() excludes nulls, unlike pl.len()

    weight = n_samples / (n_classes * count_over_class)
    weight = pl.when(label.is_null()).then(None).otherwise(weight)
    return weight.cast(pl.Float64).alias(f"balanced_sample_weight_{col_name(target)}")


def balanced_class_weights(series: pl.Series) -> dict[object, float]:
    """Compute the balanced weight for each distinct class.

    Mirrors ``sklearn.utils.class_weight.compute_class_weight("balanced", classes, y)``,
    returning a mapping from class label to its weight
    ``n_samples / (n_classes * count(c))``.

    Null labels are treated as missing (dropped), not as their own class, so they
    neither appear as a key nor dilute the real classes' weights.

    Args:
        series: A Polars Series of class labels. Integer, string, and boolean
            dtypes are all supported.

    Returns:
        A dict mapping each distinct (non-null) class label to its balanced weight.
    """
    labels = series.drop_nulls()
    n_samples = labels.len()
    counts = labels.value_counts()
    label_col, count_col = counts.columns[0], counts.columns[1]
    n_classes = counts.height

    return {
        row[label_col]: n_samples / (n_classes * row[count_col])
        for row in counts.iter_rows(named=True)
    }
