"""Shared helpers for metric implementations."""

import inspect
from collections.abc import Sequence
from typing import TypeAlias

import polars as pl

# Polars 1.42 deprecates calling ``explode`` without ``empty_as_null``; in Polars 2.0
# the default flips from ``True`` to ``False``. Our reshape sites explode a
# ``concat_list`` whose lists have a fixed length >= 1 — an empty list is structurally
# impossible — so the flag is behaviorally inert. We pin it to the going-forward
# default (``False``) where the parameter exists and omit it on the 1.0.0 floor (where
# it does not), silencing the deprecation without changing behavior on any Polars.
# Splat into any ``explode`` call: ``frame.explode(cols, **EXPLODE_KW)``.
EXPLODE_KW: dict[str, bool] = (
    {"empty_as_null": False}
    if "empty_as_null" in inspect.signature(pl.LazyFrame.explode).parameters
    else {}
)

# A column reference: either a column name or a Polars expression (Polars' own
# ``IntoExpr`` convention). Every parameter that names a column accepts this, so a
# computed column (e.g. ``pl.col("raw").rank()``) can be passed without a prior
# ``with_columns``.
IntoExpr: TypeAlias = str | pl.Expr

# A sample-weight input: a column reference or ``None`` for the unweighted case.
WeightInput: TypeAlias = IntoExpr | None

# A positive-class label: any scalar value comparable to the target column
# (e.g. 1, 100, "cancer", True). Defaults to 1 for backward compatibility.
PosLabel: TypeAlias = int | float | str | bool


def col_expr(col: IntoExpr) -> pl.Expr:
    """Coerce a column reference (name or expression) to an expression.

    Args:
        col: A column name, or a Polars expression to use as-is.

    Returns:
        ``pl.col(col)`` for a name, otherwise ``col`` unchanged.
    """
    return pl.col(col) if isinstance(col, str) else col


def col_name(col: IntoExpr) -> str:
    """Resolve a column reference to a name for building output aliases.

    A string is used verbatim; an expression's name is its root output name, so
    ``pl.col("y").rank()`` yields ``"y"`` and aliases stay readable. For a plain
    string this is the identity, so existing alias strings are unchanged.

    Args:
        col: A column name or a Polars expression.

    Returns:
        The column/output name.
    """
    return col if isinstance(col, str) else col.meta.output_name()


def by_columns(by: IntoExpr | Sequence[IntoExpr] | None) -> list[IntoExpr]:
    """Normalize a ``by=`` grouping argument to a list of column references.

    A single column reference (name or expression) is wrapped in a one-element
    list; ``None`` becomes the empty list; any other sequence is materialized as a
    list. Used by the curve helpers so ``by`` accepts a scalar, a list, or nothing.

    Args:
        by: ``None``, a single column reference, or a sequence of references.

    Returns:
        A list of column references (possibly empty).
    """
    if by is None:
        return []
    if isinstance(by, str | pl.Expr):
        return [by]
    return list(by)


def weight_expr(weight: IntoExpr) -> pl.Expr:
    """Resolve a (non-None) weight to a Float64 expression.

    Args:
        weight: A column name or a Polars expression.

    Returns:
        A ``Float64``-cast Polars expression.
    """
    return col_expr(weight).cast(pl.Float64)


def resolve_weight(weight: WeightInput) -> pl.Expr | None:
    """Return the weight as a Float64 expression, or None when unweighted.

    Args:
        weight: A column name, a Polars expression, or None.

    Returns:
        A ``Float64``-cast Polars expression, or ``None`` when ``weight`` is None.
    """
    return None if weight is None else weight_expr(weight)


def weight_suffix(weight: WeightInput) -> str:
    """Build an alias suffix for a weight argument.

    Args:
        weight: A column name, a Polars expression, or None.

    Returns:
        ``_<col>`` for a named column, ``_w`` for an expression, and ``""`` when
        ``weight`` is None.
    """
    if weight is None:
        return ""
    return f"_{weight}" if isinstance(weight, str) else "_w"


def _value_missing(col: IntoExpr) -> pl.Expr:
    """Missing flag for a NUMERIC value column: null OR NaN.

    The ``cast(Float64, strict=False)`` makes ``is_nan`` valid (and is cheap on a
    genuinely numeric column); ``strict=False`` avoids a hard error on accidental
    non-numeric input, which the metric's own float math would reject anyway.
    """
    e = col_expr(col)
    return e.is_null() | e.cast(pl.Float64, strict=False).is_nan()


def _label_missing(col: IntoExpr) -> pl.Expr:
    """Missing flag for a LABEL column of any dtype: null only.

    A NaN cannot occur in a non-float label, so this skips the float-cast entirely
    — on a 10M-row string column that is ~0.03 ms (read the validity bitmap) versus
    ~54 ms to parse-cast every value hunting for an impossible NaN.

    Uses ``has_nulls()``, which reads Arrow's cached null bitmap; this is ~3-5x
    faster than ``is_null().any()`` for the null check and reveals intent more
    clearly.
    """
    return col_expr(col).has_nulls()


def any_missing(
    values: Sequence[IntoExpr] = (),
    labels: Sequence[IntoExpr] = (),
    weight: WeightInput = None,
) -> pl.Expr:
    """Aggregation expr that is True if any referenced entry is missing.

    ``values`` are numeric columns (null *or* NaN is missing) and ``weight`` is
    treated as one; ``labels`` are class-label columns of any dtype (null only —
    a NaN cannot occur there, so the costly float-cast is skipped). The final
    ``any()`` scopes to the whole frame in a ``select`` and to each group under
    ``group_by().agg()``.

    Args:
        values: Numeric column references (score/prob/pred, regression/continuous
            target) — checked for null and NaN.
        labels: Class-label column references (any dtype) — checked for null only.
        weight: Optional sample-weight column reference (numeric).

    Returns:
        A boolean aggregation expression.
    """
    flags = [_value_missing(c) for c in values]
    flags += [_label_missing(c) for c in labels]
    if weight is not None:
        flags.append(_value_missing(weight))
    combined = flags[0]
    for flag in flags[1:]:
        combined = combined | flag
    return combined.any()


def guarded(
    result: pl.Expr,
    *,
    values: Sequence[IntoExpr] = (),
    labels: Sequence[IntoExpr] = (),
    weight: WeightInput = None,
) -> pl.Expr:
    """Return ``result``, but null within any context that has a missing input.

    Missingness is detected on the *raw* columns (before any ``== pos_label`` or
    ``>= threshold`` comparison can turn a NaN into a real class/prediction), so a
    metric is null for the whole frame — or for just the affected group under
    ``group_by`` — if any score/target/weight entry is null or NaN.
    """
    return pl.when(any_missing(values, labels, weight)).then(None).otherwise(result)


def row_has_missing(
    values: Sequence[IntoExpr] = (),
    labels: Sequence[IntoExpr] = (),
    weight: WeightInput = None,
) -> pl.Expr:
    """Per-row mask: True where a referenced column is missing in that row.

    The row-granularity counterpart of :func:`any_missing`, for *filtering* frames
    — the curve helpers drop incomplete rows rather than nulling a single result.
    Same role split (``values``/``weight`` are null-or-NaN, ``labels`` null-only),
    but evaluated per row, so labels use ``is_null`` rather than the scalar
    ``has_nulls``.
    """
    mask: pl.Expr = pl.lit(value=False)
    for col in values:
        mask = mask | _value_missing(col)
    for col in labels:
        mask = mask | col_expr(col).is_null()
    if weight is not None:
        mask = mask | _value_missing(weight)
    return mask
