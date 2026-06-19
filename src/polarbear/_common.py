"""Shared helpers for metric implementations."""

import polars as pl

# A sample-weight input: a column name, a Polars expression (e.g. a composed
# weight like ``pl.col("w") * boot_weight``), or ``None`` for the unweighted case.
WeightInput = str | pl.Expr | None


def weight_expr(weight: str | pl.Expr) -> pl.Expr:
    """Resolve a (non-None) weight to a Float64 expression.

    Args:
        weight: A column name or a Polars expression.

    Returns:
        A ``Float64``-cast Polars expression.
    """
    col = pl.col(weight) if isinstance(weight, str) else weight
    return col.cast(pl.Float64)


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
