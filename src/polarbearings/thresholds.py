"""Threshold specifications for sweeping classification metrics.

A *threshold spec* is a small callable that, given the score column name, returns
a list of ``(label, threshold)`` pairs. A threshold may be a plain ``float`` or a
Polars **expression** (e.g. a data-derived quantile), so a spec like
``quantiles(10)`` is computed inside the query graph rather than against a
materialized series — and inside ``group_by().agg(...)`` each group gets its own
thresholds in a single pass.

``threshold_sweep`` accepts a spec, a plain ``list[float]`` (resolved to
fixed-value thresholds with their numeric labels, preserving the original column
names), or a plain ``int`` ``N`` — shorthand for ``quantiles(N)``, i.e. ``N``
data-driven thresholds at evenly spaced score quantiles (per group under
``group_by``). The int form is the smart default: it adapts to the score
distribution rather than assuming a fixed ``[0, 1]`` grid.

Factories evenly space their thresholds on the *interior* of the range
(``i / (n + 1)`` for ``i`` in ``1..n``), avoiding the degenerate endpoints where
every prediction falls on one side.
"""

from collections.abc import Callable
from typing import cast

import polars as pl

from polarbearings._common import IntoExpr, col_expr

# (label, value): ``value`` is a fixed float or an expression evaluated against
# the data when the metric is computed.
ResolvedThreshold = tuple[str, float | pl.Expr]
# A spec maps the score column reference (name or expression) to its thresholds.
ThresholdSpec = Callable[[IntoExpr], list[ResolvedThreshold]]
# What ``threshold_sweep`` accepts for its ``thresholds`` argument. A bare ``int``
# ``N`` is shorthand for ``quantiles(N)``.
ThresholdsLike = int | list[float] | ThresholdSpec


def _interior_fractions(n: int) -> list[float]:
    """Return ``n`` evenly spaced fractions strictly inside ``(0, 1)``."""
    if n < 1:
        raise ValueError("number of thresholds must be >= 1.")
    return [i / (n + 1) for i in range(1, n + 1)]


def quantiles(n: int) -> ThresholdSpec:
    """Spec for ``n`` thresholds at evenly spaced interior quantiles of the score.

    Each threshold is ``pl.col(score).quantile(q)`` for ``q`` in
    ``1/(n+1) .. n/(n+1)`` — computed in-engine, so under ``group_by`` each group
    is thresholded at its own quantiles.

    Args:
        n: Number of thresholds (quantile levels).

    Returns:
        A threshold spec.
    """
    qs = _interior_fractions(n)

    def spec(score: IntoExpr) -> list[ResolvedThreshold]:
        col = col_expr(score)
        # Pin "linear" interpolation explicitly: it matches every other quantile in
        # the library (the calibration bin edges) and the computed cut points of
        # equal_width/linspace, so all threshold specs return comparable interpolated
        # values rather than quantiles() alone snapping to the nearest observed score
        # (Polars' default is "nearest").
        return [(f"q{q:g}", col.quantile(q, interpolation="linear")) for q in qs]

    return spec


def equal_width(n: int) -> ThresholdSpec:
    """Spec for ``n`` thresholds evenly spaced inside the observed ``[min, max]``.

    Each threshold is ``min + f·(max − min)`` for ``f`` in ``1/(n+1) .. n/(n+1)``,
    with ``min``/``max`` taken from the data (per group under ``group_by``).

    Args:
        n: Number of thresholds.

    Returns:
        A threshold spec.
    """
    fracs = _interior_fractions(n)

    def spec(score: IntoExpr) -> list[ResolvedThreshold]:
        col = col_expr(score)
        lo, hi = col.min(), col.max()
        return [(f"ew{f:g}", lo + f * (hi - lo)) for f in fracs]

    return spec


def linspace(n: int, lo: float = 0.0, hi: float = 1.0) -> ThresholdSpec:
    """Spec for ``n`` fixed thresholds evenly spaced inside ``[lo, hi]``.

    Data-free (the values are plain floats), so it needs no scan — a good default
    for probability scores, which already live on ``[0, 1]``.

    Args:
        n: Number of thresholds.
        lo: Lower bound of the (open) interval. Defaults to 0.0.
        hi: Upper bound of the (open) interval. Defaults to 1.0.

    Returns:
        A threshold spec.
    """
    values = [lo + f * (hi - lo) for f in _interior_fractions(n)]

    def spec(_score: IntoExpr) -> list[ResolvedThreshold]:
        return [(f"{v:g}", v) for v in values]

    return spec


def resolve_thresholds(thresholds: ThresholdsLike, score: IntoExpr) -> list[ResolvedThreshold]:
    """Resolve an int, a spec, or a plain ``list[float]`` to ``(label, threshold)`` pairs.

    An ``int`` ``N`` is shorthand for ``quantiles(N)`` — ``N`` data-driven score
    quantiles. A ``list[float]`` becomes fixed-value thresholds labeled by their
    value, so the resulting column names match the pre-spec behavior exactly.

    Args:
        thresholds: A count ``N`` (``quantiles(N)``), a threshold spec, or a list of
            fixed thresholds.
        score: Score/probability column (name or expression) the spec resolves against.

    Returns:
        The resolved ``(label, threshold)`` pairs.

    Raises:
        TypeError: If ``thresholds`` is a ``bool`` (an ``int`` subclass, but not a
            valid threshold count).
    """
    if isinstance(thresholds, bool):
        raise TypeError("`thresholds` must be an int, a list of floats, or a spec, not a bool.")
    if isinstance(thresholds, int):
        return quantiles(thresholds)(score)
    if isinstance(thresholds, list):
        return [(f"{v:g}", v) for v in (float(t) for t in cast("list[float]", thresholds))]
    return thresholds(score)
