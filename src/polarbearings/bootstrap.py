"""Bootstrap confidence intervals for any metric.

A bootstrap replicate reweights the rows; because every polarbearings metric accepts
a ``weight`` expression, B replicates are just the metric evaluated against B
random weight vectors. This module uses the **Bayesian bootstrap**: each row's
weight is an ``Exp(1)`` draw (``-log(u)``), generated *in-engine* from a per-row
hash so the distribution stays a single composable expression — it runs inside
``select`` and ``group_by().agg()``. Any existing sample weights are folded in
multiplicatively (``base_weight * Exp(1)``).

Interval methods (``method=``):
    - ``"percentile"`` — raw empirical quantiles (default).
    - ``"basic"`` — reverse-percentile, reflects quantiles through the estimate.
    - ``"normal"`` — ``estimate ± z * bootstrap_std`` (assumes normality).
    - ``"bc"`` — bias-corrected; shifts the quantile levels to remove the median
      bias of the bootstrap distribution. Whole-frame only (see below).

Entry points:
    - :func:`bootstrap` returns the full distribution as a ``List[f64]`` column
      (composable; works in ``select`` and ``group_by().agg()``).
    - :func:`bootstrap_ci` computes a whole-frame ``{estimate, low, high}`` CI. It
      takes the frame, materializes the (small) distribution, and reduces it in
      Python — so all four methods, including ``bc``, are exact and cheap.
    - :func:`ci_from_distribution` derives ``{low, high}`` from an already
      *materialized* distribution column (the per-group two-step). Supports
      ``percentile``/``basic``/``normal``; ``bc`` is whole-frame only because a
      per-group bias correction can't be expressed without re-materialization.
"""

import functools
from collections.abc import Callable, Sequence
from statistics import NormalDist, stdev
from typing import Literal, TypedDict, get_args, overload

import polars as pl

from polarbearings._common import WeightInput, resolve_weight


@functools.cache
def _supports_list_agg() -> bool:
    """Whether Polars can reduce a list built from aggregations in a single pass.

    Fixed in Polars 1.28.0 (pola-rs/polars#22249, "Support literal:list agg"). On
    older Polars, reducing a freshly built list inside ``group_by().agg()`` — or a
    single fused lazy plan — raises ``invalid series dtype: expected List`` /
    ``failed to determine supertype of list[f64] and f64``, so the per-group CI
    must insert a materialization boundary between generating the distribution and
    reducing it. Feature-detected (not version-parsed) to be robust to dev builds.
    """
    try:
        probe = pl.DataFrame({"_g": [0, 0], "_x": [1.0, 2.0]})
        probe.group_by("_g").agg(
            pl.concat_list([pl.col("_x").sum(), pl.col("_x").mean()]).list.max()
        )
    except Exception:  # pragma: no cover - this outcome runs on Polars < 1.28 (floor/mid CI)
        return False
    return True  # pragma: no cover - this outcome runs on Polars >= 1.28 (latest CI; #22249)


# Divisor mapping a UInt64 hash into a (0, 1] uniform.
_U64_SCALE = 2.0**64

# Metrics with no ``weight`` parameter cannot use the weighted bootstrap.
_NON_WEIGHTABLE = frozenset({"max_error", "median_absolute_error"})

_Method = Literal["percentile", "basic", "normal", "bc"]
_VALID_METHODS = get_args(_Method)


class BootstrapCI(TypedDict):
    """The whole-frame interval returned by :func:`bootstrap_ci`."""

    estimate: float
    low: float
    high: float


_PPF_CLAMP = 1e-12


def _metric_name(metric: Callable[..., pl.Expr]) -> str:
    """Return a display name for a metric callable (handles functools.partial)."""
    name = getattr(metric, "__name__", None)
    if name is not None:
        return name
    func = getattr(metric, "func", None)  # functools.partial
    return getattr(func, "__name__", "metric")


def _boot_weight(base: WeightInput, seed: int, row_index: str | None = None) -> pl.Expr:
    """Build one replicate's weight: ``Exp(1)`` per row, times any base weight.

    The ``Exp(1)`` draw is ``-log(u)`` with ``u`` a uniform from hashing a per-row
    id with ``seed`` — deterministic given the seed and computed in-engine. The id
    is ``int_range(0, len())`` by default (fine in a whole-frame ``select``), or the
    ``row_index`` column when given. Inside ``group_by`` some Polars versions make
    ``int_range`` per-group-list-valued, which breaks the weight; hashing a
    materialized ``row_index`` column avoids that (the ``by=`` helper sets it).
    """
    idx = pl.int_range(0, pl.len()) if row_index is None else pl.col(row_index)
    u = (idx.hash(seed=seed).cast(pl.Float64) + 1.0) / _U64_SCALE
    boot = -u.log()
    base_expr = resolve_weight(base)
    return boot if base_expr is None else base_expr * boot


def _quantile_sorted(sorted_list: pl.Expr, length: pl.Expr, q: float) -> pl.Expr:
    """Linear-interpolated quantile of a pre-sorted list expression (NumPy 'linear').

    Uses ``list.get`` with an expression index (a scalar in both ``select`` and
    ``group_by`` contexts) and reads the length from the data, so per-group lists of
    differing sizes work.
    """
    n = length.cast(pl.Float64)
    pos = q * (n - 1.0)
    lo = pos.floor().cast(pl.Int64)
    frac = pos - lo.cast(pl.Float64)
    hi = pl.min_horizontal(lo + 1, length.cast(pl.Int64) - 1)
    return sorted_list.list.get(lo) * (1.0 - frac) + sorted_list.list.get(hi) * frac


def bootstrap(
    metric: Callable[..., pl.Expr],
    *cols: str,
    weight: WeightInput = None,
    n_resamples: int = 200,
    seed: int = 0,
    row_index: str | None = None,
    **metric_kwargs: object,
) -> pl.Expr:
    """Bootstrap a metric's sampling distribution as a ``List[f64]`` expression.

    Each of ``n_resamples`` replicates evaluates ``metric`` against fresh Bayesian
    (``Exp(1)``) bootstrap weights; the results are collected into one list-valued
    column. Composes inside ``select`` and ``group_by().agg()``.

    Replicate weights depend only on row position and seed (not the metric), so two
    metrics bootstrapped with the same ``seed`` on the same rows are *paired* — their
    replicates use identical resamples, which is what enables paired comparisons.

    Args:
        metric: A polarbearings metric function (or ``functools.partial`` of one) that
            accepts column names, a ``weight`` argument, and any extra params.
        *cols: Column names forwarded positionally to ``metric``.
        weight: Optional existing sample weights (column name or expression),
            folded in multiplicatively. None for an unweighted base.
        n_resamples: Number of bootstrap replicates (B).
        seed: Base seed; replicate ``b`` uses ``seed + b``.
        row_index: Optional column holding a stable per-row index to hash for the
            weights. Leave None in a whole-frame ``select``; set it (to a column
            added via ``with_row_index``) when bootstrapping inside ``group_by``,
            where ``int_range`` is unstable on some Polars versions. The
            :func:`bootstrap_ci` ``by=`` path sets this automatically.
        **metric_kwargs: Extra keyword arguments forwarded to ``metric`` (e.g.
            ``threshold``, ``beta``, ``pos_label``).

    Returns:
        A Polars expression yielding a ``List[f64]`` of length ``n_resamples``.

    Raises:
        ValueError: If ``metric`` has no ``weight`` parameter (e.g. ``max_error``).

    Examples:
        >>> import polars as pl
        >>> from polarbearings import bootstrap, roc_auc
        >>>
        >>> df = pl.DataFrame({"y": [0, 0, 1, 1], "p": [0.1, 0.4, 0.6, 0.9]})
        >>> dist = df.select(bootstrap(roc_auc, "y", "p", n_resamples=100))
        >>> dist.to_series().dtype
        List(Float64)
    """
    name = _metric_name(metric)
    if name in _NON_WEIGHTABLE:
        msg = (
            f"bootstrap() needs a weightable metric; {name!r} has no weight parameter. "
            "Index-resampling for such metrics is not yet supported."
        )
        raise ValueError(msg)

    reps = [
        metric(*cols, weight=_boot_weight(weight, seed + b, row_index), **metric_kwargs).alias(
            f"_rep{b}"
        )
        for b in range(n_resamples)
    ]
    return pl.concat_list(reps).alias(f"bootstrap_{name}_{'_'.join(cols)}")


def _quantile_py(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated quantile of a sorted Python list (matches NumPy 'linear')."""
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _reduce_ci(
    dist: list[float], estimate: float, level: float, method: _Method
) -> tuple[float, float]:
    """Compute (low, high) from a materialized distribution in pure Python."""
    s = sorted(dist)
    alpha = (1.0 - level) / 2.0
    if method == "percentile":
        return _quantile_py(s, alpha), _quantile_py(s, 1.0 - alpha)
    if method == "basic":
        return 2.0 * estimate - _quantile_py(s, 1.0 - alpha), 2.0 * estimate - _quantile_py(
            s, alpha
        )
    if method == "normal":
        z = NormalDist().inv_cdf(1.0 - alpha)
        se = stdev(dist)
        return estimate - z * se, estimate + z * se
    # method == "bc": bias-corrected
    below = sum(1 for v in dist if v < estimate) / len(dist)
    z0 = NormalDist().inv_cdf(min(max(below, _PPF_CLAMP), 1.0 - _PPF_CLAMP))
    za = NormalDist().inv_cdf(alpha)
    lo_level = NormalDist().cdf(2.0 * z0 + za)
    hi_level = NormalDist().cdf(2.0 * z0 - za)
    return _quantile_py(s, lo_level), _quantile_py(s, hi_level)


def _bootstrap_ci_by(
    data: pl.DataFrame | pl.LazyFrame,
    metric: Callable[..., pl.Expr],
    cols: tuple[str, ...],
    weight: WeightInput,
    n_resamples: int,
    level: float,
    method: _Method,
    by: Sequence[str],
    seed: int,
    metric_kwargs: dict[str, object],
) -> pl.DataFrame:
    """Per-group confidence intervals, returned as a DataFrame.

    On Polars >= 1.28 the distribution is reduced in the same ``agg`` (one fused
    pass); on older Polars it materializes the per-group distribution first, then
    reduces it. Either way the reduction is a single vectorized expression over all
    groups — never a Python per-group loop. The result is identical across versions.
    """
    keys = list(by)
    ridx = "__pb_row_index"
    framed = data.lazy().with_row_index(ridx)
    estimate = metric(*cols, weight=weight, **metric_kwargs)
    # Hash a materialized row index for the weights (int_range is unstable in agg
    # on some Polars versions); see _boot_weight.
    dist = bootstrap(
        metric,
        *cols,
        weight=weight,
        n_resamples=n_resamples,
        seed=seed,
        row_index=ridx,
        **metric_kwargs,
    )
    if _supports_list_agg():  # pragma: no cover - fused path, runs on Polars >= 1.28 (latest CI)
        # Polars >= 1.28: reduce the freshly built distribution in the same agg.
        ci = ci_from_distribution(dist, level=level, method=method, estimate=estimate)
        out = (
            framed.group_by(keys)
            .agg(estimate.alias("estimate"), ci.alias("_ci"))
            .unnest("_ci")
            .collect()
        )
    else:
        # Older Polars: materialization boundary, then reduce the real list column.
        materialized = (
            framed.group_by(keys).agg(estimate.alias("estimate"), dist.alias("_dist")).collect()
        )
        out = (
            materialized.with_columns(
                ci_from_distribution(
                    "_dist", level=level, method=method, estimate="estimate"
                ).alias("_ci")
            )
            .drop("_dist")
            .unnest("_ci")
        )
    return out.sort(keys)


@overload
def bootstrap_ci(
    data: pl.DataFrame | pl.LazyFrame,
    metric: Callable[..., pl.Expr],
    *cols: str,
    weight: WeightInput = ...,
    n_resamples: int = ...,
    level: float = ...,
    method: _Method = ...,
    by: None = ...,
    seed: int = ...,
    **metric_kwargs: object,
) -> BootstrapCI: ...


@overload
def bootstrap_ci(
    data: pl.DataFrame | pl.LazyFrame,
    metric: Callable[..., pl.Expr],
    *cols: str,
    weight: WeightInput = ...,
    n_resamples: int = ...,
    level: float = ...,
    method: _Method = ...,
    by: str | Sequence[str],
    seed: int = ...,
    **metric_kwargs: object,
) -> pl.DataFrame: ...


def bootstrap_ci(
    data: pl.DataFrame | pl.LazyFrame,
    metric: Callable[..., pl.Expr],
    *cols: str,
    weight: WeightInput = None,
    n_resamples: int = 200,
    level: float = 0.95,
    method: _Method = "percentile",
    by: str | Sequence[str] | None = None,
    seed: int = 0,
    **metric_kwargs: object,
) -> BootstrapCI | pl.DataFrame:
    """Compute a whole-frame bootstrap confidence interval for a metric.

    Returns ``{estimate, low, high}``: ``estimate`` is the metric on the observed
    data; ``low``/``high`` are the interval bounds at confidence ``level`` using
    ``method`` (see the module docstring). The distribution is materialized and the
    interval reduced in Python, so all four methods are exact and cheap. For
    **per-group** intervals, compute the distribution inside ``group_by().agg()``
    with :func:`bootstrap` and apply :func:`ci_from_distribution` afterward.

    Args:
        data: The DataFrame or LazyFrame to bootstrap over.
        metric: A polarbearings metric function (or ``functools.partial`` of one).
        *cols: Column names forwarded positionally to ``metric``.
        weight: Optional existing sample weights, folded in multiplicatively.
        n_resamples: Number of bootstrap replicates (B).
        level: Confidence level (e.g. 0.95 for a 95% interval).
        method: Interval method — ``"percentile"``, ``"basic"``, ``"normal"``, or
            ``"bc"`` (bias-corrected). ``"bc"`` is whole-frame only.
        by: Optional grouping column(s). When given, returns one CI per group as a
            DataFrame instead of a single whole-frame dict. The per-group reduction
            is a single vectorized expression over all groups (no Python loop); on
            Polars >= 1.28 it fuses into one ``agg``, on older Polars it materializes
            the per-group distribution first (same result either way).
        seed: Base seed; replicate ``b`` uses ``seed + b``.
        **metric_kwargs: Extra keyword arguments forwarded to ``metric``.

    Returns:
        Without ``by``: a dict ``{"estimate": ..., "low": ..., "high": ...}``.
        With ``by``: a DataFrame with columns ``[*by, estimate, low, high]``.

    Raises:
        ValueError: If ``method`` is not one of the supported methods.
        NotImplementedError: If ``by`` is given with ``method="bc"``.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import bootstrap_ci, roc_auc
        >>>
        >>> df = pl.DataFrame({"y": [0, 0, 1, 1], "p": [0.1, 0.4, 0.6, 0.9]})
        >>> ci = bootstrap_ci(df, roc_auc, "y", "p", n_resamples=200, method="bc")
    """
    if method not in _VALID_METHODS:
        msg = f"method must be one of {_VALID_METHODS}, got {method!r}."
        raise ValueError(msg)
    if by is not None:
        if method == "bc":
            msg = (
                "per-group method='bc' is not yet supported; use 'percentile', 'basic', "
                "or 'normal' per group, or whole-frame bc (by=None)."
            )
            raise NotImplementedError(msg)
        keys: Sequence[str] = (by,) if isinstance(by, str) else tuple(by)
        return _bootstrap_ci_by(
            data, metric, cols, weight, n_resamples, level, method, keys, seed, metric_kwargs
        )
    materialized = (
        data.lazy()
        .select(
            bootstrap(
                metric,
                *cols,
                weight=weight,
                n_resamples=n_resamples,
                seed=seed,
                row_index=None,
                **metric_kwargs,
            ).alias("_dist"),
            metric(*cols, weight=weight, **metric_kwargs).alias("_estimate"),
        )
        .collect()
    )
    dist = materialized.get_column("_dist").item().to_list()
    estimate = float(materialized.get_column("_estimate").item())
    low, high = _reduce_ci(dist, estimate, level, method)
    return {"estimate": estimate, "low": low, "high": high}


def ci_from_distribution(
    distribution: str | pl.Expr,
    *,
    level: float = 0.95,
    method: _Method = "percentile",
    estimate: str | pl.Expr | None = None,
) -> pl.Expr:
    """Build a ``{low, high}`` confidence-interval struct from a bootstrap distribution.

    Use this for the per-group pattern: compute the distribution inside
    ``group_by().agg()`` with :func:`bootstrap`, then derive the interval in a
    follow-up ``with_columns``/``select`` on the materialized list column.

    Args:
        distribution: A ``List[f64]`` column name or expression (e.g. from
            :func:`bootstrap`). Must be a *materialized* column, not a raw
            :func:`bootstrap` expression.
        level: Confidence level (e.g. 0.95 for a 95% interval).
        method: Interval method — ``"percentile"``, ``"basic"``, or ``"normal"``.
            ``"bc"`` is whole-frame only (use :func:`bootstrap_ci`).
        estimate: The point-estimate column/expression. Required for ``"basic"``
            and ``"normal"``.

    Returns:
        A Polars expression yielding a struct ``{low, high}``.

    Raises:
        ValueError: If ``method`` is unsupported, or a method needs ``estimate`` and
            none is given.
        NotImplementedError: If ``method="bc"`` (use :func:`bootstrap_ci`).
    """
    if method not in _VALID_METHODS:
        msg = f"method must be one of {_VALID_METHODS}, got {method!r}."
        raise ValueError(msg)
    if method == "bc":
        msg = (
            "method='bc' is not supported per-group; a per-group bias correction "
            "can't be expressed without re-materialization. Use "
            "bootstrap_ci(df, ..., method='bc') for a whole-frame bias-corrected CI."
        )
        raise NotImplementedError(msg)
    dist = pl.col(distribution) if isinstance(distribution, str) else distribution
    est: pl.Expr | None = None
    if estimate is not None:
        est = pl.col(estimate) if isinstance(estimate, str) else estimate
    if method != "percentile" and est is None:
        msg = f"method={method!r} requires the point estimate; pass `estimate=`."
        raise ValueError(msg)

    alpha = (1.0 - level) / 2.0
    s = dist.list.sort()
    n = dist.list.len()
    if method == "percentile":
        low, high = _quantile_sorted(s, n, alpha), _quantile_sorted(s, n, 1.0 - alpha)
    elif method == "basic":
        assert est is not None
        low = 2.0 * est - _quantile_sorted(s, n, 1.0 - alpha)
        high = 2.0 * est - _quantile_sorted(s, n, alpha)
    else:  # normal
        assert est is not None
        z = NormalDist().inv_cdf(1.0 - alpha)
        se = dist.list.std()
        low, high = est - z * se, est + z * se
    return pl.struct(low.alias("low"), high.alias("high"))
