"""Regression metrics implemented as Polars expressions."""

import math

import polars as pl

from polarbearings._common import (
    IntoExpr,
    WeightInput,
    col_expr,
    col_name,
    guarded,
    resolve_weight,
    weight_suffix,
)


def _regression_alias(name: str, target: IntoExpr, pred: IntoExpr, weight: WeightInput) -> str:
    alias = f"{name}_{col_name(target)}_{col_name(pred)}"
    alias += weight_suffix(weight)
    return alias


def r2_score(target: IntoExpr, pred: IntoExpr, *, weight: WeightInput = None) -> pl.Expr:
    """Compute the coefficient of determination (R-squared).

    R² = 1 - SS_res / SS_tot, where SS_res = sum((y - pred)²) and
    SS_tot = sum((y - mean(y))²). Returns null on empty data or when
    all target values are identical (SS_tot = 0).

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.
        weight: Optional column with sample weights.
    """
    y = col_expr(target).cast(pl.Float64)
    p = col_expr(pred).cast(pl.Float64)

    w = resolve_weight(weight)
    if w is not None:
        w_sum = w.sum()
        y_mean = (y * w).sum() / w_sum
        ss_res = ((y - p) ** 2 * w).sum()
        ss_tot = ((y - y_mean) ** 2 * w).sum()
    else:
        y_mean = y.mean()
        ss_res = ((y - p) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()

    result = pl.when(ss_tot == 0).then(None).otherwise(1 - ss_res / ss_tot)
    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("r2_score", target, pred, weight)
    )


def mape(target: IntoExpr, pred: IntoExpr, *, weight: WeightInput = None) -> pl.Expr:
    """Compute mean absolute percentage error.

    MAPE = mean(|y - pred| / |y|). Returns null on empty data. Rows where
    target == 0 are excluded from the calculation (division by zero).

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.
        weight: Optional column with sample weights.
    """
    y = col_expr(target).cast(pl.Float64)
    p = col_expr(pred).cast(pl.Float64)
    pct_error = (y - p).abs() / y.abs()

    w = resolve_weight(weight)
    if w is not None:
        # Filter out y==0 rows by setting their weight to null
        w_valid = pl.when(y == 0).then(None).otherwise(w)
        pct_valid = pl.when(y == 0).then(None).otherwise(pct_error)
        result = (pct_valid * w_valid).sum() / w_valid.sum()
    else:
        pct_valid = pl.when(y == 0).then(None).otherwise(pct_error)
        result = pct_valid.mean()

    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("mape", target, pred, weight)
    )


def mae(target: IntoExpr, pred: IntoExpr, *, weight: WeightInput = None) -> pl.Expr:
    """Compute mean absolute error.

    MAE = mean(|target - pred|), or weighted: sum(w·|target - pred|) / sum(w).

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.
        weight: Optional column with sample weights.
    """
    diff = (col_expr(target).cast(pl.Float64) - col_expr(pred).cast(pl.Float64)).abs()

    w = resolve_weight(weight)
    result = (diff * w).sum() / w.sum() if w is not None else diff.mean()

    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("mae", target, pred, weight)
    )


def mse(target: IntoExpr, pred: IntoExpr, *, weight: WeightInput = None) -> pl.Expr:
    """Compute mean squared error.

    MSE = mean((target - pred)²), or weighted: sum(w·(target - pred)²) / sum(w).

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.
        weight: Optional column with sample weights.
    """
    diff_sq = (col_expr(target).cast(pl.Float64) - col_expr(pred).cast(pl.Float64)) ** 2

    w = resolve_weight(weight)
    result = (diff_sq * w).sum() / w.sum() if w is not None else diff_sq.mean()

    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("mse", target, pred, weight)
    )


def rmse(target: IntoExpr, pred: IntoExpr, *, weight: WeightInput = None) -> pl.Expr:
    """Compute root mean squared error.

    RMSE = sqrt(MSE).

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.
        weight: Optional column with sample weights.
    """
    diff_sq = (col_expr(target).cast(pl.Float64) - col_expr(pred).cast(pl.Float64)) ** 2

    w = resolve_weight(weight)
    result = ((diff_sq * w).sum() / w.sum()).sqrt() if w is not None else diff_sq.mean().sqrt()

    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("rmse", target, pred, weight)
    )


def mean_squared_log_error(
    target: IntoExpr, pred: IntoExpr, *, weight: WeightInput = None
) -> pl.Expr:
    """Compute mean squared logarithmic error (MSLE).

    MSLE = mean((log1p(y) - log1p(pred))²), or weighted by sample weights.
    Mirrors scikit-learn's ``mean_squared_log_error``.

    Both ``target`` and ``pred`` must be non-negative. Negative inputs make the
    logarithm undefined; like scikit-learn (which raises), this expression yields
    NaN for those rows (``log1p`` of a value < -1 is NaN), propagating to the
    result rather than raising.

    Args:
        target: Column name or expression with actual values (must be >= 0).
        pred: Column name or expression with predicted values (must be >= 0).
        weight: Optional column with sample weights.

    Example:
        >>> import polars as pl
        >>> from polarbearings import mean_squared_log_error
        >>> df = pl.DataFrame({"y": [3.0, 5.0, 2.5], "pred": [2.5, 5.0, 4.0]})
        >>> df.select(mean_squared_log_error("y", "pred")).to_series()[0]  # doctest: +SKIP
        0.039...
    """
    sq_log_err = (
        col_expr(target).cast(pl.Float64).log1p() - col_expr(pred).cast(pl.Float64).log1p()
    ) ** 2

    w = resolve_weight(weight)
    result = (sq_log_err * w).sum() / w.sum() if w is not None else sq_log_err.mean()

    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("mean_squared_log_error", target, pred, weight)
    )


def root_mean_squared_log_error(
    target: IntoExpr, pred: IntoExpr, *, weight: WeightInput = None
) -> pl.Expr:
    """Compute root mean squared logarithmic error (RMSLE).

    RMSLE = sqrt(MSLE) = sqrt(mean((log1p(y) - log1p(pred))²)). Mirrors
    scikit-learn's ``root_mean_squared_log_error``.

    Both ``target`` and ``pred`` must be non-negative (see
    :func:`mean_squared_log_error` for the negative-input behavior).

    Args:
        target: Column name or expression with actual values (must be >= 0).
        pred: Column name or expression with predicted values (must be >= 0).
        weight: Optional column with sample weights.

    Example:
        >>> import polars as pl
        >>> from polarbearings import root_mean_squared_log_error
        >>> df = pl.DataFrame({"y": [3.0, 5.0, 2.5], "pred": [2.5, 5.0, 4.0]})
        >>> df.select(root_mean_squared_log_error("y", "pred")).to_series()[0]  # doctest: +SKIP
        0.198...
    """
    sq_log_err = (
        col_expr(target).cast(pl.Float64).log1p() - col_expr(pred).cast(pl.Float64).log1p()
    ) ** 2

    w = resolve_weight(weight)
    if w is not None:
        result = ((sq_log_err * w).sum() / w.sum()).sqrt()
    else:
        result = sq_log_err.mean().sqrt()

    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("root_mean_squared_log_error", target, pred, weight)
    )


def max_error(target: IntoExpr, pred: IntoExpr) -> pl.Expr:
    """Compute the maximum residual error.

    max_error = max(|y - pred|). Mirrors scikit-learn's ``max_error``.

    No ``weight`` parameter is provided: the maximum absolute residual is a
    single worst-case observation, and scaling observations by sample weights
    does not change which residual is largest, so a weighted form is undefined
    (scikit-learn's ``max_error`` likewise accepts no ``sample_weight``).

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.

    Example:
        >>> import polars as pl
        >>> from polarbearings import max_error
        >>> df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.5, 5.0]})
        >>> df.select(max_error("y", "pred")).to_series()[0]
        2.0
    """
    abs_err = (col_expr(target).cast(pl.Float64) - col_expr(pred).cast(pl.Float64)).abs()
    return guarded(abs_err.max(), values=[target, pred]).alias(
        _regression_alias("max_error", target, pred, None)
    )


def median_absolute_error(target: IntoExpr, pred: IntoExpr) -> pl.Expr:
    """Compute the median absolute error.

    median_absolute_error = median(|y - pred|). Mirrors scikit-learn's
    ``median_absolute_error`` (unweighted).

    No ``weight`` parameter is provided. scikit-learn's weighted variant uses a
    *weighted percentile* of the absolute residuals, which cannot be expressed
    correctly as a single pure Polars expression (it requires sorting residuals
    and interpolating against the cumulative weight distribution). Rather than
    ship an incorrect weighted median, weighting is intentionally unsupported
    here; use :func:`mae` if you need a weighted central-tendency error.

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.

    Example:
        >>> import polars as pl
        >>> from polarbearings import median_absolute_error
        >>> df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "pred": [1.0, 2.0, 3.0, 7.0]})
        >>> df.select(median_absolute_error("y", "pred")).to_series()[0]
        0.5
    """
    abs_err = (col_expr(target).cast(pl.Float64) - col_expr(pred).cast(pl.Float64)).abs()
    return guarded(abs_err.median(), values=[target, pred]).alias(
        _regression_alias("median_absolute_error", target, pred, None)
    )


def explained_variance_score(
    target: IntoExpr, pred: IntoExpr, *, weight: WeightInput = None
) -> pl.Expr:
    """Compute the explained variance score.

    explained_variance = 1 - Var(y - pred) / Var(y), using population variances
    (ddof = 0). Mirrors scikit-learn's ``explained_variance_score``. Unlike R²,
    it does not penalize a systematic offset in the residuals. Returns null when
    Var(y) = 0 (undefined).

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.
        weight: Optional column with sample weights.

    Example:
        >>> import polars as pl
        >>> from polarbearings import explained_variance_score
        >>> df = pl.DataFrame({"y": [3.0, -0.5, 2.0, 7.0], "pred": [2.5, 0.0, 2.0, 8.0]})
        >>> df.select(explained_variance_score("y", "pred")).to_series()[0]  # doctest: +SKIP
        0.957...
    """
    y = col_expr(target).cast(pl.Float64)
    p = col_expr(pred).cast(pl.Float64)
    diff = y - p

    w = resolve_weight(weight)
    if w is not None:
        w_sum = w.sum()
        diff_mean = (diff * w).sum() / w_sum
        y_mean = (y * w).sum() / w_sum
        var_diff = ((diff - diff_mean) ** 2 * w).sum() / w_sum
        var_y = ((y - y_mean) ** 2 * w).sum() / w_sum
    else:
        diff_mean = diff.mean()
        y_mean = y.mean()
        var_diff = ((diff - diff_mean) ** 2).mean()
        var_y = ((y - y_mean) ** 2).mean()

    result = pl.when(var_y == 0).then(None).otherwise(1 - var_diff / var_y)
    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("explained_variance_score", target, pred, weight)
    )


def mean_pinball_loss(
    target: IntoExpr, pred: IntoExpr, *, alpha: float = 0.5, weight: WeightInput = None
) -> pl.Expr:
    """Compute the mean pinball loss (a.k.a. quantile loss).

    For residual ``e = y - pred``, the per-sample loss is ``alpha * e`` when
    ``e >= 0`` and ``(alpha - 1) * e`` otherwise, then averaged (or weighted).
    Mirrors scikit-learn's ``mean_pinball_loss``. With ``alpha = 0.5`` this is
    half the MAE; ``alpha`` is the target quantile of a quantile regressor.

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted (quantile) values.
        alpha: Target quantile in [0, 1]. Defaults to 0.5.
        weight: Optional column with sample weights.

    Example:
        >>> import polars as pl
        >>> from polarbearings import mean_pinball_loss
        >>> df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.5, 1.5, 3.5]})
        >>> df.select(mean_pinball_loss("y", "pred", alpha=0.5)).to_series()[0]
        0.25
    """
    err = col_expr(target).cast(pl.Float64) - col_expr(pred).cast(pl.Float64)
    loss = pl.when(err >= 0).then(alpha * err).otherwise((alpha - 1) * err)

    w = resolve_weight(weight)
    result = (loss * w).sum() / w.sum() if w is not None else loss.mean()

    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("mean_pinball_loss", target, pred, weight)
    )


def smape(target: IntoExpr, pred: IntoExpr, *, weight: WeightInput = None) -> pl.Expr:
    """Compute the symmetric mean absolute percentage error (sMAPE).

    sMAPE = mean(2 * |y - pred| / (|y| + |pred|)). Has no scikit-learn analog.

    Convention for the 0/0 case: when both ``y`` and ``pred`` are 0 the
    denominator is 0; that sample's contribution is defined to be 0 (a perfect
    prediction of zero), avoiding a division-by-zero blow-up. Output is in
    fractional form (multiply by 100 for a percentage); it lies in [0, 2].

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.
        weight: Optional column with sample weights.

    Example:
        >>> import polars as pl
        >>> from polarbearings import smape
        >>> df = pl.DataFrame({"y": [100.0, 0.0], "pred": [110.0, 0.0]})
        >>> df.select(smape("y", "pred")).to_series()[0]  # doctest: +SKIP
        0.0476...
    """
    y = col_expr(target).cast(pl.Float64)
    p = col_expr(pred).cast(pl.Float64)
    denom = y.abs() + p.abs()
    per_sample = pl.when(denom == 0).then(0.0).otherwise(2.0 * (y - p).abs() / denom)

    w = resolve_weight(weight)
    result = (per_sample * w).sum() / w.sum() if w is not None else per_sample.mean()

    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("smape", target, pred, weight)
    )


def huber_loss(
    target: IntoExpr, pred: IntoExpr, *, delta: float = 1.0, weight: WeightInput = None
) -> pl.Expr:
    """Compute the mean Huber loss.

    For residual ``e = y - pred``, the per-sample loss is ``0.5 * e²`` when
    ``|e| <= delta`` and ``delta * (|e| - 0.5 * delta)`` otherwise, then averaged
    (or weighted). Has no direct scikit-learn metric. Quadratic near zero and
    linear in the tails, so it is more robust to outliers than MSE.

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.
        delta: Transition point between quadratic and linear regions. Defaults to 1.0.
        weight: Optional column with sample weights.

    Example:
        >>> import polars as pl
        >>> from polarbearings import huber_loss
        >>> df = pl.DataFrame({"y": [0.0, 0.0], "pred": [0.5, 3.0]})
        >>> df.select(huber_loss("y", "pred", delta=1.0)).to_series()[0]
        1.3125
    """
    abs_err = (col_expr(target).cast(pl.Float64) - col_expr(pred).cast(pl.Float64)).abs()
    loss = (
        pl.when(abs_err <= delta).then(0.5 * abs_err**2).otherwise(delta * (abs_err - 0.5 * delta))
    )

    w = resolve_weight(weight)
    result = (loss * w).sum() / w.sum() if w is not None else loss.mean()

    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("huber_loss", target, pred, weight)
    )


def log_cosh_loss(target: IntoExpr, pred: IntoExpr, *, weight: WeightInput = None) -> pl.Expr:
    """Compute the mean log-cosh loss.

    log_cosh = mean(log(cosh(pred - y))). Has no scikit-learn metric. Behaves
    like MSE for small residuals and like MAE for large ones, and is smooth
    everywhere.

    Computed via the numerically stable identity
    ``log(cosh(e)) = |e| + log1p(exp(-2|e|)) - log(2)``, which avoids the
    ``cosh`` overflow that a naive ``cosh(e).log()`` hits for large residuals.

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.
        weight: Optional column with sample weights.

    Example:
        >>> import polars as pl
        >>> from polarbearings import log_cosh_loss
        >>> df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        >>> df.select(log_cosh_loss("y", "pred")).to_series()[0]
        0.0
    """
    abs_err = (col_expr(pred).cast(pl.Float64) - col_expr(target).cast(pl.Float64)).abs()
    per_sample = abs_err + (-2.0 * abs_err).exp().log1p() - math.log(2.0)

    w = resolve_weight(weight)
    result = (per_sample * w).sum() / w.sum() if w is not None else per_sample.mean()

    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("log_cosh_loss", target, pred, weight)
    )


def _tweedie_unit_deviance(y: pl.Expr, p: pl.Expr, power: float) -> pl.Expr:
    """Compute the per-sample Tweedie unit deviance for a given power.

    Mirrors the unit-deviance formulas used by scikit-learn's
    ``mean_tweedie_deviance``. ``power`` is a Python float (a metric
    hyper-parameter, not data), so the formula branch is chosen with plain
    ``if`` statements.

    Args:
        y: Target values as a Float64 expression.
        p: Predicted values as a Float64 expression.
        power: Tweedie power parameter.

    Returns:
        A per-sample unit-deviance expression.
    """
    if power == 0:
        return (y - p) ** 2
    if power == 1:
        # Poisson: the y*log(y/p) term -> 0 as y -> 0.
        y_log = pl.when(y == 0).then(0.0).otherwise(y * (y / p).log())
        return 2.0 * (y_log - y + p)
    if power == 2:
        # Gamma.
        return 2.0 * ((p / y).log() + y / p - 1.0)
    # General power (p < 0, 1 < power < 2, or power > 2).
    return 2.0 * (
        y ** (2.0 - power) / ((1.0 - power) * (2.0 - power))
        - y * p ** (1.0 - power) / (1.0 - power)
        + p ** (2.0 - power) / (2.0 - power)
    )


def mean_tweedie_deviance(
    target: IntoExpr, pred: IntoExpr, *, power: float = 0.0, weight: WeightInput = None
) -> pl.Expr:
    """Compute the mean Tweedie deviance.

    Mirrors scikit-learn's ``mean_tweedie_deviance``. The per-sample unit
    deviance depends on ``power``:

    - ``power == 0``: ``(y - pred)²`` (equivalent to MSE).
    - ``power == 1`` (Poisson): ``2·(y·log(y/pred) - y + pred)``, with the
      ``y·log(y/pred)`` term taken as 0 when ``y == 0``.
    - ``power == 2`` (Gamma): ``2·(log(pred/y) + y/pred - 1)``.
    - general ``power``: ``2·( y^(2-p)/((1-p)(2-p)) - y·pred^(1-p)/(1-p)
      + pred^(2-p)/(2-p) )``.

    Domain requirements (mirroring scikit-learn's documentation):
    ``power == 0`` accepts any real ``y`` and ``pred``; ``0 < power < 1`` is not
    defined; ``power == 1`` requires ``y >= 0`` and ``pred > 0``;
    ``1 < power < 2`` requires ``y >= 0`` and ``pred > 0``; ``power >= 2``
    requires ``y > 0`` and ``pred > 0``. Like the existing log-based metrics
    (e.g. :func:`mean_squared_log_error`), out-of-domain inputs are not rejected;
    they yield NaN/inf naturally rather than raising.

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.
        power: Tweedie power parameter. Defaults to 0.0.
        weight: Optional column with sample weights.

    Example:
        >>> import polars as pl
        >>> from polarbearings import mean_tweedie_deviance
        >>> df = pl.DataFrame({"y": [2.0, 0.5, 1.0], "pred": [0.5, 0.5, 2.0]})
        >>> df.select(mean_tweedie_deviance("y", "pred", power=1.0)).to_series()[0]  # doctest: +SKIP
        1.144...
    """
    y = col_expr(target).cast(pl.Float64)
    p = col_expr(pred).cast(pl.Float64)
    dev = _tweedie_unit_deviance(y, p, power)

    w = resolve_weight(weight)
    result = (dev * w).sum() / w.sum() if w is not None else dev.mean()

    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("mean_tweedie_deviance", target, pred, weight)
    )


def mean_poisson_deviance(
    target: IntoExpr, pred: IntoExpr, *, weight: WeightInput = None
) -> pl.Expr:
    """Compute the mean Poisson deviance.

    Thin wrapper over :func:`mean_tweedie_deviance` with ``power = 1``. Mirrors
    scikit-learn's ``mean_poisson_deviance``. Requires ``y >= 0`` and
    ``pred > 0``; out-of-domain inputs yield NaN/inf rather than raising.

    Args:
        target: Column name or expression with actual values (must be >= 0).
        pred: Column name or expression with predicted values (must be > 0).
        weight: Optional column with sample weights.

    Example:
        >>> import polars as pl
        >>> from polarbearings import mean_poisson_deviance
        >>> df = pl.DataFrame({"y": [2.0, 0.0, 1.0, 4.0], "pred": [0.5, 0.5, 2.0, 2.0]})
        >>> df.select(mean_poisson_deviance("y", "pred")).to_series()[0]  # doctest: +SKIP
        1.420...
    """
    y = col_expr(target).cast(pl.Float64)
    p = col_expr(pred).cast(pl.Float64)
    dev = _tweedie_unit_deviance(y, p, 1.0)

    w = resolve_weight(weight)
    result = (dev * w).sum() / w.sum() if w is not None else dev.mean()

    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("mean_poisson_deviance", target, pred, weight)
    )


def mean_gamma_deviance(target: IntoExpr, pred: IntoExpr, *, weight: WeightInput = None) -> pl.Expr:
    """Compute the mean Gamma deviance.

    Thin wrapper over :func:`mean_tweedie_deviance` with ``power = 2``. Mirrors
    scikit-learn's ``mean_gamma_deviance``. Requires ``y > 0`` and ``pred > 0``;
    out-of-domain inputs yield NaN/inf rather than raising.

    Args:
        target: Column name or expression with actual values (must be > 0).
        pred: Column name or expression with predicted values (must be > 0).
        weight: Optional column with sample weights.

    Example:
        >>> import polars as pl
        >>> from polarbearings import mean_gamma_deviance
        >>> df = pl.DataFrame({"y": [2.0, 0.5, 1.0, 4.0], "pred": [0.5, 0.5, 2.0, 2.0]})
        >>> df.select(mean_gamma_deviance("y", "pred")).to_series()[0]  # doctest: +SKIP
        1.056...
    """
    y = col_expr(target).cast(pl.Float64)
    p = col_expr(pred).cast(pl.Float64)
    dev = _tweedie_unit_deviance(y, p, 2.0)

    w = resolve_weight(weight)
    result = (dev * w).sum() / w.sum() if w is not None else dev.mean()

    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("mean_gamma_deviance", target, pred, weight)
    )


def d2_tweedie_score(
    target: IntoExpr, pred: IntoExpr, *, power: float = 0.0, weight: WeightInput = None
) -> pl.Expr:
    """Compute the D² regression score with a Tweedie deviance.

    D² is the deviance generalization of R²:
    ``1 - deviance(y, pred) / deviance(y, baseline)``, where the baseline
    prediction is the optimal constant — for Tweedie deviance, the (weighted)
    mean of ``y``. Mirrors scikit-learn's ``d2_tweedie_score``. A perfect fit
    scores 1.0, predicting the baseline scores 0.0, and worse-than-baseline
    fits score negative. Returns null when the baseline deviance is 0
    (e.g. constant ``y``), where the score is undefined.

    The per-sample deviance reuses the same unit-deviance formula as
    :func:`mean_tweedie_deviance`; see it for the per-``power`` domain
    requirements. Weighting is supported because the baseline (weighted mean) is
    expressible as a pure expression.

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.
        power: Tweedie power parameter. Defaults to 0.0.
        weight: Optional column with sample weights.

    Example:
        >>> import polars as pl
        >>> from polarbearings import d2_tweedie_score
        >>> df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        >>> df.select(d2_tweedie_score("y", "pred")).to_series()[0]
        1.0
    """
    y = col_expr(target).cast(pl.Float64)
    p = col_expr(pred).cast(pl.Float64)

    w = resolve_weight(weight)
    if w is not None:
        w_sum = w.sum()
        y_mean = (y * w).sum() / w_sum
        num = (_tweedie_unit_deviance(y, p, power) * w).sum()
        den = (_tweedie_unit_deviance(y, y_mean, power) * w).sum()
    else:
        y_mean = y.mean()
        num = _tweedie_unit_deviance(y, p, power).sum()
        den = _tweedie_unit_deviance(y, y_mean, power).sum()

    result = pl.when(den == 0).then(None).otherwise(1.0 - num / den)
    return guarded(result, values=[target, pred], weight=weight).alias(
        _regression_alias("d2_tweedie_score", target, pred, weight)
    )


def d2_absolute_error_score(target: IntoExpr, pred: IntoExpr) -> pl.Expr:
    """Compute the D² regression score with an absolute-error deviance.

    D² = ``1 - MAE(y, pred) / MAE(y, median(y))``, where the baseline is the
    optimal constant for absolute error — the median of ``y``. Mirrors
    scikit-learn's ``d2_absolute_error_score``. A perfect fit scores 1.0,
    predicting the median scores 0.0. Returns null when the baseline absolute
    error is 0 (e.g. constant ``y``), where the score is undefined.

    No ``weight`` parameter is provided. The baseline is a *median* of ``y``,
    and scikit-learn's weighted variant uses a weighted percentile, which cannot
    be expressed correctly as a single pure Polars expression (see
    :func:`median_absolute_error` for the same reasoning). Weighting is
    intentionally unsupported here; the result matches scikit-learn's unweighted
    ``d2_absolute_error_score``.

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted values.

    Example:
        >>> import polars as pl
        >>> from polarbearings import d2_absolute_error_score
        >>> df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        >>> df.select(d2_absolute_error_score("y", "pred")).to_series()[0]
        1.0
    """
    y = col_expr(target).cast(pl.Float64)
    p = col_expr(pred).cast(pl.Float64)

    num = (y - p).abs().sum()
    den = (y - y.median()).abs().sum()

    result = pl.when(den == 0).then(None).otherwise(1.0 - num / den)
    return guarded(result, values=[target, pred]).alias(
        _regression_alias("d2_absolute_error_score", target, pred, None)
    )


def d2_pinball_score(target: IntoExpr, pred: IntoExpr, *, alpha: float = 0.5) -> pl.Expr:
    """Compute the D² regression score with a pinball-loss deviance.

    D² = ``1 - pinball(y, pred) / pinball(y, baseline)``, where the baseline is
    the optimal constant for the pinball loss at quantile ``alpha`` — the
    ``alpha``-quantile of ``y``. Mirrors scikit-learn's ``d2_pinball_score``. A
    perfect fit scores 1.0, predicting the ``alpha``-quantile scores 0.0. With
    ``alpha = 0.5`` this reduces to :func:`d2_absolute_error_score`. Returns null
    when the baseline pinball loss is 0 (e.g. constant ``y``), where the score is
    undefined.

    No ``weight`` parameter is provided. The baseline is a *quantile* of ``y``,
    and scikit-learn's weighted variant uses a weighted percentile, which cannot
    be expressed correctly as a single pure Polars expression (see
    :func:`median_absolute_error` for the same reasoning). Weighting is
    intentionally unsupported here; the result matches scikit-learn's unweighted
    ``d2_pinball_score``.

    Args:
        target: Column name or expression with actual values.
        pred: Column name or expression with predicted (quantile) values.
        alpha: Target quantile in [0, 1]. Defaults to 0.5.

    Example:
        >>> import polars as pl
        >>> from polarbearings import d2_pinball_score
        >>> df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        >>> df.select(d2_pinball_score("y", "pred")).to_series()[0]
        1.0
    """
    y = col_expr(target).cast(pl.Float64)
    p = col_expr(pred).cast(pl.Float64)

    def _pinball(err: pl.Expr) -> pl.Expr:
        return pl.when(err >= 0).then(alpha * err).otherwise((alpha - 1.0) * err)

    # scikit-learn's baseline quantile is the "inverted_cdf" percentile: the
    # smallest ``y`` whose cumulative count fraction reaches ``alpha``, i.e.
    # ``sorted(y)[ceil(alpha * n) - 1]`` (clamped to index >= 0). None of
    # Polars' built-in quantile interpolations reproduce this, so build it
    # explicitly so the baseline pinball loss matches scikit-learn exactly.
    idx = ((alpha * y.len()).ceil() - 1.0).clip(lower_bound=0).cast(pl.Int64)
    # ``implode().list.get`` rather than ``sort().get(idx)``: the latter broadcasts a
    # scalar against the column inside ``group_by().agg()`` in a way that panics on
    # Polars 1.24.0 ("expected List, got f64"). Imploding to an explicit list keeps
    # the indexed read unambiguously a list op — robust on every supported version.
    baseline = y.sort().implode().list.get(idx)

    num = _pinball(y - p).sum()
    den = _pinball(y - baseline).sum()

    result = pl.when(den == 0).then(None).otherwise(1.0 - num / den)
    return guarded(result, values=[target, pred]).alias(
        _regression_alias("d2_pinball_score", target, pred, None)
    )
