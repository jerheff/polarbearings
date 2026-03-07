"""Regression metrics implemented as Polars expressions."""

import polars as pl


def _regression_alias(name: str, target: str, pred: str, weight: str | None) -> str:
    alias = f"{name}_{target}_{pred}"
    if weight is not None:
        alias += f"_{weight}"
    return alias


def r2_score(target: str, pred: str, weight: str | None = None) -> pl.Expr:
    """Compute the coefficient of determination (R-squared).

    R² = 1 - SS_res / SS_tot, where SS_res = sum((y - pred)²) and
    SS_tot = sum((y - mean(y))²). Returns null on empty data or when
    all target values are identical (SS_tot = 0).

    Args:
        target: Column with actual values.
        pred: Column with predicted values.
        weight: Optional column with sample weights.
    """
    y = pl.col(target).cast(pl.Float64)
    p = pl.col(pred).cast(pl.Float64)

    if weight is not None:
        w = pl.col(weight).cast(pl.Float64)
        w_sum = w.sum()
        y_mean = (y * w).sum() / w_sum
        ss_res = ((y - p) ** 2 * w).sum()
        ss_tot = ((y - y_mean) ** 2 * w).sum()
    else:
        y_mean = y.mean()
        ss_res = ((y - p) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()

    result = pl.when(ss_tot == 0).then(None).otherwise(1 - ss_res / ss_tot)
    return result.alias(_regression_alias("r2_score", target, pred, weight))


def mape(target: str, pred: str, weight: str | None = None) -> pl.Expr:
    """Compute mean absolute percentage error.

    MAPE = mean(|y - pred| / |y|). Returns null on empty data. Rows where
    target == 0 are excluded from the calculation (division by zero).

    Args:
        target: Column with actual values.
        pred: Column with predicted values.
        weight: Optional column with sample weights.
    """
    y = pl.col(target).cast(pl.Float64)
    p = pl.col(pred).cast(pl.Float64)
    pct_error = (y - p).abs() / y.abs()

    if weight is not None:
        w = pl.col(weight).cast(pl.Float64)
        # Filter out y==0 rows by setting their weight to null
        w_valid = pl.when(y == 0).then(None).otherwise(w)
        pct_valid = pl.when(y == 0).then(None).otherwise(pct_error)
        result = (pct_valid * w_valid).sum() / w_valid.sum()
    else:
        pct_valid = pl.when(y == 0).then(None).otherwise(pct_error)
        result = pct_valid.mean()

    return result.alias(_regression_alias("mape", target, pred, weight))


def mae(target: str, pred: str, weight: str | None = None) -> pl.Expr:
    """Compute mean absolute error.

    MAE = mean(|target - pred|), or weighted: sum(w·|target - pred|) / sum(w).

    Args:
        target: Column with actual values.
        pred: Column with predicted values.
        weight: Optional column with sample weights.
    """
    diff = (pl.col(target).cast(pl.Float64) - pl.col(pred).cast(pl.Float64)).abs()

    if weight is not None:
        w = pl.col(weight).cast(pl.Float64)
        result = (diff * w).sum() / w.sum()
    else:
        result = diff.mean()

    return result.alias(_regression_alias("mae", target, pred, weight))


def mse(target: str, pred: str, weight: str | None = None) -> pl.Expr:
    """Compute mean squared error.

    MSE = mean((target - pred)²), or weighted: sum(w·(target - pred)²) / sum(w).

    Args:
        target: Column with actual values.
        pred: Column with predicted values.
        weight: Optional column with sample weights.
    """
    diff_sq = (pl.col(target).cast(pl.Float64) - pl.col(pred).cast(pl.Float64)) ** 2

    if weight is not None:
        w = pl.col(weight).cast(pl.Float64)
        result = (diff_sq * w).sum() / w.sum()
    else:
        result = diff_sq.mean()

    return result.alias(_regression_alias("mse", target, pred, weight))


def rmse(target: str, pred: str, weight: str | None = None) -> pl.Expr:
    """Compute root mean squared error.

    RMSE = sqrt(MSE).

    Args:
        target: Column with actual values.
        pred: Column with predicted values.
        weight: Optional column with sample weights.
    """
    diff_sq = (pl.col(target).cast(pl.Float64) - pl.col(pred).cast(pl.Float64)) ** 2

    if weight is not None:
        w = pl.col(weight).cast(pl.Float64)
        result = ((diff_sq * w).sum() / w.sum()).sqrt()
    else:
        result = diff_sq.mean().sqrt()

    return result.alias(_regression_alias("rmse", target, pred, weight))
