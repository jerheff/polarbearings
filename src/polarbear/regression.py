"""Regression metrics implemented as Polars expressions."""

import polars as pl


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

    alias = f"mae_{target}_{pred}"
    if weight is not None:
        alias += f"_{weight}"
    return result.alias(alias)


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

    alias = f"mse_{target}_{pred}"
    if weight is not None:
        alias += f"_{weight}"
    return result.alias(alias)


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

    alias = f"rmse_{target}_{pred}"
    if weight is not None:
        alias += f"_{weight}"
    return result.alias(alias)
