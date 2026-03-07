"""Polarbear: High-performance metrics for Polars DataFrames.

A library providing efficient implementations of machine learning metrics
as native Polars expressions.
"""

from polarbear.average_precision import average_precision
from polarbear.brier_score import brier_score
from polarbear.classification import (
    accuracy,
    balanced_accuracy,
    f1_score,
    percentile_thresholds,
    precision,
    recall,
    threshold_sweep,
)
from polarbear.log_loss import log_loss
from polarbear.regression import mae, mse, rmse
from polarbear.roc_auc import roc_auc

__version__: str = "0.1.0"
__all__: list[str] = [
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score",
    "precision",
    "recall",
    "f1_score",
    "accuracy",
    "balanced_accuracy",
    "threshold_sweep",
    "percentile_thresholds",
    "mae",
    "mse",
    "rmse",
]
