"""Polarbearings: High-performance metrics for Polars DataFrames.

A library providing efficient implementations of machine learning metrics
as native Polars expressions.
"""

from polarbearings.average_precision import average_precision
from polarbearings.bootstrap import (
    BootstrapCI,
    bootstrap,
    bootstrap_ci,
    ci_from_distribution,
)
from polarbearings.brier_score import brier_score
from polarbearings.classification import (
    accuracy,
    balanced_accuracy,
    cohens_kappa,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    percentile_thresholds,
    precision,
    recall,
    specificity,
    threshold_sweep,
)
from polarbearings.gini import gini_coefficient
from polarbearings.log_loss import log_loss
from polarbearings.regression import (
    explained_variance_score,
    huber_loss,
    log_cosh_loss,
    mae,
    mape,
    max_error,
    mean_pinball_loss,
    mean_squared_log_error,
    median_absolute_error,
    mse,
    r2_score,
    rmse,
    root_mean_squared_log_error,
    smape,
)
from polarbearings.roc_auc import roc_auc

__all__: list[str] = [
    # Ranking / probabilistic
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score",
    "gini_coefficient",
    # Classification (threshold-based)
    "confusion_matrix",
    "precision",
    "recall",
    "f1_score",
    "fbeta_score",
    "specificity",
    "accuracy",
    "balanced_accuracy",
    "matthews_corrcoef",
    "cohens_kappa",
    # Utilities
    "threshold_sweep",
    "percentile_thresholds",
    "BootstrapCI",
    "bootstrap",
    "bootstrap_ci",
    "ci_from_distribution",
    # Regression
    "mae",
    "mse",
    "rmse",
    "r2_score",
    "mape",
    "mean_squared_log_error",
    "root_mean_squared_log_error",
    "max_error",
    "median_absolute_error",
    "explained_variance_score",
    "mean_pinball_loss",
    "smape",
    "huber_loss",
    "log_cosh_loss",
]
