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
from polarbearings.calibration import calibration_curve
from polarbearings.class_weight import balanced_class_weights, balanced_sample_weight
from polarbearings.classification import (
    accuracy,
    balanced_accuracy,
    cohens_kappa,
    confusion_matrix,
    f1_score,
    fbeta_score,
    jaccard_score,
    matthews_corrcoef,
    percentile_thresholds,
    precision,
    recall,
    specificity,
    threshold_sweep,
)
from polarbearings.confusion_curve import confusion_curve
from polarbearings.gini import gini_coefficient
from polarbearings.log_loss import log_loss
from polarbearings.ranking import dcg_score, ndcg_score
from polarbearings.regression import (
    d2_absolute_error_score,
    d2_pinball_score,
    d2_tweedie_score,
    explained_variance_score,
    huber_loss,
    log_cosh_loss,
    mae,
    mape,
    max_error,
    mean_gamma_deviance,
    mean_pinball_loss,
    mean_poisson_deviance,
    mean_squared_log_error,
    mean_tweedie_deviance,
    median_absolute_error,
    mse,
    r2_score,
    rmse,
    root_mean_squared_log_error,
    smape,
)
from polarbearings.roc_auc import roc_auc
from polarbearings.thresholds import equal_width, linspace, quantiles, resolve_thresholds

__all__: list[str] = [
    # Ranking / probabilistic
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score",
    "gini_coefficient",
    "dcg_score",
    "ndcg_score",
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
    "jaccard_score",
    # Curves (one row per threshold / bin)
    "confusion_curve",
    "calibration_curve",
    # Utilities
    "threshold_sweep",
    "percentile_thresholds",
    "quantiles",
    "equal_width",
    "linspace",
    "resolve_thresholds",
    "balanced_sample_weight",
    "balanced_class_weights",
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
    "mean_tweedie_deviance",
    "mean_poisson_deviance",
    "mean_gamma_deviance",
    "d2_tweedie_score",
    "d2_absolute_error_score",
    "d2_pinball_score",
]
