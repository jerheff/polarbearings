"""Polarbear: High-performance metrics for Polars DataFrames.

A library providing efficient implementations of machine learning metrics
as native Polars expressions.
"""

from polarbear.metrics import brier_score, log_loss, roc_auc

__version__: str = "0.1.0"
__all__: list[str] = ["roc_auc", "log_loss", "brier_score"]
