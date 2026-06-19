"""Benchmarks for log loss implementation."""

from typing import Any

import polars as pl
from pytest_benchmark.fixture import BenchmarkFixture
from sklearn.metrics import log_loss as sklearn_log_loss

from polarbear import log_loss


class TestLogLossPerformance:
    """Performance benchmarks for log loss (shared ``binary_probs``)."""

    def test_polarbear_log_loss(
        self, benchmark: BenchmarkFixture, binary_probs: tuple[Any, Any, int]
    ) -> None:
        """Benchmark polarbear log loss."""
        labels, probs, n = binary_probs
        benchmark.group = f"Log Loss n={n}"
        df = pl.DataFrame({"label": labels, "prob": probs})

        def compute() -> Any:
            return df.select(log_loss("label", "prob")).to_series()[0]

        result = benchmark(compute)
        assert result > 0

    def test_sklearn_log_loss(
        self, benchmark: BenchmarkFixture, binary_probs: tuple[Any, Any, int]
    ) -> None:
        """Benchmark sklearn log loss for comparison."""
        labels, probs, n = binary_probs
        benchmark.group = f"Log Loss n={n}"

        def compute() -> Any:
            return sklearn_log_loss(labels, probs)

        result = benchmark(compute)
        assert result > 0
