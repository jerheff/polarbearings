"""Benchmarks for log loss implementation."""

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sklearn.metrics import log_loss as sklearn_log_loss

from polarbear import log_loss


class TestLogLossPerformance:
    """Performance benchmarks for log loss."""

    @pytest.fixture(params=[100, 1000, 10000, 100000])
    def data(
        self, request: pytest.FixtureRequest
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
        """Generate test data of various sizes."""
        n: int = request.param
        np.random.seed(42)
        labels = np.random.randint(0, 2, n)
        probs = np.random.rand(n)
        return labels, probs

    def test_polarbear_log_loss(
        self, benchmark: BenchmarkFixture, data: tuple[Any, Any], request: pytest.FixtureRequest
    ) -> None:
        """Benchmark polarbear log loss."""
        labels, probs = data
        n = request.node.callspec.params["data"]
        benchmark.group = f"Log Loss n={n}"
        df = pl.DataFrame({"label": labels, "prob": probs})

        def compute() -> Any:
            return df.select(log_loss("label", "prob")).to_series()[0]

        result = benchmark(compute)
        assert result > 0

    def test_sklearn_log_loss(
        self, benchmark: BenchmarkFixture, data: tuple[Any, Any], request: pytest.FixtureRequest
    ) -> None:
        """Benchmark sklearn log loss for comparison."""
        labels, probs = data
        n = request.node.callspec.params["data"]
        benchmark.group = f"Log Loss n={n}"

        def compute() -> Any:
            return sklearn_log_loss(labels, probs)

        result = benchmark(compute)
        assert result > 0
