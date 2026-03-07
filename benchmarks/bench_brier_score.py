"""Benchmarks for Brier score implementation."""

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sklearn.metrics import brier_score_loss

from polarbear import brier_score


class TestBrierScorePerformance:
    """Performance benchmarks for Brier score."""

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

    def test_polarbear_brier_score(
        self, benchmark: BenchmarkFixture, data: tuple[Any, Any], request: pytest.FixtureRequest
    ) -> None:
        """Benchmark polarbear Brier score."""
        labels, probs = data
        n = request.node.callspec.params["data"]
        benchmark.group = f"Brier Score n={n}"
        df = pl.DataFrame({"label": labels, "prob": probs})

        def compute() -> Any:
            return df.select(brier_score("label", "prob")).to_series()[0]

        result = benchmark(compute)
        assert 0.0 <= result <= 1.0

    def test_sklearn_brier_score(
        self, benchmark: BenchmarkFixture, data: tuple[Any, Any], request: pytest.FixtureRequest
    ) -> None:
        """Benchmark sklearn Brier score for comparison."""
        labels, probs = data
        n = request.node.callspec.params["data"]
        benchmark.group = f"Brier Score n={n}"

        def compute() -> Any:
            return brier_score_loss(labels, probs)

        result = benchmark(compute)
        assert 0.0 <= result <= 1.0
