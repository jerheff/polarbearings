"""Benchmarks for ROC AUC implementation."""

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sklearn.metrics import roc_auc_score

from polarbear import roc_auc


class TestROCAUCPerformance:
    """Performance benchmarks for ROC AUC."""

    @pytest.fixture(params=[100, 1000, 10000, 100000])
    def data(
        self, request: pytest.FixtureRequest
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64], int]:
        """Generate test data of various sizes."""
        n: int = request.param
        np.random.seed(42)
        labels = np.random.randint(0, 2, n)
        scores = labels * 0.6 + np.random.randn(n) * 0.3
        return labels, scores, n

    def test_polarbear_roc_auc(
        self, benchmark: BenchmarkFixture, data: tuple[Any, Any, int]
    ) -> None:
        """Benchmark polarbear ROC AUC implementation."""
        labels, scores, n = data
        benchmark.group = f"ROC AUC n={n}"
        df = pl.DataFrame({"label": labels, "score": scores})

        def compute_auc() -> Any:
            return df.select(roc_auc("label", "score")).to_series()[0]

        result = benchmark(compute_auc)
        assert 0.0 <= result <= 1.0

    def test_sklearn_roc_auc(self, benchmark: BenchmarkFixture, data: tuple[Any, Any, int]) -> None:
        """Benchmark sklearn ROC AUC for comparison."""
        labels, scores, n = data
        benchmark.group = f"ROC AUC n={n}"

        def compute_auc() -> Any:
            return roc_auc_score(labels, scores)

        result = benchmark(compute_auc)
        assert 0.0 <= result <= 1.0


class TestTiedScoresPerformance:
    """Benchmarks for datasets with many tied scores."""

    @pytest.fixture(params=[1000, 10000])
    def tied_data(
        self, request: pytest.FixtureRequest
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64], int]:
        """Generate data with many ties."""
        n: int = request.param
        np.random.seed(42)
        labels = np.random.randint(0, 2, n)
        # Create scores with only 10 unique values (lots of ties)
        scores = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], n)
        return labels, scores, n

    def test_polarbear_tied_scores(
        self, benchmark: BenchmarkFixture, tied_data: tuple[Any, Any, int]
    ) -> None:
        """Benchmark polarbear with tied scores."""
        labels, scores, n = tied_data
        benchmark.group = f"ROC AUC Tied n={n}"
        df = pl.DataFrame({"label": labels, "score": scores})

        def compute_auc() -> Any:
            return df.select(roc_auc("label", "score")).to_series()[0]

        result = benchmark(compute_auc)
        assert 0.0 <= result <= 1.0

    def test_sklearn_tied_scores(
        self, benchmark: BenchmarkFixture, tied_data: tuple[Any, Any, int]
    ) -> None:
        """Benchmark sklearn with tied scores."""
        labels, scores, n = tied_data
        benchmark.group = f"ROC AUC Tied n={n}"

        def compute_auc() -> Any:
            return roc_auc_score(labels, scores)

        result = benchmark(compute_auc)
        assert 0.0 <= result <= 1.0
