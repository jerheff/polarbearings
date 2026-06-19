"""Benchmarks for Brier score implementation."""

from typing import Any

import polars as pl
from pytest_benchmark.fixture import BenchmarkFixture
from sklearn.metrics import brier_score_loss

from polarbearings import brier_score


class TestBrierScorePerformance:
    """Performance benchmarks for Brier score (shared ``binary_probs``)."""

    def test_polarbearings_brier_score(
        self, benchmark: BenchmarkFixture, binary_probs: tuple[Any, Any, int]
    ) -> None:
        """Benchmark polarbearings Brier score."""
        labels, probs, n = binary_probs
        benchmark.group = f"Brier Score n={n}"
        df = pl.DataFrame({"label": labels, "prob": probs})

        def compute() -> Any:
            return df.select(brier_score("label", "prob")).to_series()[0]

        result = benchmark(compute)
        assert 0.0 <= result <= 1.0

    def test_sklearn_brier_score(
        self, benchmark: BenchmarkFixture, binary_probs: tuple[Any, Any, int]
    ) -> None:
        """Benchmark sklearn Brier score for comparison."""
        labels, probs, n = binary_probs
        benchmark.group = f"Brier Score n={n}"

        def compute() -> Any:
            return brier_score_loss(labels, probs)

        result = benchmark(compute)
        assert 0.0 <= result <= 1.0
