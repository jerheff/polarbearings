"""Benchmarks for average precision implementation."""

from typing import Any

import polars as pl
from pytest_benchmark.fixture import BenchmarkFixture
from sklearn.metrics import average_precision_score

from polarbear import average_precision


class TestAveragePrecisionPerformance:
    """Performance benchmarks for average precision (shared ``binary_scores``)."""

    def test_polarbear_ap(
        self, benchmark: BenchmarkFixture, binary_scores: tuple[Any, Any, int]
    ) -> None:
        labels, scores, n = binary_scores
        benchmark.group = f"Average Precision n={n}"
        df = pl.DataFrame({"label": labels, "score": scores})

        def compute() -> Any:
            return df.select(average_precision("label", "score")).to_series()[0]

        result = benchmark(compute)
        assert result is not None

    def test_sklearn_ap(
        self, benchmark: BenchmarkFixture, binary_scores: tuple[Any, Any, int]
    ) -> None:
        labels, scores, n = binary_scores
        benchmark.group = f"Average Precision n={n}"

        def compute() -> Any:
            return average_precision_score(labels, scores)

        result = benchmark(compute)
        assert result is not None
