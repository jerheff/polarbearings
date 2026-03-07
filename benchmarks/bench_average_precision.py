"""Benchmarks for average precision implementation."""

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sklearn.metrics import average_precision_score

from polarbear import average_precision


class TestAveragePrecisionPerformance:
    @pytest.fixture(params=[100, 1000, 10000, 100000])
    def data(
        self, request: pytest.FixtureRequest
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64], int]:
        n: int = request.param
        np.random.seed(42)
        labels = np.random.randint(0, 2, n)
        scores = labels * 0.6 + np.random.randn(n) * 0.3
        return labels, scores, n

    def test_polarbear_ap(self, benchmark: BenchmarkFixture, data: tuple[Any, Any, int]) -> None:
        labels, scores, n = data
        benchmark.group = f"Average Precision n={n}"
        df = pl.DataFrame({"label": labels, "score": scores})

        def compute() -> Any:
            return df.select(average_precision("label", "score")).to_series()[0]

        result = benchmark(compute)
        assert result is not None

    def test_sklearn_ap(self, benchmark: BenchmarkFixture, data: tuple[Any, Any, int]) -> None:
        labels, scores, n = data
        benchmark.group = f"Average Precision n={n}"

        def compute() -> Any:
            return average_precision_score(labels, scores)

        result = benchmark(compute)
        assert result is not None
