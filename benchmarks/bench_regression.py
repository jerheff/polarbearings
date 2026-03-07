"""Benchmarks for regression metrics implementation."""

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sklearn.metrics import mean_absolute_error, mean_squared_error

from polarbear import mae, mse


class TestMAEPerformance:
    @pytest.fixture(params=[1000, 10000, 100000])
    def data(
        self, request: pytest.FixtureRequest
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
        n: int = request.param
        np.random.seed(42)
        y = np.random.randn(n)
        pred = y + np.random.randn(n) * 0.5
        return y, pred, n

    def test_polarbear_mae(self, benchmark: BenchmarkFixture, data: tuple[Any, Any, int]) -> None:
        y, pred, n = data
        benchmark.group = f"MAE n={n}"
        df = pl.DataFrame({"y": y, "pred": pred})

        def compute() -> Any:
            return df.select(mae("y", "pred")).to_series()[0]

        benchmark(compute)

    def test_sklearn_mae(self, benchmark: BenchmarkFixture, data: tuple[Any, Any, int]) -> None:
        y, pred, n = data
        benchmark.group = f"MAE n={n}"

        def compute() -> Any:
            return mean_absolute_error(y, pred)

        benchmark(compute)


class TestMSEPerformance:
    @pytest.fixture(params=[1000, 10000, 100000])
    def data(
        self, request: pytest.FixtureRequest
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
        n: int = request.param
        np.random.seed(42)
        y = np.random.randn(n)
        pred = y + np.random.randn(n) * 0.5
        return y, pred, n

    def test_polarbear_mse(self, benchmark: BenchmarkFixture, data: tuple[Any, Any, int]) -> None:
        y, pred, n = data
        benchmark.group = f"MSE n={n}"
        df = pl.DataFrame({"y": y, "pred": pred})

        def compute() -> Any:
            return df.select(mse("y", "pred")).to_series()[0]

        benchmark(compute)

    def test_sklearn_mse(self, benchmark: BenchmarkFixture, data: tuple[Any, Any, int]) -> None:
        y, pred, n = data
        benchmark.group = f"MSE n={n}"

        def compute() -> Any:
            return mean_squared_error(y, pred)

        benchmark(compute)
