"""Benchmarks for regression metrics implementation."""

from typing import Any

import polars as pl
from pytest_benchmark.fixture import BenchmarkFixture
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.metrics import (
    r2_score as sklearn_r2,
)

from polarbear import mae, mape, mse, r2_score


class TestMAEPerformance:
    """MAE vs sklearn (shared ``regression_data``)."""

    def test_polarbear_mae(
        self, benchmark: BenchmarkFixture, regression_data: tuple[Any, Any, int]
    ) -> None:
        y, pred, n = regression_data
        benchmark.group = f"MAE n={n}"
        df = pl.DataFrame({"y": y, "pred": pred})

        def compute() -> Any:
            return df.select(mae("y", "pred")).to_series()[0]

        benchmark(compute)

    def test_sklearn_mae(
        self, benchmark: BenchmarkFixture, regression_data: tuple[Any, Any, int]
    ) -> None:
        y, pred, n = regression_data
        benchmark.group = f"MAE n={n}"

        def compute() -> Any:
            return mean_absolute_error(y, pred)

        benchmark(compute)


class TestMSEPerformance:
    """MSE vs sklearn (shared ``regression_data``)."""

    def test_polarbear_mse(
        self, benchmark: BenchmarkFixture, regression_data: tuple[Any, Any, int]
    ) -> None:
        y, pred, n = regression_data
        benchmark.group = f"MSE n={n}"
        df = pl.DataFrame({"y": y, "pred": pred})

        def compute() -> Any:
            return df.select(mse("y", "pred")).to_series()[0]

        benchmark(compute)

    def test_sklearn_mse(
        self, benchmark: BenchmarkFixture, regression_data: tuple[Any, Any, int]
    ) -> None:
        y, pred, n = regression_data
        benchmark.group = f"MSE n={n}"

        def compute() -> Any:
            return mean_squared_error(y, pred)

        benchmark(compute)


class TestR2Performance:
    """R² vs sklearn — needs a total-variance pass (distinct shape)."""

    def test_polarbear_r2(
        self, benchmark: BenchmarkFixture, regression_data: tuple[Any, Any, int]
    ) -> None:
        y, pred, n = regression_data
        benchmark.group = f"R2 n={n}"
        df = pl.DataFrame({"y": y, "pred": pred})

        def compute() -> Any:
            return df.select(r2_score("y", "pred")).to_series()[0]

        benchmark(compute)

    def test_sklearn_r2(
        self, benchmark: BenchmarkFixture, regression_data: tuple[Any, Any, int]
    ) -> None:
        y, pred, n = regression_data
        benchmark.group = f"R2 n={n}"

        def compute() -> Any:
            return sklearn_r2(y, pred)

        benchmark(compute)


class TestMAPEPerformance:
    """MAPE vs sklearn — per-element division (strictly-positive targets)."""

    def test_polarbear_mape(
        self, benchmark: BenchmarkFixture, regression_positive: tuple[Any, Any, int]
    ) -> None:
        y, pred, n = regression_positive
        benchmark.group = f"MAPE n={n}"
        df = pl.DataFrame({"y": y, "pred": pred})

        def compute() -> Any:
            return df.select(mape("y", "pred")).to_series()[0]

        benchmark(compute)

    def test_sklearn_mape(
        self, benchmark: BenchmarkFixture, regression_positive: tuple[Any, Any, int]
    ) -> None:
        y, pred, n = regression_positive
        benchmark.group = f"MAPE n={n}"

        def compute() -> Any:
            return mean_absolute_percentage_error(y, pred)

        benchmark(compute)
