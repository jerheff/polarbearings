"""Benchmarks for the normalized Gini coefficient.

Gini has no scikit-learn equivalent, so these track polarbear's absolute
throughput rather than a speedup ratio.
"""

from typing import Any

import numpy as np
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from polarbear import gini_coefficient


class TestGiniPerformance:
    """Performance benchmarks for the normalized Gini coefficient (shared ``gini_data``)."""

    def test_polarbear_gini(
        self, benchmark: BenchmarkFixture, gini_data: tuple[Any, Any, int]
    ) -> None:
        """Benchmark polarbear Gini implementation."""
        losses, scores, n = gini_data
        benchmark.group = f"Gini n={n}"
        df = pl.DataFrame({"loss": losses, "score": scores})

        def compute() -> Any:
            return df.select(gini_coefficient("loss", "score")).to_series()[0]

        result = benchmark(compute)
        assert -1.0 <= result <= 1.0


class TestGiniGroupedPerformance:
    """Benchmarks for group-wise Gini calculation (group-count sweep)."""

    @pytest.fixture(params=[10, 100, 1000])
    def grouped_data(self, request: pytest.FixtureRequest) -> tuple[pl.DataFrame, int]:
        """Generate grouped data with 100 rows per group."""
        groups: int = request.param
        n = groups * 100
        rng = np.random.default_rng(42)
        group_ids = np.repeat(np.arange(groups), 100)
        losses = rng.exponential(scale=1.0, size=n)
        scores = losses + rng.normal(scale=0.5 * losses.std(), size=n)
        df = pl.DataFrame({"group": group_ids.astype(str), "loss": losses, "score": scores})
        return df, groups

    def test_polarbear_grouped_gini(
        self, benchmark: BenchmarkFixture, grouped_data: tuple[pl.DataFrame, int]
    ) -> None:
        """Benchmark polarbear grouped Gini."""
        df, groups = grouped_data
        benchmark.group = f"Grouped Gini groups={groups}"

        def compute() -> Any:
            return df.group_by("group").agg(gini_coefficient("loss", "score"))

        result = benchmark(compute)
        assert result.shape[0] == groups
