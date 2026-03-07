"""Benchmarks for grouped metric computation — polarbear vs sklearn loop."""

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.metrics import log_loss as sklearn_log_loss

from polarbear import brier_score, log_loss, roc_auc


class TestGroupedPerformance:
    """Compare polarbear group_by vs sklearn Python loop."""

    @pytest.fixture(params=[10, 100, 1000])
    def grouped_data(
        self, request: pytest.FixtureRequest
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.float64], int]:
        """Generate grouped data with 1000 samples per group."""
        n_groups: int = request.param
        samples_per_group = 1000
        np.random.seed(42)

        groups = np.repeat(np.arange(n_groups), samples_per_group)
        labels = np.random.randint(0, 2, n_groups * samples_per_group)
        scores = labels * 0.6 + np.random.randn(n_groups * samples_per_group) * 0.3
        # Clip to [0, 1] for use as probabilities
        probs = np.clip(scores, 0, 1)

        return groups, labels, probs, n_groups

    def test_polarbear_grouped_all_metrics(
        self,
        benchmark: BenchmarkFixture,
        grouped_data: tuple[Any, Any, Any, int],
        request: pytest.FixtureRequest,
    ) -> None:
        """Benchmark polarbear: all 3 metrics via group_by().agg()."""
        groups, labels, probs, n_groups = grouped_data
        benchmark.group = f"Grouped groups={n_groups}"
        df = pl.DataFrame({"group": groups, "label": labels, "prob": probs})

        def compute() -> pl.DataFrame:
            return df.group_by("group").agg(
                roc_auc("label", "prob"),
                log_loss("label", "prob"),
                brier_score("label", "prob"),
            )

        result = benchmark(compute)
        assert len(result) == n_groups

    def test_sklearn_grouped_all_metrics(
        self,
        benchmark: BenchmarkFixture,
        grouped_data: tuple[Any, Any, Any, int],
        request: pytest.FixtureRequest,
    ) -> None:
        """Benchmark sklearn: all 3 metrics via Python loop over groups."""
        groups, labels, probs, n_groups = grouped_data
        benchmark.group = f"Grouped groups={n_groups}"

        def compute() -> list[tuple[float, float, float]]:
            results: list[tuple[float, float, float]] = []
            for group in range(n_groups):
                mask = groups == group
                g_labels = labels[mask]
                g_probs = probs[mask]
                auc = float(roc_auc_score(g_labels, g_probs))
                ll = float(sklearn_log_loss(g_labels, g_probs))
                bs = float(brier_score_loss(g_labels, g_probs))
                results.append((auc, ll, bs))
            return results

        result = benchmark(compute)
        assert len(result) == n_groups
