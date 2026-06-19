"""Benchmarks for classification metrics implementation."""

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sklearn.metrics import f1_score as sklearn_f1
from sklearn.metrics import matthews_corrcoef as sklearn_mcc
from sklearn.metrics import precision_score as sklearn_precision

from polarbear import f1_score, matthews_corrcoef, precision, threshold_sweep


class TestPrecisionPerformance:
    """Precision vs sklearn (shared ``binary_scores``)."""

    def test_polarbear_precision(
        self, benchmark: BenchmarkFixture, binary_scores: tuple[Any, Any, int]
    ) -> None:
        labels, probs, n = binary_scores
        benchmark.group = f"Precision n={n}"
        df = pl.DataFrame({"label": labels, "prob": probs})

        def compute() -> Any:
            return df.select(precision("label", "prob")).to_series()[0]

        benchmark(compute)

    def test_sklearn_precision(
        self, benchmark: BenchmarkFixture, binary_scores: tuple[Any, Any, int]
    ) -> None:
        labels, probs, n = binary_scores
        benchmark.group = f"Precision n={n}"
        preds = (probs >= 0.5).astype(int)

        def compute() -> Any:
            return sklearn_precision(labels, preds)

        benchmark(compute)


class TestF1Performance:
    """F1 vs sklearn (shared ``binary_scores``)."""

    def test_polarbear_f1(
        self, benchmark: BenchmarkFixture, binary_scores: tuple[Any, Any, int]
    ) -> None:
        labels, probs, n = binary_scores
        benchmark.group = f"F1 Score n={n}"
        df = pl.DataFrame({"label": labels, "prob": probs})

        def compute() -> Any:
            return df.select(f1_score("label", "prob")).to_series()[0]

        benchmark(compute)

    def test_sklearn_f1(
        self, benchmark: BenchmarkFixture, binary_scores: tuple[Any, Any, int]
    ) -> None:
        labels, probs, n = binary_scores
        benchmark.group = f"F1 Score n={n}"
        preds = (probs >= 0.5).astype(int)

        def compute() -> Any:
            return sklearn_f1(labels, preds)

        benchmark(compute)


class TestMCCPerformance:
    """Matthews correlation coefficient vs sklearn (distinct aggregation shape)."""

    def test_polarbear_mcc(
        self, benchmark: BenchmarkFixture, binary_scores: tuple[Any, Any, int]
    ) -> None:
        labels, probs, n = binary_scores
        benchmark.group = f"MCC n={n}"
        df = pl.DataFrame({"label": labels, "prob": probs})

        def compute() -> Any:
            return df.select(matthews_corrcoef("label", "prob")).to_series()[0]

        benchmark(compute)

    def test_sklearn_mcc(
        self, benchmark: BenchmarkFixture, binary_scores: tuple[Any, Any, int]
    ) -> None:
        labels, probs, n = binary_scores
        benchmark.group = f"MCC n={n}"
        preds = (probs >= 0.5).astype(int)

        def compute() -> Any:
            return sklearn_mcc(labels, preds)

        benchmark(compute)


class TestThresholdSweepPerformance:
    """Threshold sweep vs a sklearn Python loop.

    Bounded sizes on purpose: the sklearn baseline recomputes F1 at every
    threshold in Python, so the 10M extreme would be pathological here.
    """

    @pytest.fixture(params=[1000, 10000, 100000])
    def data(
        self, request: pytest.FixtureRequest
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64], int]:
        n: int = request.param
        np.random.seed(42)
        labels = np.random.randint(0, 2, n)
        probs = labels * 0.6 + np.random.randn(n) * 0.3
        return labels, probs, n

    def test_polarbear_sweep(self, benchmark: BenchmarkFixture, data: tuple[Any, Any, int]) -> None:
        labels, probs, n = data
        benchmark.group = f"Threshold Sweep (10 thresholds) n={n}"
        df = pl.DataFrame({"label": labels, "prob": probs})
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

        def compute() -> Any:
            return df.select(*threshold_sweep(f1_score, "label", "prob", thresholds))

        benchmark(compute)

    def test_sklearn_sweep(self, benchmark: BenchmarkFixture, data: tuple[Any, Any, int]) -> None:
        labels, probs, n = data
        benchmark.group = f"Threshold Sweep (10 thresholds) n={n}"
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

        def compute() -> dict[float, float]:
            results = {}
            for t in thresholds:
                preds = (probs >= t).astype(int)
                results[t] = sklearn_f1(labels, preds)
            return results

        benchmark(compute)
