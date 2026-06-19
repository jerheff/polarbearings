"""Shared data fixtures and the size sweep for the benchmark suite.

Centralizing data generation keeps every metric on identically shaped inputs
(so cross-metric numbers are comparable) and puts the size sweep — including
the 1M and 10M extremes — in one place instead of duplicated across every
``bench_*.py`` file.

The sweep can be capped via the ``BENCH_MAX_N`` environment variable (e.g.
``BENCH_MAX_N=100000`` in CI) to skip the slow large-N cases without editing
code. Metric-specific data shapes whose sklearn baseline is a Python loop
(tied scores, grouped, threshold sweep) define their own bounded params
locally — the 10M extreme is pathological for those.
"""

import os

import numpy as np
import numpy.typing as npt
import pytest

SEED = 42

# Single source of truth for the per-sample size sweep.
_ALL_SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
_cap = os.environ.get("BENCH_MAX_N")
SIZES = [n for n in _ALL_SIZES if not _cap or n <= int(_cap)]


def make_binary_scores(n: int) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """Binary labels with scores correlated to the label (separable-ish)."""
    np.random.seed(SEED)
    labels = np.random.randint(0, 2, n)
    scores = labels * 0.6 + np.random.randn(n) * 0.3
    return labels, scores


def make_binary_probs(n: int) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """Binary labels with independent uniform probabilities in [0, 1)."""
    np.random.seed(SEED)
    labels = np.random.randint(0, 2, n)
    probs = np.random.rand(n)
    return labels, probs


def make_regression(n: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Continuous target with a noisy prediction."""
    np.random.seed(SEED)
    y = np.random.randn(n)
    pred = y + np.random.randn(n) * 0.5
    return y, pred


def make_regression_positive(n: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Strictly-positive target/prediction, for ratio metrics like MAPE."""
    np.random.seed(SEED)
    y = np.exp(np.random.randn(n))  # lognormal: always > 0, so MAPE is well-defined
    pred = y * (1 + np.random.randn(n) * 0.1)
    return y, pred


@pytest.fixture(params=SIZES)
def binary_scores(
    request: pytest.FixtureRequest,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64], int]:
    """Labels + correlated scores for ranking/threshold metrics."""
    n: int = request.param
    labels, scores = make_binary_scores(n)
    return labels, scores, n


@pytest.fixture(params=SIZES)
def binary_probs(
    request: pytest.FixtureRequest,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64], int]:
    """Labels + uniform probabilities for probabilistic metrics."""
    n: int = request.param
    labels, probs = make_binary_probs(n)
    return labels, probs, n


@pytest.fixture(params=SIZES)
def regression_data(
    request: pytest.FixtureRequest,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Continuous target + noisy prediction."""
    n: int = request.param
    y, pred = make_regression(n)
    return y, pred, n


@pytest.fixture(params=SIZES)
def regression_positive(
    request: pytest.FixtureRequest,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Strictly-positive target + prediction for ratio metrics (MAPE)."""
    n: int = request.param
    y, pred = make_regression_positive(n)
    return y, pred, n


@pytest.fixture(params=SIZES)
def gini_data(
    request: pytest.FixtureRequest,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Fraud-like exponential losses with a monotonic-but-noisy score."""
    n: int = request.param
    rng = np.random.default_rng(SEED)
    losses: npt.NDArray[np.float64] = rng.exponential(scale=1.0, size=n)
    scores = losses + rng.normal(scale=0.5 * losses.std(), size=n)
    return losses, scores, n
