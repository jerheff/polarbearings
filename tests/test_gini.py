"""Tests for the normalized Gini coefficient."""

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polarbearings import gini_coefficient


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Average-rank implementation that mirrors Polars' ``rank``."""
    sorter = np.argsort(values, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size)

    sorted_values = values[sorter]
    obs = np.concatenate(([True], sorted_values[1:] != sorted_values[:-1]))
    dense = np.cumsum(obs)
    nonzero = np.nonzero(obs)[0]
    count = np.diff(np.append(nonzero, len(sorted_values)))

    # Average rank of tied values.
    ranks = (np.cumsum(count)[dense - 1] + nonzero[dense - 1] + 1) / 2.0
    return ranks[inv]


def _reference_gini_unweighted(target: np.ndarray, score: np.ndarray) -> float:
    """NumPy reference for normalized unweighted Gini."""
    n = len(target)
    if n < 2 or target.sum() <= 0:
        return float("nan")

    rank_score = _rankdata_average(score)
    raw = (2.0 * np.sum(rank_score * target) - (n + 1.0) * target.sum()) / (n * target.sum())

    rank_target = _rankdata_average(target)
    perfect = (2.0 * np.sum(rank_target * target) - (n + 1.0) * target.sum()) / (n * target.sum())

    if perfect == 0:
        return float("nan")
    return float(raw / perfect)


def _lorenz_area(target: np.ndarray, weights: np.ndarray, ordering: np.ndarray) -> float:
    """Area under the Lorenz curve for a given ordering."""
    y = target[ordering]
    w = weights[ordering]
    cum_weight = np.cumsum(w) / weights.sum()
    cum_target = np.cumsum(y) / target.sum()

    delta_weight = cum_weight - np.concatenate(([0.0], cum_weight[:-1]))
    avg_target = (cum_target + np.concatenate(([0.0], cum_target[:-1]))) / 2
    return float(np.sum(delta_weight * avg_target))


def _reference_gini_weighted(target: np.ndarray, score: np.ndarray, weights: np.ndarray) -> float:
    """NumPy reference for normalized weighted Gini."""
    n = len(target)
    if n < 2 or target.sum() <= 0 or weights.sum() <= 0:
        return float("nan")

    model_order = np.argsort(score, kind="mergesort")[::-1]
    auc_model = _lorenz_area(target, weights, model_order)
    raw = 2.0 * auc_model - 1.0

    safe_weights = np.where(weights == 0, 1.0, weights)
    ratio = target / safe_weights
    ratio = np.where(weights == 0, np.inf, ratio)
    perfect_order = np.argsort(ratio, kind="mergesort")[::-1]
    auc_perfect = _lorenz_area(target, weights, perfect_order)
    perfect = 2.0 * auc_perfect - 1.0

    if perfect == 0:
        return float("nan")
    return float(raw / perfect)


class TestGiniUnweighted:
    def test_perfect_ranking(self):
        # Highest score should align with largest target.
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "score": [1.0, 2.0, 3.0, 4.0]})
        result = df.select(gini_coefficient("y", "score")).to_series()[0]
        assert result == pytest.approx(1.0)

    def test_inverted_ranking(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "score": [4.0, 3.0, 2.0, 1.0]})
        result = df.select(gini_coefficient("y", "score")).to_series()[0]
        assert result == pytest.approx(-1.0)

    def test_random_ranking_is_near_zero(self):
        rng = np.random.default_rng(42)
        y = rng.exponential(scale=1.0, size=2000)
        score = rng.random(2000)
        df = pl.DataFrame({"y": y, "score": score})
        result = df.select(gini_coefficient("y", "score")).to_series()[0]
        assert result == pytest.approx(0.0, abs=0.05)

    def test_matches_numpy_reference(self):
        rng = np.random.default_rng(123)
        y = rng.exponential(scale=2.0, size=50)
        score = rng.random(50)
        expected = _reference_gini_unweighted(y, score)

        df = pl.DataFrame({"y": y, "score": score})
        result = df.select(gini_coefficient("y", "score")).to_series()[0]
        assert result == pytest.approx(expected, rel=1e-5)

    def test_binary_matches_reference(self):
        rng = np.random.default_rng(7)
        n = 200
        # Use unique scores to avoid tie-ambiguity.
        score = np.arange(n) + rng.random(n) * 0.5
        target = (rng.random(n) > 0.6).astype(float)
        expected = _reference_gini_unweighted(target, score)

        df = pl.DataFrame({"y": target, "score": score})
        result = df.select(gini_coefficient("y", "score")).to_series()[0]
        if np.isnan(expected):
            assert result is None
        else:
            assert result == pytest.approx(expected, rel=1e-5)

    def test_ties_give_stable_result(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "score": [1.0, 1.0, 1.0]})
        result = df.select(gini_coefficient("y", "score")).to_series()[0]
        assert isinstance(result, float)
        # All tied scores cannot improve on random, normalized Gini stays ~0.
        assert abs(result) <= 1e-3


class TestGiniWeighted:
    def test_unit_weights_match_unweighted(self):
        rng = np.random.default_rng(99)
        y = rng.exponential(scale=1.0, size=40)
        score = rng.random(40)
        df = pl.DataFrame({"y": y, "score": score, "w": np.ones(40, dtype=float)})
        weighted = df.select(gini_coefficient("y", "score", weight="w")).to_series()[0]
        unweighted = df.select(gini_coefficient("y", "score")).to_series()[0]
        assert weighted == pytest.approx(unweighted, rel=1e-5)

    def test_weighted_matches_reference(self):
        rng = np.random.default_rng(21)
        y = rng.exponential(scale=2.0, size=60)
        score = rng.random(60)
        weights = rng.exponential(scale=1.0, size=60)
        expected = _reference_gini_weighted(y, score, weights)

        df = pl.DataFrame({"y": y, "score": score, "w": weights})
        result = df.select(gini_coefficient("y", "score", weight="w")).to_series()[0]
        assert result == pytest.approx(expected, rel=1e-5)

    def test_zero_weight_rows_ignored(self):
        df = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0],
                "score": [1.0, 2.0, 3.0, 4.0],
                "w": [1.0, 1.0, 0.0, 1.0],
            }
        )
        result = df.select(gini_coefficient("y", "score", weight="w")).to_series()[0]
        expected = _reference_gini_weighted(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 1.0, 0.0, 1.0]),
        )
        assert result == pytest.approx(expected, rel=1e-5)


class TestGiniEdgeCases:
    def test_empty_dataframe(self):
        df = pl.DataFrame({"y": [], "score": []})
        result = df.select(gini_coefficient("y", "score")).to_series()[0]
        assert result is None

    def test_single_row(self):
        df = pl.DataFrame({"y": [1.0], "score": [0.5]})
        result = df.select(gini_coefficient("y", "score")).to_series()[0]
        assert result is None

    def test_empty_dataframe_weighted(self):
        df = pl.DataFrame(
            {
                "y": pl.Series([], dtype=pl.Float64),
                "score": pl.Series([], dtype=pl.Float64),
                "w": pl.Series([], dtype=pl.Float64),
            }
        )
        result = df.select(gini_coefficient("y", "score", weight="w")).to_series()[0]
        assert result is None

    def test_all_zero_targets(self):
        df = pl.DataFrame({"y": [0.0, 0.0, 0.0], "score": [0.1, 0.5, 0.9]})
        result = df.select(gini_coefficient("y", "score")).to_series()[0]
        assert result is None

    def test_negative_total_target_is_undefined(self):
        # Target values are expected to be non-negative; mixed signs give a zero
        # or negative total and should return null rather than a misleading value.
        df = pl.DataFrame({"y": [1.0, -1.0], "score": [0.9, 0.1]})
        result = df.select(gini_coefficient("y", "score")).to_series()[0]
        assert result is None


class TestGiniGrouped:
    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "y": [1.0, 2.0, 3.0, 10.0, 5.0, 1.0],
                "score": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        )
        result = df.group_by("group").agg(gini_coefficient("y", "score")).sort("group")
        rows = result.to_dicts()
        # Group A: perfectly ordered ascending by score with target.
        assert rows[0]["gini_y_score"] == pytest.approx(1.0)
        # Group B: perfectly inverted.
        assert rows[1]["gini_y_score"] == pytest.approx(-1.0)


@given(
    size=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.differing_executors],
)
def test_unweighted_against_reference(size, seed):
    rng = np.random.default_rng(seed)
    y = rng.exponential(scale=2.0, size=size)
    # Add small jitter so ties are rare and deterministic mismatches avoided.
    score = np.arange(size) + rng.random(size)

    expected = _reference_gini_unweighted(y, score)
    df = pl.DataFrame({"y": y, "score": score})
    result = df.select(gini_coefficient("y", "score")).to_series()[0]
    assert result == pytest.approx(expected, rel=1e-5)


@given(
    size=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.differing_executors],
)
def test_weighted_against_reference(size, seed):
    rng = np.random.default_rng(seed)
    y = rng.exponential(scale=2.0, size=size)
    score = np.arange(size) + rng.random(size)
    weights = rng.exponential(scale=1.0, size=size)

    expected = _reference_gini_weighted(y, score, weights)
    df = pl.DataFrame({"y": y, "score": score, "w": weights})
    result = df.select(gini_coefficient("y", "score", weight="w")).to_series()[0]
    assert result == pytest.approx(expected, rel=1e-5)
