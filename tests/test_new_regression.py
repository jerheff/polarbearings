"""Tests for r2_score and mape."""

import hypothesis
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import (
    mean_absolute_percentage_error,
)
from sklearn.metrics import (
    r2_score as sklearn_r2,
)

from polarbear import mape, r2_score

# ---------------------------------------------------------------------------
# R-squared
# ---------------------------------------------------------------------------


class TestR2Score:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        result = df.select(r2_score("y", "pred")).to_series()[0]
        assert result == pytest.approx(1.0)

    def test_mean_predictor(self):
        """Predicting the mean should give R² = 0."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean_val = 3.0
        df = pl.DataFrame({"y": y, "pred": [mean_val] * 5})
        result = df.select(r2_score("y", "pred")).to_series()[0]
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_worse_than_mean(self):
        """Bad predictions can give negative R²."""
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [3.0, 1.0, 2.0]})
        result = df.select(r2_score("y", "pred")).to_series()[0]
        expected = sklearn_r2([1.0, 2.0, 3.0], [3.0, 1.0, 2.0])
        assert result == pytest.approx(expected, rel=1e-5)

    def test_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        df = pl.DataFrame({"y": y, "pred": pred})
        result = df.select(r2_score("y", "pred")).to_series()[0]
        expected = sklearn_r2(y, pred)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_constant_target(self):
        """When all target values are identical, R² is undefined."""
        df = pl.DataFrame({"y": [5.0, 5.0, 5.0], "pred": [4.0, 5.0, 6.0]})
        result = df.select(r2_score("y", "pred")).to_series()[0]
        assert result is None

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.1, 2.2, 2.8, 4.5],
            }
        )
        result = df.group_by("group").agg(r2_score("y", "pred")).sort("group")
        expected_a = sklearn_r2([1.0, 2.0], [1.1, 2.2])
        expected_b = sklearn_r2([3.0, 4.0], [2.8, 4.5])
        assert result["r2_score_y_pred"][0] == pytest.approx(expected_a, rel=1e-5)
        assert result["r2_score_y_pred"][1] == pytest.approx(expected_b, rel=1e-5)

    def test_weighted_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        weights = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": weights})
        result = df.select(r2_score("y", "pred", weight="w")).to_series()[0]
        expected = sklearn_r2(y, pred, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-4)

    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[hypothesis.HealthCheck.differing_executors],
    )
    def test_r2_matches_sklearn_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=2, max_value=500), label="size")
        y = data.draw(
            arrays(
                dtype=np.float64,
                shape=size,
                elements=st.floats(
                    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
                ),
            ),
            label="y",
        )
        pred = data.draw(
            arrays(
                dtype=np.float64,
                shape=size,
                elements=st.floats(
                    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
                ),
            ),
            label="pred",
        )
        # R² is undefined when all target values are identical
        hypothesis.assume(y.std() > 1e-10)

        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(r2_score("y", "pred")).to_series()[0]
        expected = sklearn_r2(y, pred)
        assert result == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------


class TestMAPE:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        result = df.select(mape("y", "pred")).to_series()[0]
        assert result == pytest.approx(0.0)

    def test_known_values(self):
        df = pl.DataFrame({"y": [100.0, 200.0], "pred": [110.0, 190.0]})
        result = df.select(mape("y", "pred")).to_series()[0]
        # |100-110|/100 = 0.1, |200-190|/200 = 0.05, mean = 0.075
        assert result == pytest.approx(0.075, rel=1e-5)

    def test_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.rand(100) * 10 + 1  # Positive values, no zeros
        pred = y + np.random.randn(100) * 0.5
        df = pl.DataFrame({"y": y, "pred": pred})
        result = df.select(mape("y", "pred")).to_series()[0]
        expected = mean_absolute_percentage_error(y, pred)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_zero_target_excluded(self):
        """Rows where target == 0 should be excluded (not cause division by zero)."""
        df = pl.DataFrame({"y": [0.0, 1.0, 2.0], "pred": [0.5, 1.5, 2.5]})
        result = df.select(mape("y", "pred")).to_series()[0]
        # Only rows 1 and 2: |1-1.5|/1=0.5, |2-2.5|/2=0.25, mean=0.375
        assert result == pytest.approx(0.375, rel=1e-5)

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.1, 2.2, 2.8, 4.5],
            }
        )
        result = df.group_by("group").agg(mape("y", "pred")).sort("group")
        expected_a = mean_absolute_percentage_error([1.0, 2.0], [1.1, 2.2])
        expected_b = mean_absolute_percentage_error([3.0, 4.0], [2.8, 4.5])
        assert result["mape_y_pred"][0] == pytest.approx(expected_a, rel=1e-5)
        assert result["mape_y_pred"][1] == pytest.approx(expected_b, rel=1e-5)

    def test_weighted(self):
        np.random.seed(42)
        y = np.random.rand(100) * 10 + 1
        pred = y + np.random.randn(100) * 0.5
        weights = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": weights})
        result = df.select(mape("y", "pred", weight="w")).to_series()[0]
        expected = mean_absolute_percentage_error(y, pred, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-4)

    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[hypothesis.HealthCheck.differing_executors],
    )
    def test_mape_matches_sklearn_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=2, max_value=200), label="size")
        y = data.draw(
            arrays(
                dtype=np.float64,
                shape=size,
                elements=st.floats(
                    min_value=0.1, max_value=1e3, allow_nan=False, allow_infinity=False
                ),
            ),
            label="y",
        )
        pred = data.draw(
            arrays(
                dtype=np.float64,
                shape=size,
                elements=st.floats(
                    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
                ),
            ),
            label="pred",
        )
        # No zeros in y (min_value=0.1 ensures this)
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(mape("y", "pred")).to_series()[0]
        expected = mean_absolute_percentage_error(y, pred)
        assert result == pytest.approx(expected, rel=1e-5)
