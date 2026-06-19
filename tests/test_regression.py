"""Tests for regression metrics."""

import hypothesis
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)
from sklearn.metrics import (
    r2_score as sklearn_r2,
)

from polarbear import mae, mape, mse, r2_score, rmse


class TestMAE:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        result = df.select(mae("y", "pred")).to_series()[0]
        assert result == pytest.approx(0.0)

    def test_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        df = pl.DataFrame({"y": y, "pred": pred})
        result = df.select(mae("y", "pred")).to_series()[0]
        expected = mean_absolute_error(y, pred)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.1, 2.2, 2.8, 4.5],
            }
        )
        result = df.group_by("group").agg(mae("y", "pred")).sort("group")
        expected_a = mean_absolute_error([1.0, 2.0], [1.1, 2.2])
        expected_b = mean_absolute_error([3.0, 4.0], [2.8, 4.5])
        assert result["mae_y_pred"][0] == pytest.approx(expected_a, rel=1e-5)
        assert result["mae_y_pred"][1] == pytest.approx(expected_b, rel=1e-5)


class TestMSE:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        result = df.select(mse("y", "pred")).to_series()[0]
        assert result == pytest.approx(0.0)

    def test_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        df = pl.DataFrame({"y": y, "pred": pred})
        result = df.select(mse("y", "pred")).to_series()[0]
        expected = mean_squared_error(y, pred)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_known_values(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [2.0, 3.0, 4.0]})
        result = df.select(mse("y", "pred")).to_series()[0]
        assert result == pytest.approx(1.0, rel=1e-5)


class TestRMSE:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        result = df.select(rmse("y", "pred")).to_series()[0]
        assert result == pytest.approx(0.0)

    def test_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        df = pl.DataFrame({"y": y, "pred": pred})
        result = df.select(rmse("y", "pred")).to_series()[0]
        expected = root_mean_squared_error(y, pred)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_is_sqrt_of_mse(self):
        np.random.seed(42)
        y = np.random.randn(50)
        pred = y + np.random.randn(50) * 0.3
        df = pl.DataFrame({"y": y, "pred": pred})
        rmse_val = df.select(rmse("y", "pred")).to_series()[0]
        mse_val = df.select(mse("y", "pred")).to_series()[0]
        assert rmse_val == pytest.approx(mse_val**0.5, rel=1e-5)

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.1, 2.2, 2.8, 4.5],
            }
        )
        result = df.group_by("group").agg(rmse("y", "pred")).sort("group")
        expected_a = root_mean_squared_error([1.0, 2.0], [1.1, 2.2])
        expected_b = root_mean_squared_error([3.0, 4.0], [2.8, 4.5])
        assert result["rmse_y_pred"][0] == pytest.approx(expected_a, rel=1e-5)
        assert result["rmse_y_pred"][1] == pytest.approx(expected_b, rel=1e-5)


class TestRegressionWeighted:
    def test_mae_uniform_weights(self):
        np.random.seed(42)
        y = np.random.randn(50)
        pred = y + np.random.randn(50) * 0.3
        df = pl.DataFrame({"y": y, "pred": pred, "w": [1.0] * 50})
        weighted = df.select(mae("y", "pred", weight="w")).to_series()[0]
        unweighted = df.select(mae("y", "pred")).to_series()[0]
        assert weighted == pytest.approx(unweighted, rel=1e-5)

    def test_mae_weighted_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        weights = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": weights})
        result = df.select(mae("y", "pred", weight="w")).to_series()[0]
        expected = mean_absolute_error(y, pred, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_mse_weighted_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        weights = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": weights})
        result = df.select(mse("y", "pred", weight="w")).to_series()[0]
        expected = mean_squared_error(y, pred, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_rmse_weighted_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        weights = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": weights})
        result = df.select(rmse("y", "pred", weight="w")).to_series()[0]
        expected = root_mean_squared_error(y, pred, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-4)


class TestRegressionEdgeCases:
    def test_single_row(self):
        df = pl.DataFrame({"y": [3.0], "pred": [5.0]})
        assert df.select(mae("y", "pred")).to_series()[0] == pytest.approx(2.0)
        assert df.select(mse("y", "pred")).to_series()[0] == pytest.approx(4.0)
        assert df.select(rmse("y", "pred")).to_series()[0] == pytest.approx(2.0)

    def test_negative_values(self):
        y = [-10.0, -5.0, 0.0, 5.0, 10.0]
        pred = [-8.0, -6.0, 1.0, 4.0, 12.0]
        df = pl.DataFrame({"y": y, "pred": pred})
        assert df.select(mae("y", "pred")).to_series()[0] == pytest.approx(
            mean_absolute_error(y, pred), rel=1e-5
        )
        assert df.select(mse("y", "pred")).to_series()[0] == pytest.approx(
            mean_squared_error(y, pred), rel=1e-5
        )

    def test_large_values(self):
        y = [1e8, 2e8, 3e8]
        pred = [1.1e8, 1.9e8, 3.2e8]
        df = pl.DataFrame({"y": y, "pred": pred})
        assert df.select(mae("y", "pred")).to_series()[0] == pytest.approx(
            mean_absolute_error(y, pred), rel=1e-5
        )

    def test_mse_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.1, 2.2, 2.8, 4.5],
            }
        )
        result = df.group_by("group").agg(mse("y", "pred")).sort("group")
        expected_a = mean_squared_error([1.0, 2.0], [1.1, 2.2])
        expected_b = mean_squared_error([3.0, 4.0], [2.8, 4.5])
        assert result["mse_y_pred"][0] == pytest.approx(expected_a, rel=1e-5)
        assert result["mse_y_pred"][1] == pytest.approx(expected_b, rel=1e-5)


class TestRegressionHypothesis:
    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[hypothesis.HealthCheck.differing_executors],
    )
    def test_mae_matches_sklearn(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=1, max_value=500), label="size")
        y = data.draw(
            arrays(
                dtype=np.float64,
                shape=size,
                elements=st.floats(
                    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
                ),
            ),
            label="y",
        )
        pred = data.draw(
            arrays(
                dtype=np.float64,
                shape=size,
                elements=st.floats(
                    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
                ),
            ),
            label="pred",
        )
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(mae("y", "pred")).to_series()[0]
        expected = mean_absolute_error(y, pred)
        assert result == pytest.approx(expected, rel=1e-5)

    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[hypothesis.HealthCheck.differing_executors],
    )
    def test_mse_matches_sklearn(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=1, max_value=500), label="size")
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
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(mse("y", "pred")).to_series()[0]
        expected = mean_squared_error(y, pred)
        assert result == pytest.approx(expected, rel=1e-5)

    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[hypothesis.HealthCheck.differing_executors],
    )
    def test_rmse_matches_sklearn(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=1, max_value=500), label="size")
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
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(rmse("y", "pred")).to_series()[0]
        expected = root_mean_squared_error(y, pred)
        assert result == pytest.approx(expected, rel=1e-5)


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
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "pred": [3.0] * 5})
        result = df.select(r2_score("y", "pred")).to_series()[0]
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_worse_than_mean(self):
        """Bad predictions can give negative R²."""
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [3.0, 1.0, 2.0]})
        result = df.select(r2_score("y", "pred")).to_series()[0]
        assert result == pytest.approx(sklearn_r2([1.0, 2.0, 3.0], [3.0, 1.0, 2.0]), rel=1e-5)

    def test_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        df = pl.DataFrame({"y": y, "pred": pred})
        assert df.select(r2_score("y", "pred")).to_series()[0] == pytest.approx(
            sklearn_r2(y, pred), rel=1e-5
        )

    def test_constant_target_undefined(self):
        """When all target values are identical, R² is undefined."""
        df = pl.DataFrame({"y": [5.0, 5.0, 5.0], "pred": [4.0, 5.0, 6.0]})
        assert df.select(r2_score("y", "pred")).to_series()[0] is None

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.1, 2.2, 2.8, 4.5],
            }
        )
        result = df.group_by("group").agg(r2_score("y", "pred")).sort("group")
        assert result["r2_score_y_pred"][0] == pytest.approx(
            sklearn_r2([1.0, 2.0], [1.1, 2.2]), rel=1e-5
        )
        assert result["r2_score_y_pred"][1] == pytest.approx(
            sklearn_r2([3.0, 4.0], [2.8, 4.5]), rel=1e-5
        )

    def test_weighted_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        weights = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": weights})
        result = df.select(r2_score("y", "pred", weight="w")).to_series()[0]
        assert result == pytest.approx(sklearn_r2(y, pred, sample_weight=weights), rel=1e-4)

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
        hypothesis.assume(y.std() > 1e-10)
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(r2_score("y", "pred")).to_series()[0]
        assert result == pytest.approx(sklearn_r2(y, pred), rel=1e-5)


# ---------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------


class TestMAPE:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        assert df.select(mape("y", "pred")).to_series()[0] == pytest.approx(0.0)

    def test_known_values(self):
        df = pl.DataFrame({"y": [100.0, 200.0], "pred": [110.0, 190.0]})
        # |100-110|/100=0.1, |200-190|/200=0.05 → mean=0.075
        assert df.select(mape("y", "pred")).to_series()[0] == pytest.approx(0.075, rel=1e-5)

    def test_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.rand(100) * 10 + 1
        pred = y + np.random.randn(100) * 0.5
        df = pl.DataFrame({"y": y, "pred": pred})
        assert df.select(mape("y", "pred")).to_series()[0] == pytest.approx(
            mean_absolute_percentage_error(y, pred), rel=1e-5
        )

    def test_zero_target_excluded(self):
        """Rows where target == 0 are excluded (MAPE undefined for zero target)."""
        df = pl.DataFrame({"y": [0.0, 1.0, 2.0], "pred": [0.5, 1.5, 2.5]})
        # Only rows 1 and 2: |1-1.5|/1=0.5, |2-2.5|/2=0.25, mean=0.375
        assert df.select(mape("y", "pred")).to_series()[0] == pytest.approx(0.375, rel=1e-5)

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.1, 2.2, 2.8, 4.5],
            }
        )
        result = df.group_by("group").agg(mape("y", "pred")).sort("group")
        assert result["mape_y_pred"][0] == pytest.approx(
            mean_absolute_percentage_error([1.0, 2.0], [1.1, 2.2]), rel=1e-5
        )
        assert result["mape_y_pred"][1] == pytest.approx(
            mean_absolute_percentage_error([3.0, 4.0], [2.8, 4.5]), rel=1e-5
        )

    def test_weighted(self):
        np.random.seed(42)
        y = np.random.rand(100) * 10 + 1
        pred = y + np.random.randn(100) * 0.5
        weights = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": weights})
        result = df.select(mape("y", "pred", weight="w")).to_series()[0]
        assert result == pytest.approx(
            mean_absolute_percentage_error(y, pred, sample_weight=weights), rel=1e-4
        )

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
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(mape("y", "pred")).to_series()[0]
        assert result == pytest.approx(mean_absolute_percentage_error(y, pred), rel=1e-5)
