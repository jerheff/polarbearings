"""Tests for regression metrics."""

import hypothesis
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import (
    explained_variance_score as sklearn_ev,
)
from sklearn.metrics import (
    max_error as sklearn_max_error,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)
from sklearn.metrics import (
    mean_pinball_loss as sklearn_pinball,
)
from sklearn.metrics import (
    mean_squared_log_error as sklearn_msle,
)
from sklearn.metrics import (
    median_absolute_error as sklearn_medae,
)
from sklearn.metrics import (
    r2_score as sklearn_r2,
)
from sklearn.metrics import (
    root_mean_squared_log_error as sklearn_rmsle,
)

from polarbear import (
    explained_variance_score,
    huber_loss,
    log_cosh_loss,
    mae,
    mape,
    max_error,
    mean_pinball_loss,
    mean_squared_log_error,
    median_absolute_error,
    mse,
    r2_score,
    rmse,
    root_mean_squared_log_error,
    smape,
)


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


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------


def _nonneg_arrays(data, size, label, hi=1e4):
    return data.draw(
        arrays(
            dtype=np.float64,
            shape=size,
            elements=st.floats(min_value=0.0, max_value=hi, allow_nan=False, allow_infinity=False),
        ),
        label=label,
    )


def _real_arrays(data, size, label, lo=-1e3, hi=1e3):
    return data.draw(
        arrays(
            dtype=np.float64,
            shape=size,
            elements=st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False),
        ),
        label=label,
    )


_HYP_SETTINGS = settings(
    deadline=None,
    database=None,
    suppress_health_check=[hypothesis.HealthCheck.differing_executors],
)


# ---------------------------------------------------------------------------
# MSLE / RMSLE
# ---------------------------------------------------------------------------


class TestMSLE:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        assert df.select(mean_squared_log_error("y", "pred")).to_series()[0] == pytest.approx(0.0)

    def test_known_value(self):
        # single sample: (log1p(3) - log1p(2.5))**2
        df = pl.DataFrame({"y": [3.0], "pred": [2.5]})
        expected = (np.log1p(3.0) - np.log1p(2.5)) ** 2
        assert df.select(mean_squared_log_error("y", "pred")).to_series()[0] == pytest.approx(
            expected, rel=1e-9
        )

    def test_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.rand(100) * 10
        pred = np.abs(y + np.random.randn(100) * 0.5)
        df = pl.DataFrame({"y": y, "pred": pred})
        assert df.select(mean_squared_log_error("y", "pred")).to_series()[0] == pytest.approx(
            sklearn_msle(y, pred), rel=1e-6
        )

    def test_weighted_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.rand(100) * 10
        pred = np.abs(y + np.random.randn(100) * 0.5)
        w = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        result = df.select(mean_squared_log_error("y", "pred", weight="w")).to_series()[0]
        assert result == pytest.approx(sklearn_msle(y, pred, sample_weight=w), rel=1e-6)

    def test_negative_input_is_nan(self):
        """Negative inputs make log1p undefined -> NaN propagates (sklearn raises)."""
        df = pl.DataFrame({"y": [3.0, -5.0], "pred": [2.0, 1.0]})
        result = df.select(mean_squared_log_error("y", "pred")).to_series()[0]
        assert result is None or np.isnan(result)

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.1, 2.2, 2.8, 4.5],
            }
        )
        result = df.group_by("group").agg(mean_squared_log_error("y", "pred")).sort("group")
        col = "mean_squared_log_error_y_pred"
        assert result[col][0] == pytest.approx(sklearn_msle([1.0, 2.0], [1.1, 2.2]), rel=1e-6)
        assert result[col][1] == pytest.approx(sklearn_msle([3.0, 4.0], [2.8, 4.5]), rel=1e-6)

    @given(st.data())
    @_HYP_SETTINGS
    def test_matches_sklearn_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=1, max_value=300), label="size")
        y = _nonneg_arrays(data, size, "y")
        pred = _nonneg_arrays(data, size, "pred")
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(mean_squared_log_error("y", "pred")).to_series()[0]
        assert result == pytest.approx(sklearn_msle(y, pred), rel=1e-6, abs=1e-12)


class TestRMSLE:
    def test_is_sqrt_of_msle(self):
        np.random.seed(0)
        y = np.random.rand(50) * 5
        pred = np.abs(y + np.random.randn(50) * 0.3)
        df = pl.DataFrame({"y": y, "pred": pred})
        rmsle_val = df.select(root_mean_squared_log_error("y", "pred")).to_series()[0]
        msle_val = df.select(mean_squared_log_error("y", "pred")).to_series()[0]
        assert rmsle_val == pytest.approx(msle_val**0.5, rel=1e-9)

    def test_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.rand(100) * 10
        pred = np.abs(y + np.random.randn(100) * 0.5)
        df = pl.DataFrame({"y": y, "pred": pred})
        assert df.select(root_mean_squared_log_error("y", "pred")).to_series()[0] == pytest.approx(
            sklearn_rmsle(y, pred), rel=1e-6
        )

    def test_weighted_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.rand(100) * 10
        pred = np.abs(y + np.random.randn(100) * 0.5)
        w = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        result = df.select(root_mean_squared_log_error("y", "pred", weight="w")).to_series()[0]
        assert result == pytest.approx(sklearn_rmsle(y, pred, sample_weight=w), rel=1e-6)

    @given(st.data())
    @_HYP_SETTINGS
    def test_matches_sklearn_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=1, max_value=300), label="size")
        y = _nonneg_arrays(data, size, "y")
        pred = _nonneg_arrays(data, size, "pred")
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(root_mean_squared_log_error("y", "pred")).to_series()[0]
        assert result == pytest.approx(sklearn_rmsle(y, pred), rel=1e-6, abs=1e-12)


# ---------------------------------------------------------------------------
# max_error
# ---------------------------------------------------------------------------


class TestMaxError:
    def test_known_value(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.5, 5.0]})
        assert df.select(max_error("y", "pred")).to_series()[0] == pytest.approx(2.0)

    def test_perfect_predictions(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        assert df.select(max_error("y", "pred")).to_series()[0] == pytest.approx(0.0)

    def test_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        df = pl.DataFrame({"y": y, "pred": pred})
        assert df.select(max_error("y", "pred")).to_series()[0] == pytest.approx(
            sklearn_max_error(y, pred), rel=1e-9
        )

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.1, 2.2, 2.8, 4.5],
            }
        )
        result = df.group_by("group").agg(max_error("y", "pred")).sort("group")
        assert result["max_error_y_pred"][0] == pytest.approx(
            sklearn_max_error([1.0, 2.0], [1.1, 2.2]), rel=1e-9
        )
        assert result["max_error_y_pred"][1] == pytest.approx(
            sklearn_max_error([3.0, 4.0], [2.8, 4.5]), rel=1e-9
        )

    @given(st.data())
    @_HYP_SETTINGS
    def test_matches_sklearn_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=1, max_value=300), label="size")
        y = _real_arrays(data, size, "y")
        pred = _real_arrays(data, size, "pred")
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(max_error("y", "pred")).to_series()[0]
        assert result == pytest.approx(sklearn_max_error(y, pred), rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# median_absolute_error
# ---------------------------------------------------------------------------


class TestMedianAbsoluteError:
    def test_known_value(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "pred": [1.0, 2.0, 3.0, 7.0]})
        # abs errs = [0,0,0,3] -> median = 0.0
        assert df.select(median_absolute_error("y", "pred")).to_series()[0] == pytest.approx(0.0)

    def test_known_value_odd(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [2.0, 5.0, 3.5]})
        # abs errs = [1.0, 3.0, 0.5] -> median = 1.0
        assert df.select(median_absolute_error("y", "pred")).to_series()[0] == pytest.approx(1.0)

    def test_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(101)
        pred = y + np.random.randn(101) * 0.5
        df = pl.DataFrame({"y": y, "pred": pred})
        assert df.select(median_absolute_error("y", "pred")).to_series()[0] == pytest.approx(
            sklearn_medae(y, pred), rel=1e-9
        )

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "y": [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
                "pred": [1.5, 2.0, 4.0, 2.0, 4.0, 6.5],
            }
        )
        result = df.group_by("group").agg(median_absolute_error("y", "pred")).sort("group")
        assert result["median_absolute_error_y_pred"][0] == pytest.approx(
            sklearn_medae([1.0, 2.0, 3.0], [1.5, 2.0, 4.0]), rel=1e-9
        )
        assert result["median_absolute_error_y_pred"][1] == pytest.approx(
            sklearn_medae([3.0, 4.0, 5.0], [2.0, 4.0, 6.5]), rel=1e-9
        )

    @given(st.data())
    @_HYP_SETTINGS
    def test_matches_sklearn_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=1, max_value=300), label="size")
        y = _real_arrays(data, size, "y")
        pred = _real_arrays(data, size, "pred")
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(median_absolute_error("y", "pred")).to_series()[0]
        assert result == pytest.approx(sklearn_medae(y, pred), rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# explained_variance_score
# ---------------------------------------------------------------------------


class TestExplainedVariance:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        assert df.select(explained_variance_score("y", "pred")).to_series()[0] == pytest.approx(1.0)

    def test_constant_offset_is_one(self):
        """Explained variance ignores a constant offset (unlike R²)."""
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [2.0, 3.0, 4.0]})
        assert df.select(explained_variance_score("y", "pred")).to_series()[0] == pytest.approx(1.0)

    def test_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        df = pl.DataFrame({"y": y, "pred": pred})
        assert df.select(explained_variance_score("y", "pred")).to_series()[0] == pytest.approx(
            sklearn_ev(y, pred), rel=1e-6
        )

    def test_weighted_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        w = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        result = df.select(explained_variance_score("y", "pred", weight="w")).to_series()[0]
        assert result == pytest.approx(sklearn_ev(y, pred, sample_weight=w), rel=1e-6)

    def test_constant_target_undefined(self):
        df = pl.DataFrame({"y": [5.0, 5.0, 5.0], "pred": [4.0, 5.0, 6.0]})
        assert df.select(explained_variance_score("y", "pred")).to_series()[0] is None

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "y": [1.0, 2.0, 3.0, 3.0, 5.0, 4.0],
                "pred": [1.1, 2.2, 2.8, 2.5, 5.1, 4.2],
            }
        )
        result = df.group_by("group").agg(explained_variance_score("y", "pred")).sort("group")
        col = "explained_variance_score_y_pred"
        assert result[col][0] == pytest.approx(
            sklearn_ev([1.0, 2.0, 3.0], [1.1, 2.2, 2.8]), rel=1e-6
        )
        assert result[col][1] == pytest.approx(
            sklearn_ev([3.0, 5.0, 4.0], [2.5, 5.1, 4.2]), rel=1e-6
        )

    @given(st.data())
    @_HYP_SETTINGS
    def test_matches_sklearn_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=2, max_value=300), label="size")
        y = _real_arrays(data, size, "y")
        pred = _real_arrays(data, size, "pred")
        hypothesis.assume(y.std() > 1e-6)
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(explained_variance_score("y", "pred")).to_series()[0]
        assert result == pytest.approx(sklearn_ev(y, pred), rel=1e-5, abs=1e-9)


# ---------------------------------------------------------------------------
# mean_pinball_loss
# ---------------------------------------------------------------------------


class TestMeanPinballLoss:
    def test_known_value(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.5, 1.5, 3.5]})
        # alpha=0.5: errs=[-0.5,0.5,-0.5] -> losses=[0.25,0.25,0.25] -> 0.25
        assert df.select(mean_pinball_loss("y", "pred", alpha=0.5)).to_series()[0] == pytest.approx(
            0.25
        )

    def test_half_alpha_is_half_mae(self):
        np.random.seed(0)
        y = np.random.randn(50)
        pred = y + np.random.randn(50) * 0.4
        df = pl.DataFrame({"y": y, "pred": pred})
        pin = df.select(mean_pinball_loss("y", "pred", alpha=0.5)).to_series()[0]
        m = df.select(mae("y", "pred")).to_series()[0]
        assert pin == pytest.approx(m / 2, rel=1e-9)

    def test_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        df = pl.DataFrame({"y": y, "pred": pred})
        for alpha in (0.1, 0.5, 0.9):
            assert df.select(mean_pinball_loss("y", "pred", alpha=alpha)).to_series()[
                0
            ] == pytest.approx(sklearn_pinball(y, pred, alpha=alpha), rel=1e-6)

    def test_weighted_matches_sklearn(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        w = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        for alpha in (0.2, 0.5, 0.8):
            result = df.select(mean_pinball_loss("y", "pred", alpha=alpha, weight="w")).to_series()[
                0
            ]
            assert result == pytest.approx(
                sklearn_pinball(y, pred, alpha=alpha, sample_weight=w), rel=1e-6
            )

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.1, 2.2, 2.8, 4.5],
            }
        )
        result = df.group_by("group").agg(mean_pinball_loss("y", "pred", alpha=0.7)).sort("group")
        col = "mean_pinball_loss_y_pred"
        assert result[col][0] == pytest.approx(
            sklearn_pinball([1.0, 2.0], [1.1, 2.2], alpha=0.7), rel=1e-6
        )
        assert result[col][1] == pytest.approx(
            sklearn_pinball([3.0, 4.0], [2.8, 4.5], alpha=0.7), rel=1e-6
        )

    @given(st.data())
    @_HYP_SETTINGS
    def test_matches_sklearn_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=1, max_value=300), label="size")
        alpha = data.draw(st.floats(min_value=0.05, max_value=0.95), label="alpha")
        y = _real_arrays(data, size, "y")
        pred = _real_arrays(data, size, "pred")
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(mean_pinball_loss("y", "pred", alpha=alpha)).to_series()[0]
        assert result == pytest.approx(sklearn_pinball(y, pred, alpha=alpha), rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# sMAPE (no sklearn analog)
# ---------------------------------------------------------------------------


def _np_smape(y, pred, w=None):
    denom = np.abs(y) + np.abs(pred)
    safe_denom = np.where(denom == 0, 1.0, denom)
    per = np.where(denom == 0, 0.0, 2.0 * np.abs(y - pred) / safe_denom)
    if w is None:
        return per.mean()
    return (w * per).sum() / w.sum()


class TestSMAPE:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        assert df.select(smape("y", "pred")).to_series()[0] == pytest.approx(0.0)

    def test_known_value(self):
        df = pl.DataFrame({"y": [100.0], "pred": [110.0]})
        # 2*10/210
        assert df.select(smape("y", "pred")).to_series()[0] == pytest.approx(2 * 10 / 210, rel=1e-9)

    def test_zero_over_zero_is_zero(self):
        """0/0 case: both y and pred zero -> contribution defined as 0."""
        df = pl.DataFrame({"y": [0.0, 100.0], "pred": [0.0, 110.0]})
        # first sample contributes 0, second 2*10/210; mean of [0, 2*10/210]
        expected = (0.0 + 2 * 10 / 210) / 2
        assert df.select(smape("y", "pred")).to_series()[0] == pytest.approx(expected, rel=1e-9)

    def test_matches_numpy_reference(self):
        np.random.seed(42)
        y = np.random.randn(100) * 5
        pred = y + np.random.randn(100) * 0.5
        df = pl.DataFrame({"y": y, "pred": pred})
        assert df.select(smape("y", "pred")).to_series()[0] == pytest.approx(
            _np_smape(y, pred), rel=1e-9
        )

    def test_weighted_matches_numpy_reference(self):
        np.random.seed(42)
        y = np.random.randn(100) * 5
        pred = y + np.random.randn(100) * 0.5
        w = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        result = df.select(smape("y", "pred", weight="w")).to_series()[0]
        assert result == pytest.approx(_np_smape(y, pred, w), rel=1e-9)

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.1, 2.2, 2.8, 4.5],
            }
        )
        result = df.group_by("group").agg(smape("y", "pred")).sort("group")
        assert result["smape_y_pred"][0] == pytest.approx(
            _np_smape(np.array([1.0, 2.0]), np.array([1.1, 2.2])), rel=1e-9
        )
        assert result["smape_y_pred"][1] == pytest.approx(
            _np_smape(np.array([3.0, 4.0]), np.array([2.8, 4.5])), rel=1e-9
        )

    @given(st.data())
    @_HYP_SETTINGS
    def test_matches_numpy_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=1, max_value=300), label="size")
        y = _real_arrays(data, size, "y")
        pred = _real_arrays(data, size, "pred")
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(smape("y", "pred")).to_series()[0]
        assert result == pytest.approx(_np_smape(y, pred), rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# Huber loss (no sklearn analog)
# ---------------------------------------------------------------------------


def _np_huber(y, pred, delta=1.0, w=None):
    err = y - pred
    per = np.where(np.abs(err) <= delta, 0.5 * err**2, delta * (np.abs(err) - 0.5 * delta))
    if w is None:
        return per.mean()
    return (w * per).sum() / w.sum()


class TestHuberLoss:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        assert df.select(huber_loss("y", "pred")).to_series()[0] == pytest.approx(0.0)

    def test_known_value(self):
        df = pl.DataFrame({"y": [0.0, 0.0], "pred": [0.5, 3.0]})
        # |0.5|<=1 -> 0.5*0.25=0.125; |3|>1 -> 1*(3-0.5)=2.5; mean=1.3125
        assert df.select(huber_loss("y", "pred", delta=1.0)).to_series()[0] == pytest.approx(1.3125)

    def test_matches_numpy_reference(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.8
        df = pl.DataFrame({"y": y, "pred": pred})
        for delta in (0.5, 1.0, 2.0):
            assert df.select(huber_loss("y", "pred", delta=delta)).to_series()[0] == pytest.approx(
                _np_huber(y, pred, delta), rel=1e-9
            )

    def test_weighted_matches_numpy_reference(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.8
        w = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        result = df.select(huber_loss("y", "pred", delta=1.5, weight="w")).to_series()[0]
        assert result == pytest.approx(_np_huber(y, pred, 1.5, w), rel=1e-9)

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.1, 5.0, 2.8, 8.0],
            }
        )
        result = df.group_by("group").agg(huber_loss("y", "pred")).sort("group")
        assert result["huber_loss_y_pred"][0] == pytest.approx(
            _np_huber(np.array([1.0, 2.0]), np.array([1.1, 5.0])), rel=1e-9
        )
        assert result["huber_loss_y_pred"][1] == pytest.approx(
            _np_huber(np.array([3.0, 4.0]), np.array([2.8, 8.0])), rel=1e-9
        )

    @given(st.data())
    @_HYP_SETTINGS
    def test_matches_numpy_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=1, max_value=300), label="size")
        delta = data.draw(st.floats(min_value=0.1, max_value=5.0), label="delta")
        y = _real_arrays(data, size, "y")
        pred = _real_arrays(data, size, "pred")
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(huber_loss("y", "pred", delta=delta)).to_series()[0]
        assert result == pytest.approx(_np_huber(y, pred, delta), rel=1e-7, abs=1e-9)


# ---------------------------------------------------------------------------
# Log-cosh loss (no sklearn analog)
# ---------------------------------------------------------------------------


def _np_log_cosh(y, pred, w=None):
    e = np.abs(pred - y)
    per = e + np.log1p(np.exp(-2.0 * e)) - np.log(2.0)
    if w is None:
        return per.mean()
    return (w * per).sum() / w.sum()


class TestLogCoshLoss:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]})
        assert df.select(log_cosh_loss("y", "pred")).to_series()[0] == pytest.approx(0.0)

    def test_known_value(self):
        df = pl.DataFrame({"y": [0.0], "pred": [1.0]})
        assert df.select(log_cosh_loss("y", "pred")).to_series()[0] == pytest.approx(
            np.log(np.cosh(1.0)), rel=1e-9
        )

    def test_numerically_stable_large_residual(self):
        """Naive cosh(e).log() overflows for large e; stable form gives |e|-log2."""
        df = pl.DataFrame({"y": [0.0], "pred": [1000.0]})
        result = df.select(log_cosh_loss("y", "pred")).to_series()[0]
        assert np.isfinite(result)
        assert result == pytest.approx(1000.0 - np.log(2.0), rel=1e-12)

    def test_matches_numpy_reference(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        df = pl.DataFrame({"y": y, "pred": pred})
        assert df.select(log_cosh_loss("y", "pred")).to_series()[0] == pytest.approx(
            _np_log_cosh(y, pred), rel=1e-9
        )

    def test_weighted_matches_numpy_reference(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y + np.random.randn(100) * 0.5
        w = np.random.rand(100) + 0.5
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        result = df.select(log_cosh_loss("y", "pred", weight="w")).to_series()[0]
        assert result == pytest.approx(_np_log_cosh(y, pred, w), rel=1e-9)

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.1, 2.2, 2.8, 4.5],
            }
        )
        result = df.group_by("group").agg(log_cosh_loss("y", "pred")).sort("group")
        assert result["log_cosh_loss_y_pred"][0] == pytest.approx(
            _np_log_cosh(np.array([1.0, 2.0]), np.array([1.1, 2.2])), rel=1e-9
        )
        assert result["log_cosh_loss_y_pred"][1] == pytest.approx(
            _np_log_cosh(np.array([3.0, 4.0]), np.array([2.8, 4.5])), rel=1e-9
        )

    @given(st.data())
    @_HYP_SETTINGS
    def test_matches_numpy_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=1, max_value=300), label="size")
        y = _real_arrays(data, size, "y")
        pred = _real_arrays(data, size, "pred")
        df = pl.DataFrame({"y": y.tolist(), "pred": pred.tolist()})
        result = df.select(log_cosh_loss("y", "pred")).to_series()[0]
        assert result == pytest.approx(_np_log_cosh(y, pred), rel=1e-7, abs=1e-9)
