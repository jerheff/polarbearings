"""Tests for the D² ("explained deviance") score family."""

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sklearn.metrics import (
    d2_absolute_error_score as sk_d2_abs,
)
from sklearn.metrics import (
    d2_pinball_score as sk_d2_pinball,
)
from sklearn.metrics import (
    d2_tweedie_score as sk_d2_tweedie,
)

from polarbearings.regression import (
    d2_absolute_error_score,
    d2_pinball_score,
    d2_tweedie_score,
)


class TestD2TweedieScore:
    def test_perfect_prediction(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "pred": [1.0, 2.0, 3.0, 4.0]})
        result = df.select(d2_tweedie_score("y", "pred")).to_series()[0]
        assert result == pytest.approx(1.0)

    def test_baseline_prediction_is_zero(self):
        # Predicting the (unweighted) mean of y -> D² == 0.
        y = np.array([1.0, 2.0, 3.0, 6.0])
        mean = y.mean()
        df = pl.DataFrame({"y": y, "pred": np.full_like(y, mean)})
        result = df.select(d2_tweedie_score("y", "pred", power=0.0)).to_series()[0]
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_constant_target_is_null(self):
        df = pl.DataFrame({"y": [2.0, 2.0, 2.0], "pred": [1.0, 2.0, 3.0]})
        result = df.select(d2_tweedie_score("y", "pred", power=0.0)).to_series()[0]
        assert result is None

    def test_matches_sklearn(self):
        rng = np.random.default_rng(10)
        y = rng.uniform(0.3, 5.0, size=50)
        pred = rng.uniform(0.3, 5.0, size=50)
        df = pl.DataFrame({"y": y, "pred": pred})
        for power in (0.0, 1.0, 1.5, 2.0, 3.0):
            result = df.select(d2_tweedie_score("y", "pred", power=power)).to_series()[0]
            assert result == pytest.approx(sk_d2_tweedie(y, pred, power=power), rel=1e-9)

    def test_matches_sklearn_weighted(self):
        rng = np.random.default_rng(11)
        y = rng.uniform(0.3, 5.0, size=50)
        pred = rng.uniform(0.3, 5.0, size=50)
        w = rng.uniform(0.1, 3.0, size=50)
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        for power in (0.0, 1.0, 2.0):
            result = df.select(d2_tweedie_score("y", "pred", power=power, weight="w")).to_series()[
                0
            ]
            assert result == pytest.approx(
                sk_d2_tweedie(y, pred, power=power, sample_weight=w), rel=1e-9
            )

    def test_weight_expression(self):
        rng = np.random.default_rng(12)
        y = rng.uniform(0.3, 5.0, size=30)
        pred = rng.uniform(0.3, 5.0, size=30)
        w = rng.uniform(0.1, 3.0, size=30)
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        result = df.select(
            d2_tweedie_score("y", "pred", power=1.0, weight=pl.col("w") * 2.0)
        ).to_series()[0]
        # Scaling all weights leaves the ratio unchanged.
        assert result == pytest.approx(sk_d2_tweedie(y, pred, power=1.0, sample_weight=w), rel=1e-9)

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "y": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
                "pred": [1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
            }
        )
        result = df.group_by("group").agg(d2_tweedie_score("y", "pred")).sort("group")
        rows = result.to_dicts()
        assert rows[0]["d2_tweedie_score_y_pred"] == pytest.approx(1.0)
        b = sk_d2_tweedie(np.array([2.0, 4.0, 6.0]), np.array([3.0, 3.0, 3.0]), power=0.0)
        assert rows[1]["d2_tweedie_score_y_pred"] == pytest.approx(b, rel=1e-9)


class TestD2AbsoluteErrorScore:
    def test_perfect_prediction(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "pred": [1.0, 2.0, 3.0, 4.0]})
        result = df.select(d2_absolute_error_score("y", "pred")).to_series()[0]
        assert result == pytest.approx(1.0)

    def test_baseline_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
        median = float(np.median(y))
        df = pl.DataFrame({"y": y, "pred": np.full_like(y, median)})
        result = df.select(d2_absolute_error_score("y", "pred")).to_series()[0]
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_constant_target_is_null(self):
        df = pl.DataFrame({"y": [5.0, 5.0, 5.0], "pred": [1.0, 5.0, 9.0]})
        result = df.select(d2_absolute_error_score("y", "pred")).to_series()[0]
        assert result is None

    def test_matches_sklearn(self):
        rng = np.random.default_rng(20)
        for n in (5, 6, 7, 8, 40):
            y = rng.uniform(-5.0, 5.0, size=n)
            pred = rng.uniform(-5.0, 5.0, size=n)
            df = pl.DataFrame({"y": y, "pred": pred})
            result = df.select(d2_absolute_error_score("y", "pred")).to_series()[0]
            assert result == pytest.approx(sk_d2_abs(y, pred), rel=1e-9)

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
                "pred": [1.0, 2.0, 3.0, 4.0, 2.5, 2.5, 2.5, 2.5],
            }
        )
        result = df.group_by("group").agg(d2_absolute_error_score("y", "pred")).sort("group")
        rows = result.to_dicts()
        assert rows[0]["d2_absolute_error_score_y_pred"] == pytest.approx(1.0)
        b = sk_d2_abs(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.5, 2.5, 2.5, 2.5]))
        assert rows[1]["d2_absolute_error_score_y_pred"] == pytest.approx(b, rel=1e-9)


class TestD2PinballScore:
    def test_perfect_prediction(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "pred": [1.0, 2.0, 3.0, 4.0]})
        result = df.select(d2_pinball_score("y", "pred", alpha=0.7)).to_series()[0]
        assert result == pytest.approx(1.0)

    def test_baseline_prediction_is_zero(self):
        # Predicting the alpha-quantile (inverted_cdf) of y -> D² == 0.
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        alpha = 0.7
        q = float(np.percentile(y, alpha * 100, method="inverted_cdf"))
        df = pl.DataFrame({"y": y, "pred": np.full_like(y, q)})
        result = df.select(d2_pinball_score("y", "pred", alpha=alpha)).to_series()[0]
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_constant_target_is_null(self):
        df = pl.DataFrame({"y": [3.0, 3.0, 3.0], "pred": [1.0, 3.0, 9.0]})
        result = df.select(d2_pinball_score("y", "pred", alpha=0.5)).to_series()[0]
        assert result is None

    def test_default_alpha_equals_absolute_error(self):
        rng = np.random.default_rng(30)
        y = rng.uniform(-5.0, 5.0, size=20)
        pred = rng.uniform(-5.0, 5.0, size=20)
        df = pl.DataFrame({"y": y, "pred": pred})
        pinball = df.select(d2_pinball_score("y", "pred")).to_series()[0]
        abs_err = df.select(d2_absolute_error_score("y", "pred")).to_series()[0]
        assert pinball == pytest.approx(abs_err, rel=1e-9)

    def test_matches_sklearn(self):
        rng = np.random.default_rng(31)
        for alpha in (0.1, 0.3, 0.5, 0.7, 0.9):
            for n in (5, 6, 7, 8, 40):
                y = rng.uniform(-5.0, 5.0, size=n)
                pred = rng.uniform(-5.0, 5.0, size=n)
                df = pl.DataFrame({"y": y, "pred": pred})
                result = df.select(d2_pinball_score("y", "pred", alpha=alpha)).to_series()[0]
                assert result == pytest.approx(sk_d2_pinball(y, pred, alpha=alpha), rel=1e-9)

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "pred": [1.0, 2.0, 3.0, 4.0, 6.0, 6.0, 6.0, 6.0],
            }
        )
        result = df.group_by("group").agg(d2_pinball_score("y", "pred", alpha=0.3)).sort("group")
        rows = result.to_dicts()
        assert rows[0]["d2_pinball_score_y_pred"] == pytest.approx(1.0)
        b = sk_d2_pinball(np.array([5.0, 6.0, 7.0, 8.0]), np.array([6.0, 6.0, 6.0, 6.0]), alpha=0.3)
        assert rows[1]["d2_pinball_score_y_pred"] == pytest.approx(b, rel=1e-9)


@given(
    size=st.integers(min_value=2, max_value=60),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    power=st.sampled_from([0.0, 1.0, 1.5, 2.0, 3.0]),
)
@settings(
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.differing_executors],
)
def test_d2_tweedie_against_sklearn(size, seed, power):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.1, 10.0, size=size)
    pred = rng.uniform(0.1, 10.0, size=size)
    df = pl.DataFrame({"y": y, "pred": pred})
    result = df.select(d2_tweedie_score("y", "pred", power=power)).to_series()[0]
    expected = sk_d2_tweedie(y, pred, power=power)
    if result is None:
        # Degenerate denominator (constant y) -> null in our impl.
        assert np.isclose(y.min(), y.max())
    else:
        assert result == pytest.approx(expected, rel=1e-9)


@given(
    size=st.integers(min_value=2, max_value=60),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.differing_executors],
)
def test_d2_absolute_error_against_sklearn(size, seed):
    rng = np.random.default_rng(seed)
    y = rng.uniform(-10.0, 10.0, size=size)
    pred = rng.uniform(-10.0, 10.0, size=size)
    df = pl.DataFrame({"y": y, "pred": pred})
    result = df.select(d2_absolute_error_score("y", "pred")).to_series()[0]
    expected = sk_d2_abs(y, pred)
    if result is None:
        assert np.isclose(y.min(), y.max())
    else:
        assert result == pytest.approx(expected, rel=1e-9)


@given(
    size=st.integers(min_value=2, max_value=60),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    alpha=st.sampled_from([0.1, 0.3, 0.5, 0.7, 0.9]),
)
@settings(
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.differing_executors],
)
def test_d2_pinball_against_sklearn(size, seed, alpha):
    rng = np.random.default_rng(seed)
    y = rng.uniform(-10.0, 10.0, size=size)
    pred = rng.uniform(-10.0, 10.0, size=size)
    df = pl.DataFrame({"y": y, "pred": pred})
    result = df.select(d2_pinball_score("y", "pred", alpha=alpha)).to_series()[0]
    expected = sk_d2_pinball(y, pred, alpha=alpha)
    if result is None:
        assert np.isclose(y.min(), y.max())
    else:
        assert result == pytest.approx(expected, rel=1e-9)
