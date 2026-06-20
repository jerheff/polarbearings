"""Tests for the Tweedie deviance family (Tweedie, Poisson, Gamma)."""

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sklearn.metrics import (
    mean_gamma_deviance as sk_gamma,
)
from sklearn.metrics import (
    mean_poisson_deviance as sk_poisson,
)
from sklearn.metrics import (
    mean_tweedie_deviance as sk_tweedie,
)

from polarbearings.regression import (
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_tweedie_deviance,
)


def _reference_tweedie(
    y: np.ndarray, pred: np.ndarray, power: float, weight: np.ndarray | None = None
) -> float:
    """NumPy reference per-sample Tweedie unit deviance, then (weighted) mean."""
    y = y.astype(float)
    pred = pred.astype(float)
    if power == 0:
        dev = (y - pred) ** 2
    elif power == 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            y_log = np.where(y == 0, 0.0, y * np.log(y / pred))
        dev = 2.0 * (y_log - y + pred)
    elif power == 2:
        dev = 2.0 * (np.log(pred / y) + y / pred - 1.0)
    else:
        dev = 2.0 * (
            y ** (2.0 - power) / ((1.0 - power) * (2.0 - power))
            - y * pred ** (1.0 - power) / (1.0 - power)
            + pred ** (2.0 - power) / (2.0 - power)
        )
    if weight is None:
        return float(dev.mean())
    return float((dev * weight).sum() / weight.sum())


class TestMeanTweedieDeviance:
    def test_power_zero_equals_mse(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.5, 1.5, 5.0]})
        result = df.select(mean_tweedie_deviance("y", "pred", power=0.0)).to_series()[0]
        # (0.25 + 0.25 + 4.0) / 3
        assert result == pytest.approx((0.25 + 0.25 + 4.0) / 3.0)

    def test_basic_poisson_value(self):
        df = pl.DataFrame({"y": [2.0, 0.0, 1.0, 4.0], "pred": [0.5, 0.5, 2.0, 2.0]})
        result = df.select(mean_tweedie_deviance("y", "pred", power=1.0)).to_series()[0]
        expected = _reference_tweedie(
            np.array([2.0, 0.0, 1.0, 4.0]), np.array([0.5, 0.5, 2.0, 2.0]), 1.0
        )
        assert result == pytest.approx(expected)

    def test_basic_gamma_value(self):
        df = pl.DataFrame({"y": [2.0, 0.5, 1.0, 4.0], "pred": [0.5, 0.5, 2.0, 2.0]})
        result = df.select(mean_tweedie_deviance("y", "pred", power=2.0)).to_series()[0]
        expected = _reference_tweedie(
            np.array([2.0, 0.5, 1.0, 4.0]), np.array([0.5, 0.5, 2.0, 2.0]), 2.0
        )
        assert result == pytest.approx(expected)

    def test_general_power(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.5, 2.5, 2.0, 3.0])
        df = pl.DataFrame({"y": y, "pred": pred})
        for power in (1.5, 3.0, -0.5):
            result = df.select(mean_tweedie_deviance("y", "pred", power=power)).to_series()[0]
            assert result == pytest.approx(_reference_tweedie(y, pred, power))

    def test_poisson_zero_target(self):
        # y == 0 must contribute 2*pred (the y*log term -> 0).
        df = pl.DataFrame({"y": [0.0, 0.0], "pred": [1.0, 3.0]})
        result = df.select(mean_tweedie_deviance("y", "pred", power=1.0)).to_series()[0]
        # 2*(0 - 0 + 1) and 2*(0 - 0 + 3) -> mean = (2 + 6)/2
        assert result == pytest.approx(4.0)

    def test_weighted(self):
        y = np.array([2.0, 0.5, 1.0, 4.0])
        pred = np.array([0.5, 0.5, 2.0, 2.0])
        w = np.array([1.0, 2.0, 0.5, 3.0])
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        result = df.select(mean_tweedie_deviance("y", "pred", power=1.0, weight="w")).to_series()[0]
        assert result == pytest.approx(_reference_tweedie(y, pred, 1.0, w))

    def test_unit_weights_match_unweighted(self):
        y = np.array([2.0, 0.5, 1.0, 4.0])
        pred = np.array([0.5, 0.5, 2.0, 2.0])
        df = pl.DataFrame({"y": y, "pred": pred, "w": np.ones(4)})
        weighted = df.select(mean_tweedie_deviance("y", "pred", power=2.0, weight="w")).to_series()[
            0
        ]
        unweighted = df.select(mean_tweedie_deviance("y", "pred", power=2.0)).to_series()[0]
        assert weighted == pytest.approx(unweighted)

    def test_matches_sklearn(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(0.3, 5.0, size=40)
        pred = rng.uniform(0.3, 5.0, size=40)
        df = pl.DataFrame({"y": y, "pred": pred})
        for power in (0.0, 1.0, 1.5, 2.0, 3.0):
            result = df.select(mean_tweedie_deviance("y", "pred", power=power)).to_series()[0]
            assert result == pytest.approx(sk_tweedie(y, pred, power=power), rel=1e-9)

    def test_matches_sklearn_weighted(self):
        rng = np.random.default_rng(2)
        y = rng.uniform(0.3, 5.0, size=40)
        pred = rng.uniform(0.3, 5.0, size=40)
        w = rng.uniform(0.1, 3.0, size=40)
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        for power in (0.0, 1.0, 2.0):
            result = df.select(
                mean_tweedie_deviance("y", "pred", power=power, weight="w")
            ).to_series()[0]
            assert result == pytest.approx(
                sk_tweedie(y, pred, power=power, sample_weight=w), rel=1e-9
            )

    def test_weight_expression(self):
        y = np.array([2.0, 0.5, 1.0, 4.0])
        pred = np.array([0.5, 0.5, 2.0, 2.0])
        w = np.array([1.0, 2.0, 0.5, 3.0])
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        result = df.select(
            mean_tweedie_deviance("y", "pred", power=1.0, weight=pl.col("w") * 2.0)
        ).to_series()[0]
        assert result == pytest.approx(_reference_tweedie(y, pred, 1.0, w))

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "y": [2.0, 1.0, 3.0, 4.0],
                "pred": [1.0, 2.0, 2.0, 5.0],
            }
        )
        result = (
            df.group_by("group").agg(mean_tweedie_deviance("y", "pred", power=1.0)).sort("group")
        )
        rows = result.to_dicts()
        a = _reference_tweedie(np.array([2.0, 1.0]), np.array([1.0, 2.0]), 1.0)
        b = _reference_tweedie(np.array([3.0, 4.0]), np.array([2.0, 5.0]), 1.0)
        assert rows[0]["mean_tweedie_deviance_y_pred"] == pytest.approx(a)
        assert rows[1]["mean_tweedie_deviance_y_pred"] == pytest.approx(b)


class TestMeanPoissonDeviance:
    def test_equals_tweedie_power_one(self):
        rng = np.random.default_rng(3)
        y = rng.uniform(0.0, 5.0, size=30)
        pred = rng.uniform(0.3, 5.0, size=30)
        df = pl.DataFrame({"y": y, "pred": pred})
        wrapper = df.select(mean_poisson_deviance("y", "pred")).to_series()[0]
        general = df.select(mean_tweedie_deviance("y", "pred", power=1.0)).to_series()[0]
        assert wrapper == pytest.approx(general)

    def test_matches_sklearn(self):
        rng = np.random.default_rng(4)
        y = rng.uniform(0.0, 5.0, size=40)
        pred = rng.uniform(0.3, 5.0, size=40)
        df = pl.DataFrame({"y": y, "pred": pred})
        result = df.select(mean_poisson_deviance("y", "pred")).to_series()[0]
        assert result == pytest.approx(sk_poisson(y, pred), rel=1e-9)

    def test_matches_sklearn_weighted(self):
        rng = np.random.default_rng(5)
        y = rng.uniform(0.0, 5.0, size=40)
        pred = rng.uniform(0.3, 5.0, size=40)
        w = rng.uniform(0.1, 3.0, size=40)
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        result = df.select(mean_poisson_deviance("y", "pred", weight="w")).to_series()[0]
        assert result == pytest.approx(sk_poisson(y, pred, sample_weight=w), rel=1e-9)


class TestMeanGammaDeviance:
    def test_equals_tweedie_power_two(self):
        rng = np.random.default_rng(6)
        y = rng.uniform(0.3, 5.0, size=30)
        pred = rng.uniform(0.3, 5.0, size=30)
        df = pl.DataFrame({"y": y, "pred": pred})
        wrapper = df.select(mean_gamma_deviance("y", "pred")).to_series()[0]
        general = df.select(mean_tweedie_deviance("y", "pred", power=2.0)).to_series()[0]
        assert wrapper == pytest.approx(general)

    def test_matches_sklearn(self):
        rng = np.random.default_rng(7)
        y = rng.uniform(0.3, 5.0, size=40)
        pred = rng.uniform(0.3, 5.0, size=40)
        df = pl.DataFrame({"y": y, "pred": pred})
        result = df.select(mean_gamma_deviance("y", "pred")).to_series()[0]
        assert result == pytest.approx(sk_gamma(y, pred), rel=1e-9)

    def test_matches_sklearn_weighted(self):
        rng = np.random.default_rng(8)
        y = rng.uniform(0.3, 5.0, size=40)
        pred = rng.uniform(0.3, 5.0, size=40)
        w = rng.uniform(0.1, 3.0, size=40)
        df = pl.DataFrame({"y": y, "pred": pred, "w": w})
        result = df.select(mean_gamma_deviance("y", "pred", weight="w")).to_series()[0]
        assert result == pytest.approx(sk_gamma(y, pred, sample_weight=w), rel=1e-9)


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
def test_tweedie_against_sklearn(size, seed, power):
    rng = np.random.default_rng(seed)
    # Strictly positive y and pred keep all powers in-domain.
    y = rng.uniform(0.1, 10.0, size=size)
    pred = rng.uniform(0.1, 10.0, size=size)
    df = pl.DataFrame({"y": y, "pred": pred})
    result = df.select(mean_tweedie_deviance("y", "pred", power=power)).to_series()[0]
    assert result == pytest.approx(sk_tweedie(y, pred, power=power), rel=1e-9)


@given(
    size=st.integers(min_value=2, max_value=60),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.differing_executors],
)
def test_poisson_against_sklearn(size, seed):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.0, 10.0, size=size)  # y == 0 allowed for Poisson
    pred = rng.uniform(0.1, 10.0, size=size)
    df = pl.DataFrame({"y": y, "pred": pred})
    result = df.select(mean_poisson_deviance("y", "pred")).to_series()[0]
    assert result == pytest.approx(sk_poisson(y, pred), rel=1e-9)


@given(
    size=st.integers(min_value=2, max_value=60),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.differing_executors],
)
def test_gamma_against_sklearn(size, seed):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.1, 10.0, size=size)
    pred = rng.uniform(0.1, 10.0, size=size)
    df = pl.DataFrame({"y": y, "pred": pred})
    result = df.select(mean_gamma_deviance("y", "pred")).to_series()[0]
    assert result == pytest.approx(sk_gamma(y, pred), rel=1e-9)
