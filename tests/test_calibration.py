"""Tests for the calibration_curve helper."""

from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sklearn.calibration import calibration_curve as sk_cc

from polarbearings import expected_calibration_error, maximum_calibration_error
from polarbearings.calibration import calibration_curve as _calibration_curve


def calibration_curve(*args, **kwargs):
    """Collect the lazy result so the eager assertions below stay terse.

    The real function returns a ``LazyFrame``; ``TestLazy`` exercises that directly.
    """
    return _calibration_curve(*args, **kwargs).collect()


def _make(seed, n=500):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(rng.beta(2.0, 2.0, n), 0.0, 1.0)
    return y, p, pl.DataFrame({"y": y.astype(int), "p": p})


class TestStrategies:
    @pytest.mark.parametrize("strategy", ["uniform", "quantile"])
    @pytest.mark.parametrize("n_bins", [3, 5, 10])
    def test_matches_sklearn(self, strategy, n_bins):
        y, p, df = _make(10)
        pt, pp = sk_cc(y, p, n_bins=n_bins, strategy=strategy)
        out = calibration_curve(df, "y", "p", n_bins=n_bins, strategy=strategy)
        assert out["prob_pred"].to_numpy() == pytest.approx(pp)
        assert out["prob_true"].to_numpy() == pytest.approx(pt)

    def test_schema(self):
        _, _, df = _make(0)
        out = calibration_curve(df, "y", "p", n_bins=4)
        assert out.columns == ["bin", "bin_lower", "bin_upper", "count", "prob_pred", "prob_true"]
        assert out["count"].dtype == pl.UInt32 or out["count"].dtype == pl.Int64

    def test_empty_bins_omitted(self):
        # All predictions fall in [0.0, 0.2): only the first uniform bin is populated.
        df = pl.DataFrame({"y": [0, 1, 0, 1], "p": [0.01, 0.05, 0.1, 0.15]})
        out = calibration_curve(df, "y", "p", n_bins=5)
        assert out.height == 1
        assert out["bin"][0] == 0


class TestCustomEdges:
    def test_custom_edges(self):
        df = pl.DataFrame({"y": [0, 0, 1, 1], "p": [0.1, 0.3, 0.6, 0.9]})
        out = calibration_curve(df, "y", "p", bins=[0.0, 0.5, 1.0])
        assert out.height == 2
        # Bin 0: p in [0,0.5) -> {0.1,0.3}, both y=0; Bin 1: {0.6,0.9}, both y=1.
        assert out["prob_true"].to_list() == pytest.approx([0.0, 1.0])
        assert out["prob_pred"].to_list() == pytest.approx([0.2, 0.75])

    def test_unsorted_edges_are_sorted(self):
        df = pl.DataFrame({"y": [0, 1], "p": [0.2, 0.8]})
        out = calibration_curve(df, "y", "p", bins=[1.0, 0.0, 0.5])
        assert out["bin_lower"].to_list() == [0.0, 0.5]


class TestWeighted:
    def test_weighted_means(self):
        df = pl.DataFrame(
            {
                "y": [0, 1, 1, 1],
                "p": [0.1, 0.2, 0.6, 0.7],
                "w": [1.0, 3.0, 1.0, 1.0],
            }
        )
        out = calibration_curve(df, "y", "p", bins=[0.0, 0.5, 1.0], weight="w").sort("bin")
        # Bin 0: p {0.1,0.2} w {1,3} -> pred=(0.1+0.6)/4=0.175, true=(0+3)/4=0.75, count=4
        assert out["prob_pred"][0] == pytest.approx(0.175)
        assert out["prob_true"][0] == pytest.approx(0.75)
        assert out["count"][0] == pytest.approx(4.0)


class TestPosLabel:
    def test_pos_label(self):
        # Labels in {1,2}; treat 2 as positive.
        df = pl.DataFrame({"y": [1, 1, 2, 2], "p": [0.1, 0.3, 0.6, 0.9]})
        out = calibration_curve(df, "y", "p", bins=[0.0, 0.5, 1.0], pos_label=2)
        assert out["prob_true"].to_list() == pytest.approx([0.0, 1.0])


class TestValidation:
    def test_bad_strategy(self):
        _, _, df = _make(0)
        with pytest.raises(ValueError, match="Unknown strategy"):
            calibration_curve(df, "y", "p", strategy=cast("Any", "nope"))

    def test_n_bins_too_small(self):
        _, _, df = _make(0)
        with pytest.raises(ValueError, match="n_bins"):
            calibration_curve(df, "y", "p", n_bins=0)

    def test_bins_too_short(self):
        _, _, df = _make(0)
        with pytest.raises(ValueError, match="at least two edges"):
            calibration_curve(df, "y", "p", bins=[0.5])

    def test_quantile_empty_raises(self):
        df = pl.DataFrame(
            {"y": pl.Series([], dtype=pl.Int64), "p": pl.Series([], dtype=pl.Float64)}
        )
        with pytest.raises(ValueError, match="empty"):
            calibration_curve(df, "y", "p", strategy="quantile")


class TestMissing:
    def test_missing_rows_dropped(self):
        clean = pl.DataFrame({"y": [0, 0, 1, 0, 1, 1], "p": [0.1, 0.2, 0.3, 0.4, 0.6, 0.9]})
        dirty = pl.DataFrame(
            {
                "y": [0, 0, 1, 0, 1, 1, 1, 0, None],
                "p": [0.1, 0.2, 0.3, 0.4, 0.6, 0.9, None, float("nan"), 0.5],
            }
        )  # null-p, nan-p, null-target rows dropped -> same as clean
        a = calibration_curve(clean, "y", "p", n_bins=3)
        b = calibration_curve(dirty, "y", "p", n_bins=3)
        assert a.equals(b)


class TestExprColumns:
    def test_expression_columns_match_string(self):
        _, _, df = _make(0)
        a = calibration_curve(df, "y", "p", n_bins=5)
        b = calibration_curve(df, pl.col("y"), pl.col("p"), n_bins=5)
        assert a.equals(b)


class TestLazy:
    def test_returns_lazyframe(self):
        _, _, df = _make(0)
        assert isinstance(_calibration_curve(df, "y", "p"), pl.LazyFrame)

    def test_accepts_lazyframe_and_matches_eager(self):
        _, _, df = _make(0)
        lazy = _calibration_curve(df.lazy(), "y", "p", n_bins=5, strategy="quantile")
        assert isinstance(lazy, pl.LazyFrame)
        eager = _calibration_curve(df, "y", "p", n_bins=5, strategy="quantile")
        assert lazy.collect().equals(eager.collect())


class TestByGroup:
    def _grouped(self, seed=3, n=900):
        rng = np.random.default_rng(seed)
        g = rng.choice(["A", "B"], n)
        y = rng.integers(0, 2, n)
        p = np.clip(rng.beta(2.0, 2.0, n), 0.0, 1.0)
        return pl.DataFrame({"g": g, "y": y.astype(int), "p": p})

    def test_by_columns_and_shape(self):
        out = calibration_curve(self._grouped(), "y", "p", n_bins=5, by="g")
        assert out.columns == [
            "g",
            "bin",
            "bin_lower",
            "bin_upper",
            "count",
            "prob_pred",
            "prob_true",
        ]
        assert set(out["g"].unique()) == {"A", "B"}

    def test_by_matches_per_group_filter(self):
        # A grouped curve equals the curve on each filtered subgroup (the bin edges
        # are shared/global, so the bins line up).
        df = self._grouped()
        grouped = calibration_curve(df, "y", "p", n_bins=8, by="g").sort("g", "bin")
        for g in ("A", "B"):
            sub = calibration_curve(df.filter(pl.col("g") == g), "y", "p", n_bins=8)
            got = grouped.filter(pl.col("g") == g)
            assert got["bin"].to_list() == sub["bin"].to_list()
            assert got["prob_pred"].to_numpy() == pytest.approx(sub["prob_pred"].to_numpy())
            assert got["prob_true"].to_numpy() == pytest.approx(sub["prob_true"].to_numpy())

    def test_counts_partition_the_frame(self):
        # Every row lands in exactly one (group, bin), so counts sum to n.
        df = self._grouped()
        out = calibration_curve(df, "y", "p", n_bins=6, strategy="quantile", by="g")
        assert out["count"].sum() == df.height


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    n_bins=st.integers(min_value=2, max_value=12),
    strategy=st.sampled_from(["uniform", "quantile"]),
)
@settings(deadline=None, database=None, suppress_health_check=[HealthCheck.differing_executors])
def test_against_sklearn(seed, n_bins, strategy):
    y, p, df = _make(seed, n=300)
    pt, pp = sk_cc(y, p, n_bins=n_bins, strategy=strategy)
    out = calibration_curve(df, "y", "p", n_bins=n_bins, strategy=strategy)
    assert out["prob_pred"].to_numpy() == pytest.approx(pp)
    assert out["prob_true"].to_numpy() == pytest.approx(pt)


def _ece_mce_np(y, p, n_bins, w=None, strategy="uniform"):
    edges = (
        np.linspace(0, 1, n_bins + 1)
        if strategy == "uniform"
        else np.quantile(p, np.linspace(0, 1, n_bins + 1))
    )
    b = np.searchsorted(edges[1:-1], p, side="left")
    w = np.ones_like(p, float) if w is None else w
    tot = w.sum()
    ece = mce = 0.0
    for bb in np.unique(b):
        m = b == bb
        wb = w[m].sum()
        gap = abs((p[m] * w[m]).sum() / wb - (y[m] * w[m]).sum() / wb)
        ece += wb / tot * gap
        mce = max(mce, gap)
    return ece, mce


class TestCalibrationError:
    @pytest.mark.parametrize("strategy", ["uniform", "quantile"])
    @pytest.mark.parametrize("n_bins", [5, 10, 15])
    def test_matches_numpy(self, strategy, n_bins):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 3000)
        p = np.clip(rng.beta(2, 2, 3000), 0, 1)
        df = pl.DataFrame({"y": y, "p": p})
        ece = df.select(
            expected_calibration_error("y", "p", n_bins=n_bins, strategy=strategy)
        ).item()
        mce = df.select(
            maximum_calibration_error("y", "p", n_bins=n_bins, strategy=strategy)
        ).item()
        te, tm = _ece_mce_np(y, p, n_bins, strategy=strategy)
        assert ece == pytest.approx(te)
        assert mce == pytest.approx(tm)

    def test_weighted_matches_numpy(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 3000)
        p = np.clip(rng.beta(2, 2, 3000), 0, 1)
        w = rng.uniform(0.5, 2.0, 3000)
        df = pl.DataFrame({"y": y, "p": p, "w": w})
        ece = df.select(expected_calibration_error("y", "p", weight="w")).item()
        te, _ = _ece_mce_np(y, p, 10, w=w)
        assert ece == pytest.approx(te)

    def test_consistent_with_calibration_curve(self):
        rng = np.random.default_rng(2)
        df = pl.DataFrame({"y": rng.integers(0, 2, 2000), "p": np.clip(rng.beta(2, 2, 2000), 0, 1)})
        cal = calibration_curve(df, "y", "p", n_bins=10)
        gap = (cal["prob_pred"] - cal["prob_true"]).abs()
        ece_cc = (cal["count"] / cal["count"].sum() * gap).sum()
        assert df.select(expected_calibration_error("y", "p")).item() == pytest.approx(ece_cc)
        assert df.select(maximum_calibration_error("y", "p")).item() == pytest.approx(gap.max())

    def test_perfect_calibration_is_zero(self):
        df = pl.DataFrame({"y": [0, 0, 1, 1] * 50, "p": [0.0, 0.0, 1.0, 1.0] * 50})
        assert df.select(expected_calibration_error("y", "p")).item() == pytest.approx(0.0)
        assert df.select(maximum_calibration_error("y", "p")).item() == pytest.approx(0.0)

    def test_group_by(self):
        rng = np.random.default_rng(3)
        seg = rng.choice(["a", "b"], 4000)
        y = rng.integers(0, 2, 4000)
        p = np.clip(rng.beta(2, 2, 4000), 0, 1)
        df = pl.DataFrame({"seg": seg, "y": y, "p": p})
        out = (
            df.group_by("seg", maintain_order=True)
            .agg(
                expected_calibration_error("y", "p").alias("ece"),
                maximum_calibration_error("y", "p").alias("mce"),
            )
            .sort("seg")
        )
        for row in out.iter_rows(named=True):
            m = seg == row["seg"]
            te, tm = _ece_mce_np(y[m], p[m], 10)
            assert row["ece"] == pytest.approx(te)
            assert row["mce"] == pytest.approx(tm)

    def test_missing_is_null(self):
        df = pl.DataFrame({"y": [0, 1, 1, None], "p": [0.1, 0.5, 0.9, 0.3]})
        assert df.select(expected_calibration_error("y", "p")).item() is None
        assert df.select(maximum_calibration_error("y", "p")).item() is None

    def test_pos_label(self):
        a = pl.DataFrame({"y": [1, 1, 2, 2], "p": [0.1, 0.3, 0.6, 0.9]})
        b = pl.DataFrame({"y": [0, 0, 1, 1], "p": [0.1, 0.3, 0.6, 0.9]})
        assert a.select(expected_calibration_error("y", "p", pos_label=2)).item() == pytest.approx(
            b.select(expected_calibration_error("y", "p")).item()
        )

    def test_custom_bins_match_calibration_curve(self):
        rng = np.random.default_rng(5)
        df = pl.DataFrame({"y": rng.integers(0, 2, 1500), "p": np.clip(rng.beta(2, 2, 1500), 0, 1)})
        edges = [0.0, 0.25, 0.5, 0.75, 1.0]
        cal = calibration_curve(df, "y", "p", bins=edges)
        gap = (cal["prob_pred"] - cal["prob_true"]).abs()
        ece_cc = (cal["count"] / cal["count"].sum() * gap).sum()
        assert df.select(expected_calibration_error("y", "p", bins=edges)).item() == pytest.approx(
            ece_cc
        )
        assert df.select(maximum_calibration_error("y", "p", bins=edges)).item() == pytest.approx(
            gap.max()
        )

    def test_validation(self):
        df = pl.DataFrame({"y": [0, 1], "p": [0.2, 0.8]})
        with pytest.raises(ValueError, match="Unknown strategy"):
            df.select(expected_calibration_error("y", "p", strategy=cast("Any", "nope")))
        with pytest.raises(ValueError, match="n_bins"):
            df.select(expected_calibration_error("y", "p", n_bins=0))
        with pytest.raises(ValueError, match="at least two edges"):
            df.select(maximum_calibration_error("y", "p", bins=[0.5]))


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    n_bins=st.integers(min_value=2, max_value=20),
    strategy=st.sampled_from(["uniform", "quantile"]),
)
@settings(deadline=None, database=None, suppress_health_check=[HealthCheck.differing_executors])
def test_calibration_error_matches_curve_property(seed, n_bins, strategy):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 1200))
    df = pl.DataFrame({"y": rng.integers(0, 2, n), "p": np.clip(rng.random(n), 0, 1)})
    cal = calibration_curve(df, "y", "p", n_bins=n_bins, strategy=strategy)
    gap = (cal["prob_pred"] - cal["prob_true"]).abs()
    ece_cc = (cal["count"] / cal["count"].sum() * gap).sum()
    ece = df.select(expected_calibration_error("y", "p", n_bins=n_bins, strategy=strategy)).item()
    mce = df.select(maximum_calibration_error("y", "p", n_bins=n_bins, strategy=strategy)).item()
    assert ece == pytest.approx(ece_cc)
    assert mce == pytest.approx(gap.max())
