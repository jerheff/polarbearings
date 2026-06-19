"""Tests for the bootstrap confidence-interval helpers.

The Bayesian bootstrap reweights rows with Exp(1) draws; because that's exactly
``sample_weight`` to scikit-learn, the oracle for the end-to-end CI is an explicit
numpy/sklearn Exp-weight bootstrap. The interval-method math is checked against
numpy/scipy on the *same* distribution (so it's the method, not Monte-Carlo noise).
"""

import functools
import sys

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from polarbear import (
    bootstrap,
    bootstrap_ci,
    ci_from_distribution,
    f1_score,
    fbeta_score,
    mae,
    max_error,
    median_absolute_error,
    roc_auc,
)
from polarbear.bootstrap import _reduce_ci

# The submodule is shadowed by the same-named function in the package namespace, so
# reach the real module via sys.modules to monkeypatch the version feature flag.
_BMOD = sys.modules["polarbear.bootstrap"]


def _binary_df(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    scores = rng.random(n) * 0.4 + labels * 0.2
    return pl.DataFrame({"y": labels, "p": scores}), labels, scores


def _reg_df(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"y": rng.normal(size=n), "p": rng.normal(size=n)})


def _unnest1(df, expr, name="x"):
    return df.select(expr.alias(name)).unnest(name).row(0, named=True)


def _numpy_ci(dist, theta, level, method):
    alpha = (1 - level) / 2
    if method == "percentile":
        return np.percentile(dist, [alpha * 100, (1 - alpha) * 100])
    if method == "basic":
        qlo, qhi = np.percentile(dist, [alpha * 100, (1 - alpha) * 100])
        return [2 * theta - qhi, 2 * theta - qlo]
    if method == "normal":
        z = norm.ppf(1 - alpha)
        return [theta - z * dist.std(ddof=1), theta + z * dist.std(ddof=1)]
    z0 = norm.ppf((dist < theta).mean())
    za = norm.ppf(alpha)
    return np.percentile(dist, [norm.cdf(2 * z0 + za) * 100, norm.cdf(2 * z0 - za) * 100])


class TestDistribution:
    def test_shape_and_dtype(self):
        df, *_ = _binary_df()
        s = df.select(bootstrap(roc_auc, "y", "p", n_resamples=150)).to_series()
        assert s.dtype == pl.List(pl.Float64)
        assert len(s[0]) == 150

    def test_reproducible_with_seed(self):
        df, *_ = _binary_df()
        a = df.select(bootstrap(f1_score, "y", "p", n_resamples=80, seed=7)).to_series()[0]
        b = df.select(bootstrap(f1_score, "y", "p", n_resamples=80, seed=7)).to_series()[0]
        assert list(a) == list(b)

    def test_different_seed_differs(self):
        df, *_ = _binary_df()
        a = df.select(bootstrap(f1_score, "y", "p", n_resamples=80, seed=1)).to_series()[0]
        b = df.select(bootstrap(f1_score, "y", "p", n_resamples=80, seed=2)).to_series()[0]
        assert list(a) != list(b)

    def test_non_weightable_metrics_raise(self):
        df, *_ = _binary_df()
        for metric in (max_error, median_absolute_error):
            with pytest.raises(ValueError, match="weightable"):
                df.select(bootstrap(metric, "y", "p"))


class TestReductionMath:
    """The Python reduction must match numpy/scipy for every method."""

    def test_reduce_ci_matches_reference(self):
        rng = np.random.default_rng(1)
        dist = rng.lognormal(0.0, 0.5, 500)  # skewed -> exercises bc
        theta = float(np.median(dist)) + 0.05
        for method in ("percentile", "basic", "normal", "bc"):
            lo, hi = _reduce_ci(dist.tolist(), theta, 0.95, method)
            rl, rh = _numpy_ci(dist, theta, 0.95, method)
            assert lo == pytest.approx(rl, abs=1e-6), method
            assert hi == pytest.approx(rh, abs=1e-6), method

    def test_bc_shifts_off_percentile_when_biased(self):
        rng = np.random.default_rng(2)
        dist = (rng.lognormal(0, 1, 3000)).tolist()
        theta = float(np.percentile(dist, 60))  # 60% of mass below -> nonzero z0
        pct = _reduce_ci(dist, theta, 0.95, "percentile")
        bc = _reduce_ci(dist, theta, 0.95, "bc")
        assert pct != pytest.approx(bc)


class TestCiFromDistributionExpr:
    """The expression path (percentile/basic/normal) must agree with the Python reduction."""

    def test_expression_matches_python(self):
        rng = np.random.default_rng(4)
        dist = rng.normal(size=400)
        theta = 0.3
        m = pl.DataFrame({"d": [dist.tolist()], "e": [theta]})
        for method in ("percentile", "basic", "normal"):
            got = _unnest1(m, ci_from_distribution("d", method=method, estimate="e"), "ci")
            lo, hi = _reduce_ci(dist.tolist(), theta, 0.95, method)
            assert got["low"] == pytest.approx(lo, abs=1e-6), method
            assert got["high"] == pytest.approx(hi, abs=1e-6), method

    def test_percentile_levels(self):
        rng = np.random.default_rng(3)
        arr = rng.normal(size=51).tolist()
        df = pl.DataFrame({"d": [arr]})
        got = _unnest1(df, ci_from_distribution("d", level=0.90), "ci")
        lo, hi = np.percentile(arr, [5, 95])
        assert got["low"] == pytest.approx(lo)
        assert got["high"] == pytest.approx(hi)

    def test_non_percentile_requires_estimate(self):
        df = pl.DataFrame({"d": [[1.0, 2.0, 3.0]]})
        for method in ("basic", "normal"):
            with pytest.raises(ValueError, match="estimate"):
                df.select(ci_from_distribution("d", method=method))

    def test_bc_not_supported_per_group(self):
        df = pl.DataFrame({"d": [[1.0, 2.0, 3.0]], "e": [2.0]})
        with pytest.raises(NotImplementedError, match="bootstrap_ci"):
            df.select(ci_from_distribution("d", method="bc", estimate="e"))

    def test_invalid_method_raises(self):
        df = pl.DataFrame({"d": [[1.0, 2.0, 3.0]]})
        with pytest.raises(ValueError, match="method must be one of"):
            df.select(ci_from_distribution("d", method="bogus"))


class TestBootstrapCi:
    def test_returns_dict_with_three_keys(self):
        df, *_ = _binary_df()
        ci = bootstrap_ci(df, roc_auc, "y", "p", n_resamples=300)
        assert set(ci) == {"estimate", "low", "high"}
        assert ci["low"] <= ci["estimate"] <= ci["high"]

    def test_estimate_is_point_metric(self):
        df, *_ = _binary_df()
        ci = bootstrap_ci(df, roc_auc, "y", "p", n_resamples=100)
        point = df.select(roc_auc("y", "p")).to_series()[0]
        assert ci["estimate"] == pytest.approx(point)

    def test_accepts_lazyframe(self):
        df, *_ = _binary_df()
        ci = bootstrap_ci(df.lazy(), roc_auc, "y", "p", n_resamples=100)
        assert ci["low"] <= ci["estimate"] <= ci["high"]

    def test_all_methods_run_and_bracket(self):
        df, *_ = _binary_df()
        for method in ("percentile", "basic", "normal", "bc"):
            ci = bootstrap_ci(df, roc_auc, "y", "p", n_resamples=300, method=method, seed=0)
            assert ci["low"] <= ci["high"]

    def test_higher_level_is_wider(self):
        df, *_ = _binary_df()
        c90 = bootstrap_ci(df, roc_auc, "y", "p", n_resamples=400, level=0.90, seed=5)
        c99 = bootstrap_ci(df, roc_auc, "y", "p", n_resamples=400, level=0.99, seed=5)
        assert c99["low"] <= c90["low"]
        assert c99["high"] >= c90["high"]

    def test_invalid_method_raises(self):
        df, *_ = _binary_df()
        with pytest.raises(ValueError, match="method must be one of"):
            bootstrap_ci(df, roc_auc, "y", "p", method="nope")

    def test_agrees_with_numpy_bayesian_bootstrap(self):
        df, labels, scores = _binary_df(n=4000, seed=0)
        b = 400
        ci = bootstrap_ci(df, roc_auc, "y", "p", n_resamples=b, seed=0)
        rng = np.random.default_rng(123)
        ref = np.array(
            [
                roc_auc_score(labels, scores, sample_weight=rng.exponential(1.0, len(labels)))
                for _ in range(b)
            ]
        )
        lo, hi = np.percentile(ref, [2.5, 97.5])
        assert ci["low"] == pytest.approx(lo, abs=5e-3)
        assert ci["high"] == pytest.approx(hi, abs=5e-3)

    def test_metric_kwargs_forwarded(self):
        df, *_ = _binary_df()
        ci = bootstrap_ci(df, f1_score, "y", "p", threshold=0.3, n_resamples=120)
        point = df.select(f1_score("y", "p", threshold=0.3)).to_series()[0]
        assert ci["low"] <= point <= ci["high"]

        fbeta2 = functools.partial(fbeta_score, beta=2.0)
        ci2 = bootstrap_ci(df, fbeta2, "y", "p", n_resamples=120)
        point2 = df.select(fbeta_score("y", "p", beta=2.0)).to_series()[0]
        assert ci2["low"] <= point2 <= ci2["high"]


class TestWeights:
    def test_weight_column_changes_result(self):
        df = _reg_df()
        rng = np.random.default_rng(9)
        dfw = df.with_columns(w=pl.Series(rng.uniform(0.2, 3.0, df.height)))
        unweighted = bootstrap_ci(dfw, mae, "y", "p", n_resamples=200, seed=1)
        weighted = bootstrap_ci(dfw, mae, "y", "p", weight="w", n_resamples=200, seed=1)
        assert weighted["estimate"] != pytest.approx(unweighted["estimate"])

    def test_weight_expression_equals_named_column(self):
        df = _reg_df()
        rng = np.random.default_rng(9)
        dfw = df.with_columns(w=pl.Series(rng.uniform(0.2, 3.0, df.height)))
        by_name = bootstrap_ci(dfw, mae, "y", "p", weight="w", n_resamples=150, seed=2)
        by_expr = bootstrap_ci(dfw, mae, "y", "p", weight=pl.col("w"), n_resamples=150, seed=2)
        assert by_name == pytest.approx(by_expr)


class TestPerGroupTwoStep:
    def test_group_by_distribution_then_ci(self):
        # Manual composable pattern. Inside group_by, bootstrap needs a materialized
        # row_index (int_range is unstable in agg on some Polars versions).
        df = _binary_df(n=6000, seed=1)[0]
        df = df.with_columns(g=pl.Series(np.arange(df.height) % 3)).with_row_index("ridx")
        out = (
            df.group_by("g")
            .agg(
                roc_auc("y", "p").alias("est"),
                bootstrap(roc_auc, "y", "p", n_resamples=300, seed=4, row_index="ridx").alias(
                    "dist"
                ),
            )
            .with_columns(
                ci_from_distribution("dist", level=0.95, method="basic", estimate="est").alias("ci")
            )
            .drop("dist")
            .unnest("ci")
            .sort("g")
        )
        assert out.height == 3
        for row in out.iter_rows(named=True):
            assert row["low"] <= row["est"] <= row["high"]


class TestPerGroupHelper:
    """bootstrap_ci(..., by=) — the version-transparent per-group helper."""

    @staticmethod
    def _segmented(n=6000):
        df = _binary_df(n=n, seed=1)[0]
        return df.with_columns(seg=pl.Series(np.arange(df.height) % 3))

    def test_returns_dataframe_per_group(self):
        out = bootstrap_ci(
            self._segmented(), roc_auc, "y", "p", by="seg", n_resamples=200, method="basic", seed=0
        )
        assert isinstance(out, pl.DataFrame)
        assert set(out.columns) == {"seg", "estimate", "low", "high"}
        assert out.height == 3
        for row in out.iter_rows(named=True):
            assert row["low"] <= row["estimate"] <= row["high"]

    def test_per_group_bc_not_supported(self):
        with pytest.raises(NotImplementedError, match="bc"):
            bootstrap_ci(self._segmented(), roc_auc, "y", "p", by="seg", method="bc")

    def test_fused_equals_materialized(self, monkeypatch):
        # Within one Polars version the default branch (fused on >=1.28) must equal
        # the forced-materialized branch exactly (identical hash + reduction).
        df = self._segmented()
        default = bootstrap_ci(df, roc_auc, "y", "p", by="seg", n_resamples=200, seed=0)
        monkeypatch.setattr(_BMOD, "_supports_list_agg", lambda: False)
        materialized = bootstrap_ci(df, roc_auc, "y", "p", by="seg", n_resamples=200, seed=0)
        assert_frame_equal(default, materialized)

    @pytest.mark.skipif(
        not _BMOD._supports_list_agg(),
        reason="fused reduce-in-agg requires Polars >= 1.28.0 (pola-rs/polars#22249)",
    )
    def test_fused_branch_executes(self):
        # Runs only on Polars >= 1.28 (skipped on the floor) — explicit fused coverage.
        out = bootstrap_ci(self._segmented(), roc_auc, "y", "p", by="seg", n_resamples=200, seed=0)
        assert out.height == 3
        for row in out.iter_rows(named=True):
            assert row["low"] <= row["estimate"] <= row["high"]


class TestFeatureFlag:
    def test_returns_bool(self):
        assert isinstance(_BMOD._supports_list_agg(), bool)
