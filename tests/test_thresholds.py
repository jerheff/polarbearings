"""Tests for threshold specs and spec-driven threshold_sweep."""

import numpy as np
import polars as pl
import pytest

from polarbearings import (
    confusion_matrix,
    equal_width,
    f1_score,
    linspace,
    precision,
    quantiles,
    resolve_thresholds,
    threshold_sweep,
)


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 4000
    return pl.DataFrame(
        {
            "g": rng.integers(0, 3, n),
            "y": rng.integers(0, 2, n),
            "p": rng.random(n),
        }
    )


class TestSpecs:
    def test_quantiles_count_and_labels(self):
        resolved = resolve_thresholds(quantiles(10), "p")
        assert len(resolved) == 10
        labels = [label for label, _ in resolved]
        assert labels[0] == "q0.0909091"  # 1/11
        assert all(isinstance(v, pl.Expr) for _, v in resolved)

    def test_quantiles_values_match_concrete(self, df):
        resolved = resolve_thresholds(quantiles(4), "p")
        for i, (_, expr) in enumerate(resolved, start=1):
            assert isinstance(expr, pl.Expr)
            got = df.select(expr.alias("t")).to_series()[0]
            exp = df["p"].quantile(i / 5, interpolation="linear")
            assert got == pytest.approx(exp)

    def test_equal_width_values_match_concrete(self, df):
        lo, hi = df["p"].min(), df["p"].max()
        resolved = resolve_thresholds(equal_width(4), "p")
        for i, (_, expr) in enumerate(resolved, start=1):
            assert isinstance(expr, pl.Expr)
            got = df.select(expr.alias("t")).to_series()[0]
            assert got == pytest.approx(lo + (i / 5) * (hi - lo))

    def test_linspace_is_data_free_floats(self):
        resolved = resolve_thresholds(linspace(4), "ignored_column")
        assert [v for _, v in resolved] == pytest.approx([0.2, 0.4, 0.6, 0.8])

    def test_linspace_custom_bounds(self):
        resolved = resolve_thresholds(linspace(3, lo=1.0, hi=5.0), "x")
        assert [v for _, v in resolved] == pytest.approx([2.0, 3.0, 4.0])

    def test_list_passthrough_labels(self):
        resolved = resolve_thresholds([0.3, 0.5, 0.7], "p")
        assert resolved == [("0.3", 0.3), ("0.5", 0.5), ("0.7", 0.7)]

    @pytest.mark.parametrize("factory", [quantiles, equal_width, linspace])
    def test_zero_raises(self, factory):
        with pytest.raises(ValueError, match=">= 1"):
            factory(0)


class TestThresholdSweep:
    def test_fixed_list_names_unchanged(self, df):
        swept = df.select(*threshold_sweep(f1_score, "y", "p", [0.3, 0.5, 0.7]))
        assert swept.columns == ["f1_score_y_p_0.3", "f1_score_y_p_0.5", "f1_score_y_p_0.7"]
        # And the values equal the single-call metric.
        native = df.select(f1_score("y", "p", threshold=0.5)).to_series()[0]
        assert swept["f1_score_y_p_0.5"][0] == native

    def test_default_is_quantiles_100(self, df):
        swept = df.select(*threshold_sweep(precision, "y", "p"))
        assert len(swept.columns) == 100

    def test_spec_values_embedded(self, df):
        swept = df.select(*threshold_sweep(precision, "y", "p", quantiles(5)))
        assert len(swept.columns) == 5
        # First column = precision at the 1/6 quantile of p.
        thr = float(df["p"].quantile(1 / 6, interpolation="linear"))
        exp = df.select(precision("y", "p", threshold=thr)).to_series()[0]
        assert swept[swept.columns[0]][0] == pytest.approx(exp)

    def test_per_group_quantiles_in_one_agg(self, df):
        out = df.group_by("g").agg(*threshold_sweep(precision, "y", "p", quantiles(5))).sort("g")
        col = next(c for c in out.columns if c != "g")
        for g in range(3):
            sub = df.filter(pl.col("g") == g)
            exp = sub.select(
                precision(
                    "y", "p", threshold=float(sub["p"].quantile(1 / 6, interpolation="linear"))
                )
            ).to_series()[0]
            got = out.filter(pl.col("g") == g)[col][0]
            assert got == pytest.approx(exp)

    def test_confusion_matrix_sweep(self, df):
        swept = df.select(*threshold_sweep(confusion_matrix, "y", "p", equal_width(4)))
        assert swept.columns == [
            "confusion_matrix_y_p_ew0.2",
            "confusion_matrix_y_p_ew0.4",
            "confusion_matrix_y_p_ew0.6",
            "confusion_matrix_y_p_ew0.8",
        ]

    def test_weight_and_pos_label_in_alias(self, df):
        d = df.with_columns(w=pl.lit(1.0))
        swept = df.with_columns(d["w"]).select(
            *threshold_sweep(precision, "y", "p", quantiles(2), weight="w", pos_label=0)
        )
        for c in swept.columns:
            assert c.endswith("_w_pos0")
            assert c.startswith("precision_y_p_q")

    def test_custom_metric_fn_fallback_alias(self, df):
        # A metric whose output name does NOT follow the _alias shape exercises the
        # non-endswith fallback (prefix = the whole probe name).
        def weird(target, prob, threshold=0.5, weight=None, pos_label=1):
            return precision(target, prob, threshold=threshold).alias("weird")

        swept = df.select(*threshold_sweep(weird, "y", "p", linspace(2)))
        assert swept.columns == ["weird_0.333333", "weird_0.666667"]

    def test_direct_expr_threshold(self, df):
        # A metric accepts an expression threshold directly.
        got = df.select(
            precision("y", "p", threshold=pl.col("p").quantile(0.9)).alias("pq")
        ).to_series()[0]
        exp = df.select(precision("y", "p", threshold=float(df["p"].quantile(0.9)))).to_series()[0]
        assert got == pytest.approx(exp)

    def test_expression_columns_match_string(self, df):
        # target/prob may be expressions; col_name keeps the swept column names.
        a = df.select(*threshold_sweep(precision, "y", "p", quantiles(5)))
        b = df.select(*threshold_sweep(precision, pl.col("y"), pl.col("p"), quantiles(5)))
        assert a.equals(b)
