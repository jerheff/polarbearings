"""Tests for confusion_curve (exact confusion cells at every distinct score)."""

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sklearn.metrics import roc_curve

from polarbearings import confusion_matrix
from polarbearings.confusion_curve import _GRID_EXACT_CUTOVER
from polarbearings.confusion_curve import confusion_curve as _confusion_curve


def confusion_curve(*args, **kwargs):
    """Collect the lazy result so the eager assertions below stay terse.

    The real function returns a ``LazyFrame``; ``TestLazy`` exercises that directly.
    """
    return _confusion_curve(*args, **kwargs).collect()


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 2000
    return pl.DataFrame(
        {
            "seg": rng.choice(["A", "B", "C"], n),
            "y": rng.integers(0, 2, n),
            "score": rng.random(n),
            "w": rng.uniform(0.5, 2.0, n),
        }
    )


class TestSchemaAndShape:
    def test_columns_and_dtypes(self, df):
        out = confusion_curve(df, "y", "score", endpoints=False)
        assert out.columns == ["threshold", "tp", "fp", "fn", "tn"]
        assert out.schema["threshold"] == pl.Float64
        assert all(out.schema[c] == pl.Int64 for c in ("tp", "fp", "fn", "tn"))

    def test_one_row_per_distinct_score(self, df):
        out = confusion_curve(df, "y", "score", endpoints=False)
        assert out.height == df["score"].n_unique()

    def test_sorted_descending(self, df):
        out = confusion_curve(df, "y", "score", endpoints=False)
        assert out["threshold"].is_sorted(descending=True)

    def test_cells_sum_to_total(self, df):
        out = confusion_curve(df, "y", "score", endpoints=False)
        total = pl.col("tp") + pl.col("fp") + pl.col("fn") + pl.col("tn")
        assert (out.select(total) == df.height).to_series().all()


class TestCompatibility:
    def test_matches_confusion_matrix_at_each_threshold(self, df):
        cc = confusion_curve(df, "y", "score", endpoints=False)
        for i in (5, 500, 1500):
            t = cc["threshold"][i]
            cm = (
                df.select(confusion_matrix("y", "score", threshold=t).alias("cm"))
                .unnest("cm")
                .row(0, named=True)
            )
            row = cc.filter(pl.col("threshold") == t).row(0, named=True)
            assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (
                cm["tp"],
                cm["fp"],
                cm["fn"],
                cm["tn"],
            )

    def test_matches_sklearn_roc_curve(self, df):
        cc = confusion_curve(df, "y", "score", endpoints=False)
        p = int((df["y"] == 1).sum())
        n = df.height - p
        roc = cc.with_columns(tpr=pl.col("tp") / p, fpr=pl.col("fp") / n)
        ours = dict(zip(roc["threshold"], zip(roc["fpr"], roc["tpr"], strict=True), strict=True))
        fpr, tpr, thr = roc_curve(df["y"].to_numpy(), df["score"].to_numpy())
        for t, f, r in zip(thr, fpr, tpr, strict=True):
            if t in ours:  # sklearn drops collinear points and adds an inf endpoint
                assert ours[t] == pytest.approx((f, r))


class TestEndpoints:
    def test_endpoint_prepended(self, df):
        out = confusion_curve(df, "y", "score", endpoints=True)
        assert out.height == df["score"].n_unique() + 1
        first = out.row(0, named=True)
        assert first["threshold"] == float("inf")
        assert (first["tp"], first["fp"]) == (0, 0)
        # the trivial point: nothing predicted positive -> fn=P, tn=N
        p = int((df["y"] == 1).sum())
        assert (first["fn"], first["tn"]) == (p, df.height - p)

    def test_last_point_is_all_positive(self, df):
        out = confusion_curve(df, "y", "score", endpoints=True)
        last = out.row(-1, named=True)
        assert (last["fn"], last["tn"]) == (0, 0)  # everything predicted positive


class TestWeighted:
    def test_weighted_cells_are_floats_and_match(self, df):
        cc = confusion_curve(df, "y", "score", weight="w", endpoints=False)
        assert cc.schema["tp"] == pl.Float64
        t = cc["threshold"][200]
        cm = (
            df.select(confusion_matrix("y", "score", threshold=t, weight="w").alias("cm"))
            .unnest("cm")
            .row(0, named=True)
        )
        row = cc.filter(pl.col("threshold") == t).row(0, named=True)
        for k in ("tp", "fp", "fn", "tn"):
            assert row[k] == pytest.approx(cm[k])


class TestPosLabel:
    def test_pos_label_matches_relabeled(self, df):
        a = confusion_curve(df, "y", "score", pos_label=0, endpoints=False)
        inv = df.with_columns(y=1 - pl.col("y"))
        b = confusion_curve(inv, "y", "score", pos_label=1, endpoints=False)
        assert a.equals(b)


class TestMissing:
    def test_missing_score_target_weight_rows_dropped(self):
        clean = pl.DataFrame(
            {"y": [0, 0, 1, 1], "score": [0.2, 0.8, 0.6, 0.9], "w": [1.0, 1.0, 2.0, 2.0]}
        )
        dirty = pl.DataFrame(
            {
                "y": [0, 0, 1, 1, 1, None, 0],
                "score": [0.2, 0.8, 0.6, 0.9, None, 0.5, float("nan")],
                "w": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            }
        )  # null-score, null-target, nan-score rows dropped -> same as clean
        a = confusion_curve(clean, "y", "score", weight="w", endpoints=False)
        b = confusion_curve(dirty, "y", "score", weight="w", endpoints=False)
        assert a.equals(b)

    def test_null_weight_row_dropped(self):
        clean = pl.DataFrame({"y": [0, 1, 1], "score": [0.2, 0.6, 0.9], "w": [1.0, 2.0, 2.0]})
        dirty = pl.DataFrame(
            {"y": [0, 1, 1, 0], "score": [0.2, 0.6, 0.9, 0.5], "w": [1.0, 2.0, 2.0, None]}
        )
        a = confusion_curve(clean, "y", "score", weight="w", endpoints=False)
        b = confusion_curve(dirty, "y", "score", weight="w", endpoints=False)
        assert a.equals(b)

    def test_group_by_drops_within_group(self):
        dirty = pl.DataFrame(
            {
                "g": ["A", "A", "A", "B", "B"],
                "y": [0, 1, 1, 0, 1],
                "score": [0.2, None, 0.8, 0.3, 0.7],
            }
        )
        clean = pl.DataFrame(
            {"g": ["A", "A", "B", "B"], "y": [0, 1, 0, 1], "score": [0.2, 0.8, 0.3, 0.7]}
        )
        a = confusion_curve(dirty, "y", "score", by="g", endpoints=False)
        b = confusion_curve(clean, "y", "score", by="g", endpoints=False)
        assert a.equals(b)


class TestExprColumns:
    def test_expression_target_and_score(self, df):
        a = confusion_curve(df, "y", "score", endpoints=False)
        b = confusion_curve(df, pl.col("y"), pl.col("score"), endpoints=False)
        assert a.equals(b)

    def test_derived_score_keeps_cells(self, df):
        # A monotonic transform of score preserves the curve's cells; only the
        # threshold values scale, so drop it and the cells must match.
        base = confusion_curve(df, "y", "score", endpoints=False).drop("threshold")
        scaled = confusion_curve(df, "y", pl.col("score") * 2, endpoints=False).drop("threshold")
        assert base.equals(scaled)

    def test_expression_by(self, df):
        out = confusion_curve(df, "y", "score", by=pl.col("seg"), endpoints=False)
        assert "seg" in out.columns


class TestLazy:
    def test_returns_lazyframe(self, df):
        assert isinstance(_confusion_curve(df, "y", "score"), pl.LazyFrame)

    def test_accepts_lazyframe_and_matches_eager(self, df):
        out = _confusion_curve(df.lazy(), "y", "score", endpoints=False)
        assert isinstance(out, pl.LazyFrame)
        assert out.collect().equals(_confusion_curve(df, "y", "score", endpoints=False).collect())


class TestBy:
    def test_per_group_matches_standalone(self, df):
        grouped = confusion_curve(df, "y", "score", by="seg", endpoints=False)
        assert grouped.columns == ["seg", "threshold", "tp", "fp", "fn", "tn"]
        for seg in ("A", "B", "C"):
            standalone = confusion_curve(
                df.filter(pl.col("seg") == seg), "y", "score", endpoints=False
            )
            got = grouped.filter(pl.col("seg") == seg).drop("seg")
            assert got.equals(standalone)

    def test_by_with_endpoints_one_origin_per_group(self, df):
        grouped = confusion_curve(df, "y", "score", by="seg", endpoints=True)
        infs = grouped.filter(pl.col("threshold") == float("inf"))
        assert infs.height == 3  # one origin per segment
        assert (infs["tp"] == 0).all()
        assert (infs["fp"] == 0).all()

    def test_by_list(self, df):
        out = confusion_curve(df, "y", "score", by=["seg"], endpoints=False)
        assert "seg" in out.columns


class TestGridFastPath:
    """The whole-frame large-N grid sampled off the exact curve (>= cutover)."""

    def test_large_grid_matches_confusion_matrix(self, df):
        thresholds = [round(t, 4) for t in np.linspace(0.02, 0.98, 40)]
        assert len(thresholds) >= _GRID_EXACT_CUTOVER  # exercises _grid_via_exact
        cc = confusion_curve(df, "y", "score", thresholds=thresholds)
        assert cc.height == len(thresholds)
        for t in (thresholds[1], thresholds[19], thresholds[38]):
            cm = (
                df.select(confusion_matrix("y", "score", threshold=t).alias("cm"))
                .unnest("cm")
                .row(0, named=True)
            )
            row = cc.filter(pl.col("threshold") == t).row(0, named=True)
            assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (
                cm["tp"],
                cm["fp"],
                cm["fn"],
                cm["tn"],
            )

    def test_fast_and_direct_paths_agree(self, df, monkeypatch):
        import sys

        from polars.testing import assert_frame_equal

        # The package re-exports the `confusion_curve` function over the submodule
        # name, so reach the module object via sys.modules to patch the cutover.
        cc_mod = sys.modules["polarbearings.confusion_curve"]
        thresholds = [round(t, 4) for t in np.linspace(0.01, 0.99, 35)]
        for weight in (None, "w"):
            fast = _confusion_curve(
                df, "y", "score", thresholds=thresholds, weight=weight
            ).collect()
            # Force the direct per-threshold path on the same thresholds and compare.
            monkeypatch.setattr(cc_mod, "_GRID_EXACT_CUTOVER", 10**9)
            direct = _confusion_curve(
                df, "y", "score", thresholds=thresholds, weight=weight
            ).collect()
            monkeypatch.setattr(cc_mod, "_GRID_EXACT_CUTOVER", _GRID_EXACT_CUTOVER)
            assert_frame_equal(fast.sort("threshold"), direct.sort("threshold"))

    def test_grouped_large_grid_stays_direct_and_correct(self, df):
        # Grouped grids never take the fast path, but must still be correct at large N.
        thresholds = [round(t, 4) for t in np.linspace(0.05, 0.95, 32)]
        out = confusion_curve(df, "y", "score", thresholds=thresholds, by="seg")
        assert set(out["seg"].unique()) == set(df["seg"].unique())
        assert out.height == len(thresholds) * df["seg"].n_unique()


@given(
    n=st.integers(min_value=5, max_value=300),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(deadline=None, database=None, suppress_health_check=[HealthCheck.differing_executors])
def test_roc_against_sklearn(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    if y.sum() == 0 or y.sum() == n:  # need both classes for an ROC
        return
    score = rng.random(n)
    df = pl.DataFrame({"y": y, "score": score})
    cc = confusion_curve(df, "y", "score", endpoints=False)
    p, neg = int(y.sum()), int(n - y.sum())
    roc = cc.with_columns(tpr=pl.col("tp") / p, fpr=pl.col("fp") / neg)
    ours = dict(zip(roc["threshold"], zip(roc["fpr"], roc["tpr"], strict=True), strict=True))
    fpr, tpr, thr = roc_curve(y, score)
    for t, f, r in zip(thr, fpr, tpr, strict=True):
        if t in ours:
            assert ours[t] == pytest.approx((f, r))
