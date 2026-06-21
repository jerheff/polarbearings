"""Tests for the diagnostic curve helpers (roc_curve, pr_curve, det_curve, expected_cost)."""

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sklearn.metrics import (
    average_precision_score,
    det_curve,
    roc_auc_score,
)

from polarbearings import (
    confusion_curve,
    confusion_matrix,
)
from polarbearings import (
    det_curve as pb_det_curve,
)
from polarbearings import (
    expected_cost as pb_expected_cost,
)
from polarbearings import (
    pr_curve as pb_pr_curve,
)
from polarbearings import (
    roc_curve as pb_roc_curve,
)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal area; avoids numpy's renamed ``trapz``/``trapezoid`` across versions."""
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    score = 1 / (1 + np.exp(-rng.normal(y * 1.3, 1.0)))
    return pl.DataFrame(
        {
            "seg": rng.choice(["A", "B"], n),
            "y": y,
            "score": score,
            "w": rng.uniform(0.5, 2.0, n),
        }
    )


class TestSchema:
    def test_roc_columns(self, df):
        assert pb_roc_curve(df, "y", "score").collect().columns == ["threshold", "fpr", "tpr"]

    def test_pr_columns(self, df):
        out = pb_pr_curve(df, "y", "score").collect()
        assert out.columns == ["threshold", "precision", "recall"]

    def test_det_columns(self, df):
        assert pb_det_curve(df, "y", "score").collect().columns == ["threshold", "fpr", "fnr"]

    def test_cost_columns(self, df):
        out = pb_expected_cost(df, "y", "score", {"fn": 5.0}).collect()
        assert out.columns == ["threshold", "cost"]

    def test_by_prepends_group_columns(self, df):
        out = pb_roc_curve(df, "y", "score", by="seg").collect()
        assert out.columns == ["seg", "threshold", "fpr", "tpr"]
        assert set(out["seg"].unique()) == {"A", "B"}


class TestRocCurve:
    def test_auc_matches_sklearn(self, df):
        roc = pb_roc_curve(df, "y", "score").collect()
        auc = _trapezoid(roc["tpr"].to_numpy(), roc["fpr"].to_numpy())
        assert auc == pytest.approx(roc_auc_score(df["y"], df["score"]), abs=1e-9)

    def test_starts_at_origin(self, df):
        roc = pb_roc_curve(df, "y", "score").collect()
        first = roc.row(0, named=True)
        assert (first["fpr"], first["tpr"]) == (0.0, 0.0)

    def test_fpr_monotone_nondecreasing(self, df):
        roc = pb_roc_curve(df, "y", "score").collect()
        assert roc["fpr"].is_sorted()

    def test_rates_null_when_single_class(self):
        df = pl.DataFrame({"y": [1, 1, 1], "score": [0.2, 0.5, 0.9]})
        roc = pb_roc_curve(df, "y", "score").collect()
        # no negatives -> fpr undefined (null); tpr defined
        assert roc["fpr"].null_count() == roc.height
        assert roc["tpr"].null_count() == 0


class TestPrCurve:
    def test_ap_matches_sklearn(self, df):
        # Average precision = sum of (recall_k - recall_{k-1}) * precision_k over the
        # curve in increasing-recall order (sklearn's step-function definition).
        pr = pb_pr_curve(df, "y", "score").collect().sort("recall")
        recall = pr["recall"].to_numpy()
        precision = pr["precision"].to_numpy()
        ap = float(np.sum(np.diff(recall, prepend=0.0) * precision))
        assert ap == pytest.approx(average_precision_score(df["y"], df["score"]), abs=1e-6)

    def test_endpoint_precision_one_recall_zero(self, df):
        pr = pb_pr_curve(df, "y", "score").collect()
        top = pr.filter(pl.col("threshold").is_infinite()).row(0, named=True)
        assert top["precision"] == 1.0
        assert top["recall"] == 0.0


class TestDetCurve:
    def test_matches_sklearn_at_shared_thresholds(self, df):
        det = pb_det_curve(df, "y", "score").collect()
        ours = {t: (f, n) for t, f, n in zip(det["threshold"], det["fpr"], det["fnr"], strict=True)}
        fpr, fnr, thr = det_curve(df["y"].to_numpy(), df["score"].to_numpy())
        matched = 0
        for t, f, n in zip(thr, fpr, fnr, strict=True):
            if t in ours:  # sklearn collapses tied thresholds; we keep all distinct scores
                matched += 1
                assert ours[t] == pytest.approx((f, n))
        assert matched > 0, "no shared thresholds — parity check would pass vacuously"

    def test_no_infinite_endpoint_by_default(self, df):
        det = pb_det_curve(df, "y", "score").collect()
        assert not det["threshold"].is_infinite().any()

    def test_fnr_is_one_minus_tpr(self, df):
        det = pb_det_curve(df, "y", "score").collect()
        roc = pb_roc_curve(df, "y", "score", endpoints=False).collect()
        merged = det.join(roc, on="threshold")
        assert (merged["fnr"] + merged["tpr"]).to_numpy() == pytest.approx(1.0)


class TestExpectedCost:
    def test_matches_manual_cost_at_argmin(self, df):
        costs = {"fn": 5.0, "fp": 1.0}
        ec = pb_expected_cost(df, "y", "score", costs).collect()
        best = ec.sort("cost").row(0, named=True)
        cm = (
            df.select(confusion_matrix("y", "score", threshold=best["threshold"]).alias("cm"))
            .unnest("cm")
            .row(0, named=True)
        )
        manual = costs["fn"] * cm["fn"] + costs["fp"] * cm["fp"]
        assert best["cost"] == pytest.approx(manual)

    def test_normalize_divides_by_count(self, df):
        total = pb_expected_cost(df, "y", "score", {"fn": 5.0}).collect()
        per = pb_expected_cost(df, "y", "score", {"fn": 5.0}, normalize=True).collect()
        merged = total.join(per, on="threshold", suffix="_per")
        # total cost == per-decision cost * N, elementwise (avoids 0/0 at the
        # all-positive endpoint where the cost is zero).
        assert merged["cost"].to_numpy() == pytest.approx(merged["cost_per"].to_numpy() * df.height)

    def test_rejects_unknown_cost_cell(self, df):
        with pytest.raises(ValueError, match="Unknown cost cell"):
            pb_expected_cost(df, "y", "score", {"oops": 1.0})


class TestGridPath:
    def test_confusion_curve_grid_matches_confusion_matrix(self, df):
        grid = [0.25, 0.5, 0.75]
        cc = confusion_curve(df, "y", "score", thresholds=grid).collect()
        assert cc.height == len(grid)
        for t in grid:
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

    def test_grid_roc_agrees_with_exact_at_grid_points(self, df):
        grid = [0.3, 0.6, 0.9]
        roc_grid = pb_roc_curve(df, "y", "score", thresholds=grid).collect()
        # The exact curve evaluated and then filtered to the same operating points
        # via confusion_matrix should give identical rates.
        for t in grid:
            cm = (
                df.select(confusion_matrix("y", "score", threshold=t).alias("cm"))
                .unnest("cm")
                .row(0, named=True)
            )
            row = roc_grid.filter(pl.col("threshold") == t).row(0, named=True)
            assert row["fpr"] == pytest.approx(cm["fp"] / (cm["fp"] + cm["tn"]))
            assert row["tpr"] == pytest.approx(cm["tp"] / (cm["tp"] + cm["fn"]))

    def test_grid_ignores_endpoints(self, df):
        cc = confusion_curve(df, "y", "score", thresholds=[0.4, 0.6], endpoints=True).collect()
        assert not cc["threshold"].is_infinite().any()
        assert cc.height == 2

    def test_int_thresholds_uses_quantiles(self, df):
        # thresholds=N is shorthand for quantiles(N): N data-driven rows whose
        # thresholds are the score quantiles, not a fixed [0, 1] grid.
        from polarbearings import quantiles

        n = 20
        as_int = confusion_curve(df, "y", "score", thresholds=n).collect().sort("threshold")
        as_spec = (
            confusion_curve(df, "y", "score", thresholds=quantiles(n)).collect().sort("threshold")
        )
        assert as_int.height == n
        assert as_int["threshold"].to_list() == pytest.approx(as_spec["threshold"].to_list())
        # quantile thresholds track the data, so they land inside its observed range
        lo, hi = df["score"].min(), df["score"].max()
        assert as_int["threshold"].min() >= lo
        assert as_int["threshold"].max() <= hi

    def test_int_thresholds_per_group(self, df):
        # Each group is thresholded at its own quantiles in one pass.
        cc = confusion_curve(df, "y", "score", thresholds=5, by="seg").collect()
        assert cc.height == 5 * df["seg"].n_unique()
        thr_a = set(cc.filter(pl.col("seg") == "A")["threshold"].to_list())
        thr_b = set(cc.filter(pl.col("seg") == "B")["threshold"].to_list())
        assert thr_a != thr_b  # data-driven, so groups differ

    def test_int_thresholds_on_roc_curve(self, df):
        roc = pb_roc_curve(df, "y", "score", thresholds=10).collect()
        assert roc.height == 10
        assert roc.columns == ["threshold", "fpr", "tpr"]

    def test_grid_per_group(self, df):
        cc = confusion_curve(df, "y", "score", thresholds=[0.5], by="seg").collect()
        assert cc.columns[0] == "seg"
        assert set(cc["seg"]) == {"A", "B"}
        assert cc.height == 2

    @pytest.mark.parametrize(
        ("fn", "kwargs", "cols"),
        [
            (pb_roc_curve, {}, ["seg", "threshold", "fpr", "tpr"]),
            (pb_pr_curve, {}, ["seg", "threshold", "precision", "recall"]),
            (pb_det_curve, {}, ["seg", "threshold", "fpr", "fnr"]),
        ],
    )
    def test_wrappers_grid_plus_by(self, df, fn, kwargs, cols):
        # Regression: the grid (thresholds=) + by= combination once left the cells
        # list-typed in the lazy schema, so the wrapper's rate select panicked.
        out = fn(df, "y", "score", thresholds=10, by="seg", **kwargs).collect()
        assert out.columns == cols
        assert out.height == 10 * df["seg"].n_unique()

    def test_expected_cost_grid_plus_by(self, df):
        out = pb_expected_cost(df, "y", "score", {"fn": 5.0}, thresholds=10, by="seg").collect()
        assert out.columns == ["seg", "threshold", "cost"]
        assert out.height == 10 * df["seg"].n_unique()

    def test_grid_rejects_bool_thresholds(self, df):
        # bool is an int subclass but not a valid threshold count.
        with pytest.raises(TypeError, match="not a bool"):
            confusion_curve(df, "y", "score", thresholds=True)

    def test_grid_by_weighted(self, df):
        out = pb_roc_curve(df, "y", "score", thresholds=8, by="seg", weight="w").collect()
        assert out.height == 8 * df["seg"].n_unique()


class TestWeighted:
    def test_weighted_roc_auc_matches_sklearn(self, df):
        roc = pb_roc_curve(df, "y", "score", weight="w").collect()
        auc = _trapezoid(roc["tpr"].to_numpy(), roc["fpr"].to_numpy())
        sk = roc_auc_score(df["y"], df["score"], sample_weight=df["w"])
        assert auc == pytest.approx(sk, abs=1e-9)


class TestLazy:
    def test_returns_lazyframe(self, df):
        for fn in (pb_roc_curve, pb_pr_curve, pb_det_curve):
            assert isinstance(fn(df, "y", "score"), pl.LazyFrame)
        assert isinstance(pb_expected_cost(df, "y", "score", {"fn": 1.0}), pl.LazyFrame)

    def test_accepts_lazyframe_input(self, df):
        out = pb_roc_curve(df.lazy(), "y", "score").collect()
        assert out.height > 0


_PROP = settings(
    deadline=None, database=None, suppress_health_check=[HealthCheck.differing_executors]
)


@given(
    n=st.integers(min_value=5, max_value=400), seed=st.integers(min_value=0, max_value=2**32 - 1)
)
@_PROP
def test_roc_curve_auc_matches_sklearn_property(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    if y.sum() in (0, n):  # need both classes for an ROC
        return
    score = rng.random(n)
    df = pl.DataFrame({"y": y, "score": score})
    roc = pb_roc_curve(df, "y", "score").collect()
    auc = _trapezoid(roc["tpr"].to_numpy(), roc["fpr"].to_numpy())
    assert auc == pytest.approx(roc_auc_score(y, score), abs=1e-9)


@given(
    n=st.integers(min_value=20, max_value=400), seed=st.integers(min_value=0, max_value=2**32 - 1)
)
@_PROP
def test_pr_curve_ap_matches_sklearn_property(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    if y.sum() in (0, n):
        return
    score = rng.random(n)
    df = pl.DataFrame({"y": y, "score": score})
    pr = pb_pr_curve(df, "y", "score").collect().sort("recall")
    ap = float(np.sum(np.diff(pr["recall"].to_numpy(), prepend=0.0) * pr["precision"].to_numpy()))
    assert ap == pytest.approx(average_precision_score(y, score), abs=1e-6)


@given(
    n=st.integers(min_value=10, max_value=400),
    k=st.integers(min_value=1, max_value=25),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    grouped=st.booleans(),
)
@_PROP
def test_grid_cells_partition_each_group_property(n, k, seed, grouped):
    # Fuzzes the thresholds= grid path (incl. the grid+by case that once panicked):
    # within each group, every threshold's cells are non-negative, sum to the group
    # size, and tp+fn equals the group's positive count.
    rng = np.random.default_rng(seed)
    df = pl.DataFrame(
        {"seg": rng.choice(["A", "B"], n), "y": rng.integers(0, 2, n), "score": rng.random(n)}
    )
    by = "seg" if grouped else None
    cc = (
        confusion_curve(df, "y", "score", thresholds=k, by=by)
        .collect()
        .with_columns(
            tot=pl.col("tp") + pl.col("fp") + pl.col("fn") + pl.col("tn"),
            pos=pl.col("tp") + pl.col("fn"),
        )
    )
    assert (cc.select(pl.min_horizontal("tp", "fp", "fn", "tn")).to_series() >= 0).all()
    if grouped:
        sizes = df.group_by("seg").agg(pl.len().alias("n"), pl.col("y").sum().alias("p"))
        j = cc.join(sizes, on="seg")
        assert (j["tot"] == j["n"]).all()
        assert (j["pos"] == j["p"]).all()
    else:
        assert (cc["tot"] == n).all()
        assert (cc["pos"] == int(df["y"].sum())).all()
