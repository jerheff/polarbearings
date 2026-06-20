"""Missing-value policy: any null/NaN in a metric's inputs yields a null result.

Detected on the raw columns before any ``== pos_label`` / ``>= threshold``
comparison, scoped to the whole frame in ``select`` and to each group under
``group_by().agg()``.
"""

import polars as pl
import pytest

import polarbearings as pb


def _val(df: pl.DataFrame, expr: pl.Expr):
    return df.select(expr).to_series()[0]


class TestScalarMetricsNullToNull:
    @pytest.mark.parametrize("bad", [None, float("nan")], ids=["null", "nan"])
    def test_classification_bad_score(self, bad):
        df = pl.DataFrame({"y": [0, 0, 1, 1], "p": [0.2, bad, 0.6, 0.9]})
        assert _val(df, pb.precision("y", "p")) is None
        assert _val(df, pb.f1_score("y", "p")) is None
        assert _val(df, pb.matthews_corrcoef("y", "p")) is None

    def test_classification_null_target(self):
        df = pl.DataFrame({"y": [0, None, 1, 1], "p": [0.2, 0.8, 0.6, 0.9]})
        assert _val(df, pb.precision("y", "p")) is None

    @pytest.mark.parametrize("metric", ["roc_auc", "average_precision", "log_loss", "brier_score"])
    @pytest.mark.parametrize("bad", [None, float("nan")], ids=["null", "nan"])
    def test_rank_and_prob_metrics(self, metric, bad):
        df = pl.DataFrame({"y": [0, 0, 1, 1], "s": [0.1, bad, 0.8, 0.9]})
        assert _val(df, getattr(pb, metric)("y", "s")) is None

    @pytest.mark.parametrize("bad", [None, float("nan")], ids=["null", "nan"])
    def test_regression(self, bad):
        df = pl.DataFrame({"y": [1.0, bad, 3.0, 4.0], "p": [1.1, 2.2, 2.8, 4.5]})
        assert _val(df, pb.mae("y", "p")) is None
        assert _val(df, pb.r2_score("y", "p")) is None
        # missing in pred, not target
        df2 = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "p": [1.1, bad, 2.8, 4.5]})
        assert _val(df2, pb.rmse("y", "p")) is None

    @pytest.mark.parametrize("bad", [None, float("nan")], ids=["null", "nan"])
    def test_gini_and_ranking(self, bad):
        g = pl.DataFrame({"y": [1.0, bad, 3.0, 4.0], "s": [1.0, 2.0, 3.0, 4.0]})
        assert _val(g, pb.gini_coefficient("y", "s")) is None
        r = pl.DataFrame({"rel": [3.0, bad, 3.0, 1.0], "s": [0.9, 0.5, 0.8, 0.1]})
        assert _val(r, pb.ndcg_score("rel", "s")) is None
        assert _val(r, pb.dcg_score("rel", "s")) is None

    def test_null_weight(self):
        df = pl.DataFrame(
            {"y": [0, 0, 1, 1], "p": [0.2, 0.8, 0.6, 0.9], "w": [1.0, None, 1.0, 1.0]}
        )
        assert _val(df, pb.precision("y", "p", weight="w")) is None
        assert _val(df, pb.mae("y", "p", weight="w")) is None

    def test_clean_data_unaffected(self):
        df = pl.DataFrame({"y": [0, 0, 1, 1], "p": [0.2, 0.8, 0.6, 0.9]})
        assert _val(df, pb.precision("y", "p")) == pytest.approx(2 / 3)


class TestGroupScoping:
    def test_only_dirty_group_is_null(self):
        df = pl.DataFrame(
            {"g": ["A", "A", "B", "B", "B"], "y": [0, 1, 0, 0, 1], "p": [0.2, None, 0.3, 0.6, 0.8]}
        )
        out = df.group_by("g").agg(pb.precision("y", "p")).sort("g")
        col = next(c for c in out.columns if c != "g")
        vals = dict(zip(out["g"], out[col], strict=True))
        assert vals["A"] is None  # has a null
        assert vals["B"] is not None  # clean group still computes


class TestStringLabelNoFalsePositive:
    def test_clean_string_labels_compute(self):
        # The universal missing check must NOT flag ordinary string labels.
        df = pl.DataFrame({"y": ["neg", "pos", "neg", "pos"], "p": [0.2, 0.8, 0.3, 0.9]})
        assert _val(df, pb.precision("y", "p", pos_label="pos")) is not None

    def test_null_string_label_is_null(self):
        df = pl.DataFrame({"y": ["neg", None, "neg", "pos"], "p": [0.2, 0.8, 0.3, 0.9]})
        assert _val(df, pb.precision("y", "p", pos_label="pos")) is None


class TestGiniPosLabel:
    def test_string_label_perfect(self):
        df = pl.DataFrame({"y": ["neg", "pos", "neg", "pos"], "score": [0.1, 0.9, 0.2, 0.8]})
        assert _val(df, pb.gini_coefficient("y", "score", pos_label="pos")) == pytest.approx(1.0)

    def test_int_non_one_label(self):
        df = pl.DataFrame({"y": [2, 5, 2, 5], "score": [0.1, 0.9, 0.2, 0.8]})
        assert _val(df, pb.gini_coefficient("y", "score", pos_label=5)) == pytest.approx(1.0)

    def test_weighted_string_label(self):
        df = pl.DataFrame(
            {
                "y": ["neg", "pos", "neg", "pos"],
                "score": [0.1, 0.9, 0.2, 0.8],
                "w": [1.0, 2.0, 1.0, 2.0],
            }
        )
        assert _val(
            df, pb.gini_coefficient("y", "score", weight="w", pos_label="pos")
        ) == pytest.approx(1.0)

    def test_matches_two_auc_minus_one(self):
        rng = pl.Series  # noqa: F841 (keep import-free)
        df = pl.DataFrame({"y": [0, 1, 0, 1, 1, 0], "score": [0.2, 0.9, 0.4, 0.7, 0.8, 0.1]})
        gini = _val(df, pb.gini_coefficient("y", "score", pos_label=1))
        auc = _val(df, pb.roc_auc("y", "score"))
        assert gini == pytest.approx(2 * auc - 1)

    def test_alias_pos_suffix(self):
        df = pl.DataFrame({"y": ["a", "b"], "score": [0.1, 0.9]})
        assert (
            df.select(pb.gini_coefficient("y", "score", pos_label="b")).columns[0].endswith("_posb")
        )
