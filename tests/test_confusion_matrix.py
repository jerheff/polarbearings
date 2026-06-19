"""Tests for the confusion_matrix struct metric.

confusion_matrix exposes the four cells (tp, fp, fn, tn) that every other
threshold metric is derived from. The oracle is sklearn's ``confusion_matrix``,
which returns ``[[tn, fp], [fn, tp]]`` for ``labels=[0, 1]``; we also check that
metrics derived from the struct agree with the dedicated metric functions.
"""

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import confusion_matrix as sklearn_cm

from polarbearings import confusion_matrix, precision, recall


def _cells(df: pl.DataFrame, expr: pl.Expr) -> dict:
    """Materialise the struct expression to a plain {tp, fp, fn, tn} dict."""
    return df.select(expr.alias("cm")).unnest("cm").row(0, named=True)


class TestConfusionMatrix:
    def test_hand_checked_values(self):
        # preds at 0.5: [0, 1, 1, 1] vs labels [0, 0, 1, 1]
        # tp=2 (rows 2,3), fp=1 (row 1), fn=0, tn=1 (row 0)
        df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.2, 0.8, 0.6, 0.9]})
        cm = _cells(df, confusion_matrix("label", "score"))
        assert cm == {"tp": 2, "fp": 1, "fn": 0, "tn": 1}

    def test_unweighted_fields_are_integers(self):
        df = pl.DataFrame({"label": [0, 1], "score": [0.1, 0.9]})
        schema = df.select(confusion_matrix("label", "score").alias("cm")).unnest("cm").schema
        assert [schema[c] for c in ("tp", "fp", "fn", "tn")] == [pl.Int64] * 4

    def test_weighted_fields_are_floats(self):
        df = pl.DataFrame({"label": [0, 1], "score": [0.1, 0.9], "w": [1.0, 2.0]})
        schema = (
            df.select(confusion_matrix("label", "score", weight="w").alias("cm"))
            .unnest("cm")
            .schema
        )
        assert [schema[c] for c in ("tp", "fp", "fn", "tn")] == [pl.Float64] * 4

    def test_matches_sklearn_random(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            n = int(rng.integers(20, 500))
            labels = rng.integers(0, 2, n)
            scores = rng.random(n)
            threshold = float(rng.uniform(0.2, 0.8))
            df = pl.DataFrame({"label": labels, "score": scores})
            cm = _cells(df, confusion_matrix("label", "score", threshold=threshold))

            preds = (scores >= threshold).astype(int)
            tn, fp, fn, tp = sklearn_cm(labels, preds, labels=[0, 1]).ravel()
            assert (cm["tp"], cm["fp"], cm["fn"], cm["tn"]) == (tp, fp, fn, tn)

    def test_weighted_matches_sklearn(self):
        rng = np.random.default_rng(1)
        n = 300
        labels = rng.integers(0, 2, n)
        scores = rng.random(n)
        weights = rng.uniform(0.1, 3.0, n)
        df = pl.DataFrame({"label": labels, "score": scores, "w": weights})
        cm = _cells(df, confusion_matrix("label", "score", weight="w"))

        preds = (scores >= 0.5).astype(int)
        tn, fp, fn, tp = sklearn_cm(labels, preds, labels=[0, 1], sample_weight=weights).ravel()
        assert cm["tp"] == pytest.approx(tp)
        assert cm["fp"] == pytest.approx(fp)
        assert cm["fn"] == pytest.approx(fn)
        assert cm["tn"] == pytest.approx(tn)

    def test_pos_label_selects_class(self):
        # Encode with a string positive class; selecting it must match the 0/1 oracle.
        labels01 = [0, 0, 1, 1, 0, 1]
        scores = [0.2, 0.7, 0.6, 0.9, 0.3, 0.8]
        encoded = ["healthy" if v == 0 else "cancer" for v in labels01]
        df = pl.DataFrame({"label": encoded, "score": scores})
        df01 = pl.DataFrame({"label": labels01, "score": scores})
        assert _cells(df, confusion_matrix("label", "score", pos_label="cancer")) == _cells(
            df01, confusion_matrix("label", "score")
        )

    def test_pos_label_other_class_matches_relabeled(self):
        # Selecting the negative value as positive flips which *actual* class is
        # positive, but the prediction stays score-driven (score >= threshold) —
        # so the oracle is the same metric on inverted labels with identical
        # scores, matching the library's established pos_label convention.
        labels = [0, 0, 1, 1]
        scores = [0.2, 0.8, 0.6, 0.9]
        df = pl.DataFrame({"label": labels, "score": scores})
        df_inv = pl.DataFrame({"label": [1 - v for v in labels], "score": scores})
        assert _cells(df, confusion_matrix("label", "score", pos_label=0)) == _cells(
            df_inv, confusion_matrix("label", "score", pos_label=1)
        )

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "g": ["a", "a", "b", "b"],
                "label": [0, 1, 0, 1],
                "score": [0.2, 0.9, 0.8, 0.4],
            }
        )
        out = (
            df.group_by("g")
            .agg(confusion_matrix("label", "score").alias("cm"))
            .sort("g")
            .unnest("cm")
        )
        # group a: pred [0,1] vs [0,1] -> tp1 fp0 fn0 tn1
        # group b: pred [1,0] vs [0,1] -> tp0 fp1 fn1 tn0
        assert out.row(0, named=True) == {"g": "a", "tp": 1, "fp": 0, "fn": 0, "tn": 1}
        assert out.row(1, named=True) == {"g": "b", "tp": 0, "fp": 1, "fn": 1, "tn": 0}

    def test_derived_metrics_match_dedicated(self):
        rng = np.random.default_rng(2)
        n = 400
        df = pl.DataFrame({"label": rng.integers(0, 2, n), "score": rng.random(n)})
        cm = _cells(df, confusion_matrix("label", "score"))
        prec = df.select(precision("label", "score")).to_series()[0]
        rec = df.select(recall("label", "score")).to_series()[0]
        assert cm["tp"] / (cm["tp"] + cm["fp"]) == pytest.approx(prec)
        assert cm["tp"] / (cm["tp"] + cm["fn"]) == pytest.approx(rec)

    def test_degenerate_single_class(self):
        # All positives, all predicted positive: only tp populated, others 0.
        df = pl.DataFrame({"label": [1, 1, 1], "score": [0.9, 0.8, 0.7]})
        assert _cells(df, confusion_matrix("label", "score")) == {
            "tp": 3,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    def test_cells_sum_to_total(self):
        rng = np.random.default_rng(3)
        n = 250
        df = pl.DataFrame({"label": rng.integers(0, 2, n), "score": rng.random(n)})
        cm = _cells(df, confusion_matrix("label", "score"))
        assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == n

    @settings(max_examples=50, deadline=None)
    @given(
        labels=arrays(np.int64, st.integers(2, 200), elements=st.integers(0, 1)),
        threshold=st.floats(0.05, 0.95),
        seed=st.integers(0, 10_000),
    )
    def test_property_matches_sklearn(self, labels, threshold, seed):
        rng = np.random.default_rng(seed)
        scores = rng.random(len(labels))
        df = pl.DataFrame({"label": labels, "score": scores})
        cm = _cells(df, confusion_matrix("label", "score", threshold=threshold))

        preds = (scores >= threshold).astype(int)
        tn, fp, fn, tp = sklearn_cm(labels, preds, labels=[0, 1]).ravel()
        assert (cm["tp"], cm["fp"], cm["fn"], cm["tn"]) == (tp, fp, fn, tn)
