"""Tests for the Jaccard index (jaccard_score) classification metric."""

import hypothesis
import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import jaccard_score as sklearn_jaccard

from polarbearings.classification import jaccard_score


class TestJaccardBasic:
    def test_known_value(self):
        # preds at threshold 0.5: [0, 1, 1, 1]; labels: [0, 0, 1, 1]
        # TP=2, FP=1, FN=0 -> 2 / 3
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.2, 0.8, 0.6, 0.9]})
        result = df.select(jaccard_score("label", "prob")).to_series()[0]
        assert result == pytest.approx(2 / 3, rel=1e-9)

    def test_matches_sklearn_basic(self):
        labels = [0, 0, 1, 1]
        probs = [0.2, 0.8, 0.6, 0.9]
        preds = [1 if p >= 0.5 else 0 for p in probs]
        df = pl.DataFrame({"label": labels, "prob": probs})
        result = df.select(jaccard_score("label", "prob")).to_series()[0]
        expected = sklearn_jaccard(labels, preds)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_perfect_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
        result = df.select(jaccard_score("label", "prob")).to_series()[0]
        assert result == pytest.approx(1.0, rel=1e-9)

    def test_various_thresholds(self):
        labels = [0, 0, 1, 1, 0, 1]
        probs = [0.1, 0.4, 0.35, 0.8, 0.3, 0.6]
        df = pl.DataFrame({"label": labels, "prob": probs})
        for threshold in [0.2, 0.5, 0.7]:
            result = df.select(jaccard_score("label", "prob", threshold=threshold)).to_series()[0]
            preds = [1 if p >= threshold else 0 for p in probs]
            tp_fp_fn = (
                sum(preds)
                + sum(labels)
                - sum(1 for p, t in zip(preds, labels, strict=True) if p == 1 and t == 1)
            )
            if tp_fp_fn == 0:
                assert result is None
            else:
                expected = sklearn_jaccard(labels, preds)
                assert result == pytest.approx(expected, rel=1e-9)


class TestJaccardWeighted:
    def test_weighted_matches_sklearn(self):
        labels = [0, 0, 1, 1, 1]
        probs = [0.2, 0.8, 0.3, 0.6, 0.9]
        weights = [1.0, 2.0, 0.5, 3.0, 1.5]
        df = pl.DataFrame({"label": labels, "prob": probs, "w": weights})
        result = df.select(jaccard_score("label", "prob", weight="w")).to_series()[0]
        preds = [1 if p >= 0.5 else 0 for p in probs]
        expected = sklearn_jaccard(labels, preds, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-9)


class TestJaccardPosLabel:
    def test_pos_label_zero(self):
        # In this package, ``prob >= threshold`` marks the *positive* prediction
        # regardless of pos_label. So with pos_label=0, a row predicted positive
        # (prob >= threshold) is predicted to be class 0. Build sklearn's label
        # array the same way so the two agree.
        labels = [0, 0, 1, 1, 0]
        probs = [0.2, 0.8, 0.6, 0.9, 0.1]
        df = pl.DataFrame({"label": labels, "prob": probs})
        result = df.select(jaccard_score("label", "prob", pos_label=0)).to_series()[0]
        # predicted positive (prob >= 0.5) -> predicted class 0; else class 1.
        preds = [0 if p >= 0.5 else 1 for p in probs]
        expected = sklearn_jaccard(labels, preds, pos_label=0)
        assert result == pytest.approx(expected, rel=1e-9)


class TestJaccardGroupBy:
    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["a", "a", "a", "b", "b", "b"],
                "label": [0, 1, 1, 0, 1, 1],
                "prob": [0.2, 0.6, 0.9, 0.8, 0.3, 0.7],
            }
        )
        result = df.group_by("group").agg(jaccard_score("label", "prob")).sort("group")
        col = result.columns[1]
        for grp in ["a", "b"]:
            sub = df.filter(pl.col("group") == grp)
            labels = sub["label"].to_list()
            preds = [1 if p >= 0.5 else 0 for p in sub["prob"].to_list()]
            expected = sklearn_jaccard(labels, preds)
            got = result.filter(pl.col("group") == grp)[col][0]
            assert got == pytest.approx(expected, rel=1e-9)


class TestJaccardDegenerate:
    def test_empty(self):
        df = pl.DataFrame({"label": [], "prob": []}, schema={"label": pl.Int64, "prob": pl.Float64})
        result = df.select(jaccard_score("label", "prob")).to_series()[0]
        assert result is None

    def test_no_positives_anywhere(self):
        # All negative labels and no positive predictions -> TP+FP+FN == 0 -> null.
        df = pl.DataFrame({"label": [0, 0, 0], "prob": [0.1, 0.2, 0.3]})
        result = df.select(jaccard_score("label", "prob", threshold=0.9)).to_series()[0]
        assert result is None


class TestJaccardRandom:
    def test_random_data(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = int(rng.integers(20, 100))
            labels = rng.integers(0, 2, n)
            probs = rng.random(n)
            threshold = float(rng.random())
            df = pl.DataFrame({"label": labels, "prob": probs})
            result = df.select(jaccard_score("label", "prob", threshold=threshold)).to_series()[0]
            preds = (probs >= threshold).astype(int)
            if preds.sum() == 0 and labels.sum() == 0:
                assert result is None
            else:
                expected = sklearn_jaccard(labels, preds)
                assert result == pytest.approx(expected, rel=1e-5)


class TestJaccardHypothesis:
    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_jaccard_matches_sklearn(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=4, max_value=200), label="size")
        labels = data.draw(
            arrays(dtype=np.int8, shape=size, elements=st.integers(min_value=0, max_value=1)),
            label="labels",
        )
        probs = data.draw(
            arrays(
                dtype=np.float64,
                shape=size,
                elements=st.floats(
                    min_value=0.0,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                ),
            ),
            label="probs",
        )
        threshold = data.draw(st.floats(min_value=0.01, max_value=0.99), label="threshold")

        preds = (probs >= threshold).astype(int)
        # jaccard is undefined (null here) only when union is empty.
        hypothesis.assume(preds.sum() > 0 or labels.sum() > 0)

        df = pl.DataFrame({"label": labels.tolist(), "prob": probs.tolist()})
        result = df.select(jaccard_score("label", "prob", threshold=threshold)).to_series()[0]
        expected = sklearn_jaccard(labels, preds)
        assert result == pytest.approx(expected, rel=1e-5)
