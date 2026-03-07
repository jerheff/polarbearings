"""Tests for specificity, fbeta_score, matthews_corrcoef, and cohens_kappa."""

import hypothesis
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import (
    cohen_kappa_score,
)
from sklearn.metrics import (
    fbeta_score as sklearn_fbeta,
)
from sklearn.metrics import (
    matthews_corrcoef as sklearn_mcc,
)

from polarbear import (
    cohens_kappa,
    fbeta_score,
    matthews_corrcoef,
    specificity,
)

# ---------------------------------------------------------------------------
# Specificity
# ---------------------------------------------------------------------------


class TestSpecificity:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
        result = df.select(specificity("label", "prob")).to_series()[0]
        # All negatives correctly predicted as negative => specificity = 1.0
        assert result == pytest.approx(1.0)

    def test_all_positive_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.6, 0.7, 0.8, 0.9]})
        result = df.select(specificity("label", "prob")).to_series()[0]
        # All predicted positive => FP=2, TN=0, specificity=0
        assert result == pytest.approx(0.0)

    def test_no_actual_negatives(self):
        df = pl.DataFrame({"label": [1, 1, 1], "prob": [0.6, 0.7, 0.8]})
        result = df.select(specificity("label", "prob")).to_series()[0]
        assert result is None

    def test_various_thresholds(self):
        labels = [0, 0, 1, 1, 0, 1]
        probs = [0.1, 0.4, 0.35, 0.8, 0.3, 0.6]
        df = pl.DataFrame({"label": labels, "prob": probs})

        for threshold in [0.2, 0.5, 0.7]:
            result = df.select(specificity("label", "prob", threshold=threshold)).to_series()[0]
            preds = [1 if p >= threshold else 0 for p in probs]
            tn = sum(1 for lbl, pr in zip(labels, preds, strict=True) if lbl == 0 and pr == 0)
            fp = sum(1 for lbl, pr in zip(labels, preds, strict=True) if lbl == 0 and pr == 1)
            if tn + fp == 0:
                assert result is None
            else:
                assert result == pytest.approx(tn / (tn + fp), rel=1e-5)

    def test_empty_df(self):
        df = pl.DataFrame(
            {"label": pl.Series([], dtype=pl.Int64), "prob": pl.Series([], dtype=pl.Float64)}
        )
        assert df.select(specificity("label", "prob")).to_series()[0] is None

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "label": [0, 0, 1, 1, 0, 1, 1, 1],
                "prob": [0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.5, 0.9],
            }
        )
        result = df.group_by("group").agg(specificity("label", "prob")).sort("group")
        # Group A: preds=[0,0,1,1], negatives correctly classified => TN=2, FP=0 => 1.0
        assert result["specificity_label_prob_0.5"][0] == pytest.approx(1.0)
        # Group B: preds=[0,0,1,1], label neg=[0], pred for it=0 => TN=1, FP=0 => 1.0
        assert result["specificity_label_prob_0.5"][1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fbeta Score
# ---------------------------------------------------------------------------


class TestFbetaScore:
    def test_f1_is_fbeta_1(self):
        """F-beta with beta=1 should equal F1."""
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.4, 0.6, 0.9]})
        from polarbear import f1_score

        f1_val = df.select(f1_score("label", "prob")).to_series()[0]
        fbeta_val = df.select(fbeta_score("label", "prob", beta=1.0)).to_series()[0]
        assert f1_val == pytest.approx(fbeta_val, rel=1e-10)

    def test_matches_sklearn(self):
        np.random.seed(42)
        for beta in [0.5, 1.0, 2.0]:
            for _ in range(10):
                n = np.random.randint(20, 100)
                labels = np.random.randint(0, 2, n)
                probs = np.random.rand(n)
                threshold = np.random.rand()
                preds = (probs >= threshold).astype(int)
                if preds.sum() == 0 and labels.sum() == 0:
                    continue
                df = pl.DataFrame({"label": labels, "prob": probs})
                result = df.select(
                    fbeta_score("label", "prob", beta=beta, threshold=threshold)
                ).to_series()[0]
                expected = sklearn_fbeta(labels, preds, beta=beta, zero_division=0)
                if (
                    (1 + beta**2) * (preds * labels).sum()
                    + beta**2 * ((1 - preds) * labels).sum()
                    + (preds * (1 - labels)).sum()
                ) == 0:
                    assert result is None
                else:
                    assert result == pytest.approx(expected, rel=1e-5)

    def test_no_positive_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.3, 0.4]})
        result = df.select(fbeta_score("label", "prob", beta=2.0, threshold=0.9)).to_series()[0]
        preds = [0, 0, 0, 0]
        expected = sklearn_fbeta([0, 0, 1, 1], preds, beta=2.0, zero_division=0)
        assert result == pytest.approx(expected, rel=1e-5)

    @given(st.data())
    @settings(deadline=None, suppress_health_check=[hypothesis.HealthCheck.differing_executors])
    def test_fbeta_matches_sklearn_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=4, max_value=200), label="size")
        beta = data.draw(st.sampled_from([0.5, 1.0, 2.0]), label="beta")
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
        hypothesis.assume(preds.sum() > 0)
        hypothesis.assume(labels.sum() > 0)

        df = pl.DataFrame({"label": labels.tolist(), "prob": probs.tolist()})
        result = df.select(
            fbeta_score("label", "prob", beta=beta, threshold=threshold)
        ).to_series()[0]
        expected = sklearn_fbeta(labels, preds, beta=beta)
        assert result == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Matthews Correlation Coefficient
# ---------------------------------------------------------------------------


class TestMatthewsCorrcoef:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
        result = df.select(matthews_corrcoef("label", "prob")).to_series()[0]
        preds = [0, 0, 1, 1]
        expected = sklearn_mcc([0, 0, 1, 1], preds)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_worst_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.9, 0.8, 0.2, 0.1]})
        result = df.select(matthews_corrcoef("label", "prob")).to_series()[0]
        preds = [1, 1, 0, 0]
        expected = sklearn_mcc([0, 0, 1, 1], preds)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_all_same_predictions(self):
        """When all predictions are the same class, MCC is undefined."""
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.3, 0.4]})
        result = df.select(matthews_corrcoef("label", "prob", threshold=0.9)).to_series()[0]
        # All predicted negative => FP=0, TP=0 => (TP+FP)=0 => undefined
        assert result is None

    def test_matches_sklearn(self):
        np.random.seed(42)
        for _ in range(20):
            n = np.random.randint(20, 100)
            labels = np.random.randint(0, 2, n)
            probs = np.random.rand(n)
            threshold = np.random.rand()
            preds = (probs >= threshold).astype(int)
            # Ensure we have all four quadrants
            tp = ((labels == 1) & (preds == 1)).sum()
            fp = ((labels == 0) & (preds == 1)).sum()
            fn = ((labels == 1) & (preds == 0)).sum()
            tn = ((labels == 0) & (preds == 0)).sum()
            if any(x == 0 for x in [tp + fp, tp + fn, tn + fp, tn + fn]):
                continue
            df = pl.DataFrame({"label": labels, "prob": probs})
            result = df.select(matthews_corrcoef("label", "prob", threshold=threshold)).to_series()[
                0
            ]
            expected = sklearn_mcc(labels, preds)
            assert result == pytest.approx(expected, rel=1e-5)

    def test_empty_df(self):
        df = pl.DataFrame(
            {"label": pl.Series([], dtype=pl.Int64), "prob": pl.Series([], dtype=pl.Float64)}
        )
        assert df.select(matthews_corrcoef("label", "prob")).to_series()[0] is None

    @given(st.data())
    @settings(
        deadline=None,
        suppress_health_check=[
            hypothesis.HealthCheck.differing_executors,
            hypothesis.HealthCheck.filter_too_much,
        ],
    )
    def test_mcc_matches_sklearn_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=20, max_value=200), label="size")
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
        # Ensure all quadrants are non-zero so MCC is defined
        hypothesis.assume(((labels == 1) & (preds == 1)).sum() > 0)
        hypothesis.assume(((labels == 0) & (preds == 1)).sum() > 0)
        hypothesis.assume(((labels == 1) & (preds == 0)).sum() > 0)
        hypothesis.assume(((labels == 0) & (preds == 0)).sum() > 0)

        df = pl.DataFrame({"label": labels.tolist(), "prob": probs.tolist()})
        result = df.select(matthews_corrcoef("label", "prob", threshold=threshold)).to_series()[0]
        expected = sklearn_mcc(labels, preds)
        assert result == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Cohen's Kappa
# ---------------------------------------------------------------------------


class TestCohensKappa:
    def test_perfect_agreement(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
        result = df.select(cohens_kappa("label", "prob")).to_series()[0]
        preds = [0, 0, 1, 1]
        expected = cohen_kappa_score([0, 0, 1, 1], preds)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_random_agreement(self):
        """50/50 predictions on 50/50 labels should give kappa near 0."""
        np.random.seed(42)
        labels = np.array([0] * 50 + [1] * 50)
        probs = np.random.rand(100)
        df = pl.DataFrame({"label": labels, "prob": probs})
        result = df.select(cohens_kappa("label", "prob")).to_series()[0]
        preds = (probs >= 0.5).astype(int)
        expected = cohen_kappa_score(labels, preds)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_matches_sklearn(self):
        np.random.seed(42)
        for _ in range(20):
            n = np.random.randint(20, 100)
            labels = np.random.randint(0, 2, n)
            probs = np.random.rand(n)
            threshold = np.random.rand()
            preds = (probs >= threshold).astype(int)
            if len(set(preds)) < 2 or len(set(labels)) < 2:
                continue
            df = pl.DataFrame({"label": labels, "prob": probs})
            result = df.select(cohens_kappa("label", "prob", threshold=threshold)).to_series()[0]
            expected = cohen_kappa_score(labels, preds)
            assert result == pytest.approx(expected, rel=1e-5)

    def test_empty_df(self):
        df = pl.DataFrame(
            {"label": pl.Series([], dtype=pl.Int64), "prob": pl.Series([], dtype=pl.Float64)}
        )
        assert df.select(cohens_kappa("label", "prob")).to_series()[0] is None

    @given(st.data())
    @settings(deadline=None, suppress_health_check=[hypothesis.HealthCheck.differing_executors])
    def test_kappa_matches_sklearn_property(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=10, max_value=200), label="size")
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
        hypothesis.assume(labels.sum() > 0)
        hypothesis.assume(labels.sum() < len(labels))
        hypothesis.assume(preds.sum() > 0)
        hypothesis.assume(preds.sum() < len(preds))

        df = pl.DataFrame({"label": labels.tolist(), "prob": probs.tolist()})
        result = df.select(cohens_kappa("label", "prob", threshold=threshold)).to_series()[0]
        expected = cohen_kappa_score(labels, preds)
        assert result == pytest.approx(expected, rel=1e-5)
