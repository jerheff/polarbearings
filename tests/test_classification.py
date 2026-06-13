"""Tests for threshold-based classification metrics."""

import hypothesis
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.metrics import (
    f1_score as sklearn_f1,
)
from sklearn.metrics import (
    precision_score as sklearn_precision,
)
from sklearn.metrics import (
    recall_score as sklearn_recall,
)

from polarbear import (
    accuracy,
    balanced_accuracy,
    f1_score,
    percentile_thresholds,
    precision,
    recall,
    threshold_sweep,
)


class TestPrecision:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
        result = df.select(precision("label", "prob")).to_series()[0]
        preds = [0, 0, 1, 1]
        expected = sklearn_precision([0, 0, 1, 1], preds, zero_division=0)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_various_thresholds(self):
        labels = [0, 0, 1, 1, 0, 1]
        probs = [0.1, 0.4, 0.35, 0.8, 0.3, 0.6]
        df = pl.DataFrame({"label": labels, "prob": probs})

        for threshold in [0.2, 0.5, 0.7]:
            result = df.select(precision("label", "prob", threshold=threshold)).to_series()[0]
            preds = [1 if p >= threshold else 0 for p in probs]
            expected = sklearn_precision(labels, preds, zero_division=0)
            if sum(preds) == 0:
                assert result is None
            else:
                assert result == pytest.approx(expected, rel=1e-5)

    def test_no_positive_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.3, 0.4]})
        result = df.select(precision("label", "prob", threshold=0.9)).to_series()[0]
        assert result is None

    def test_random_data(self):
        np.random.seed(42)
        for _ in range(10):
            n = np.random.randint(20, 100)
            labels = np.random.randint(0, 2, n)
            probs = np.random.rand(n)
            threshold = np.random.rand()
            df = pl.DataFrame({"label": labels, "prob": probs})
            result = df.select(precision("label", "prob", threshold=threshold)).to_series()[0]
            preds = (probs >= threshold).astype(int)
            if preds.sum() == 0:
                assert result is None
            else:
                expected = sklearn_precision(labels, preds)
                assert result == pytest.approx(expected, rel=1e-5)


class TestRecall:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
        result = df.select(recall("label", "prob")).to_series()[0]
        expected = sklearn_recall([0, 0, 1, 1], [0, 0, 1, 1])
        assert result == pytest.approx(expected, rel=1e-5)

    def test_no_actual_positives(self):
        df = pl.DataFrame({"label": [0, 0, 0], "prob": [0.6, 0.7, 0.8]})
        result = df.select(recall("label", "prob")).to_series()[0]
        assert result is None

    def test_random_data(self):
        np.random.seed(123)
        for _ in range(10):
            n = np.random.randint(20, 100)
            labels = np.random.randint(0, 2, n)
            if labels.sum() == 0:
                continue
            probs = np.random.rand(n)
            threshold = np.random.rand()
            df = pl.DataFrame({"label": labels, "prob": probs})
            result = df.select(recall("label", "prob", threshold=threshold)).to_series()[0]
            preds = (probs >= threshold).astype(int)
            expected = sklearn_recall(labels, preds)
            assert result == pytest.approx(expected, rel=1e-5)


class TestF1Score:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
        result = df.select(f1_score("label", "prob")).to_series()[0]
        expected = sklearn_f1([0, 0, 1, 1], [0, 0, 1, 1])
        assert result == pytest.approx(expected, rel=1e-5)

    def test_all_wrong(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.9, 0.8, 0.2, 0.1]})
        result = df.select(f1_score("label", "prob")).to_series()[0]
        expected = sklearn_f1([0, 0, 1, 1], [1, 1, 0, 0])
        assert result == pytest.approx(expected, rel=1e-5)

    def test_random_data(self):
        np.random.seed(99)
        for _ in range(10):
            n = np.random.randint(20, 100)
            labels = np.random.randint(0, 2, n)
            if labels.sum() == 0 or labels.sum() == n:
                continue
            probs = np.random.rand(n)
            threshold = np.random.rand()
            df = pl.DataFrame({"label": labels, "prob": probs})
            result = df.select(f1_score("label", "prob", threshold=threshold)).to_series()[0]
            preds = (probs >= threshold).astype(int)
            expected = sklearn_f1(labels, preds, zero_division=0)
            if (
                2 * ((probs >= threshold).astype(int) * labels).sum()
                + ((probs >= threshold).astype(int) * (1 - labels)).sum()
                + ((probs < threshold).astype(int) * labels).sum()
            ) == 0:
                assert result is None
            else:
                assert result == pytest.approx(expected, rel=1e-5)


class TestAccuracy:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
        result = df.select(accuracy("label", "prob")).to_series()[0]
        expected = accuracy_score([0, 0, 1, 1], [0, 0, 1, 1])
        assert result == pytest.approx(expected, rel=1e-5)

    def test_random_data(self):
        np.random.seed(77)
        for _ in range(10):
            n = np.random.randint(20, 100)
            labels = np.random.randint(0, 2, n)
            probs = np.random.rand(n)
            threshold = np.random.rand()
            df = pl.DataFrame({"label": labels, "prob": probs})
            result = df.select(accuracy("label", "prob", threshold=threshold)).to_series()[0]
            preds = (probs >= threshold).astype(int)
            expected = accuracy_score(labels, preds)
            assert result == pytest.approx(expected, rel=1e-5)


class TestBalancedAccuracy:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
        result = df.select(balanced_accuracy("label", "prob")).to_series()[0]
        expected = balanced_accuracy_score([0, 0, 1, 1], [0, 0, 1, 1])
        assert result == pytest.approx(expected, rel=1e-5)

    def test_single_class_returns_null(self):
        df = pl.DataFrame({"label": [1, 1, 1], "prob": [0.6, 0.7, 0.8]})
        result = df.select(balanced_accuracy("label", "prob")).to_series()[0]
        assert result is None

    def test_random_data(self):
        np.random.seed(55)
        for _ in range(10):
            n = np.random.randint(20, 100)
            labels = np.random.randint(0, 2, n)
            if labels.sum() == 0 or labels.sum() == n:
                continue
            probs = np.random.rand(n)
            threshold = np.random.rand()
            df = pl.DataFrame({"label": labels, "prob": probs})
            result = df.select(balanced_accuracy("label", "prob", threshold=threshold)).to_series()[
                0
            ]
            preds = (probs >= threshold).astype(int)
            expected = balanced_accuracy_score(labels, preds)
            assert result == pytest.approx(expected, rel=1e-5)


class TestThresholdSweep:
    def test_sweep_returns_correct_count(self):
        thresholds = [0.3, 0.5, 0.7]
        exprs = threshold_sweep(f1_score, "label", "prob", thresholds)
        assert len(exprs) == 3

    def test_sweep_results_match_individual(self):
        labels = [0, 0, 1, 1, 0, 1]
        probs = [0.1, 0.4, 0.35, 0.8, 0.3, 0.6]
        df = pl.DataFrame({"label": labels, "prob": probs})
        thresholds = [0.3, 0.5, 0.7]

        sweep_result = df.select(*threshold_sweep(f1_score, "label", "prob", thresholds))

        for t in thresholds:
            individual = df.select(f1_score("label", "prob", threshold=t)).to_series()[0]
            col_name = f"f1_score_label_prob_{t:g}"
            sweep_val = sweep_result[col_name][0]
            if individual is None:
                assert sweep_val is None
            else:
                assert sweep_val == pytest.approx(individual, rel=1e-5)

    def test_sweep_with_precision(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.4, 0.6, 0.9]})
        thresholds = [0.3, 0.5, 0.7]
        result = df.select(*threshold_sweep(precision, "label", "prob", thresholds))
        assert result.shape == (1, 3)

    def test_sweep_in_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "label": [0, 0, 1, 1, 0, 1, 1, 1],
                "prob": [0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.5, 0.9],
            }
        )
        thresholds = [0.3, 0.5]
        result = (
            df.group_by("group")
            .agg(*threshold_sweep(f1_score, "label", "prob", thresholds))
            .sort("group")
        )
        assert result.shape == (2, 3)


class TestPercentileThresholds:
    def test_basic(self):
        scores = pl.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        result = percentile_thresholds(scores, [25, 50, 75])
        assert len(result) == 3
        assert result[0] < result[1] < result[2]
        assert result[1] == pytest.approx(0.5, rel=1e-5)

    def test_with_sweep(self):
        """End-to-end: compute percentile thresholds and sweep."""
        df = pl.DataFrame(
            {
                "label": [0, 0, 1, 1, 0, 1, 0, 1],
                "prob": [0.1, 0.2, 0.3, 0.5, 0.4, 0.7, 0.6, 0.9],
            }
        )
        thresholds = percentile_thresholds(df["prob"], [25, 50, 75])
        result = df.select(*threshold_sweep(precision, "label", "prob", thresholds))
        assert result.shape == (1, 3)


class TestClassificationGroupBy:
    def test_precision_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "label": [0, 0, 1, 1, 0, 1, 1, 1],
                "prob": [0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.5, 0.9],
            }
        )
        result = df.group_by("group").agg(precision("label", "prob")).sort("group")

        group_a_preds = [0, 0, 1, 1]
        expected_a = sklearn_precision([0, 0, 1, 1], group_a_preds)
        assert result["precision_label_prob_0.5"][0] == pytest.approx(expected_a, rel=1e-5)

        group_b_preds = [0, 0, 1, 1]
        expected_b = sklearn_precision([0, 1, 1, 1], group_b_preds)
        assert result["precision_label_prob_0.5"][1] == pytest.approx(expected_b, rel=1e-5)


class TestClassificationEdgeCases:
    def _empty_df(self):
        return pl.DataFrame(
            {"label": pl.Series([], dtype=pl.Int64), "prob": pl.Series([], dtype=pl.Float64)}
        )

    def test_empty_accuracy(self):
        assert self._empty_df().select(accuracy("label", "prob")).to_series()[0] is None

    def test_empty_precision(self):
        assert self._empty_df().select(precision("label", "prob")).to_series()[0] is None

    def test_empty_recall(self):
        assert self._empty_df().select(recall("label", "prob")).to_series()[0] is None

    def test_empty_f1(self):
        assert self._empty_df().select(f1_score("label", "prob")).to_series()[0] is None

    def test_empty_balanced_accuracy(self):
        assert self._empty_df().select(balanced_accuracy("label", "prob")).to_series()[0] is None

    def test_single_row_positive(self):
        df = pl.DataFrame({"label": [1], "prob": [0.9]})
        assert df.select(precision("label", "prob")).to_series()[0] == pytest.approx(1.0)
        assert df.select(recall("label", "prob")).to_series()[0] == pytest.approx(1.0)
        assert df.select(accuracy("label", "prob")).to_series()[0] == pytest.approx(1.0)
        assert df.select(balanced_accuracy("label", "prob")).to_series()[0] is None

    def test_single_row_negative(self):
        df = pl.DataFrame({"label": [0], "prob": [0.9]})
        assert df.select(precision("label", "prob")).to_series()[0] == pytest.approx(0.0)
        assert df.select(recall("label", "prob")).to_series()[0] is None
        assert df.select(accuracy("label", "prob")).to_series()[0] == pytest.approx(0.0)

    def test_all_same_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.6, 0.6, 0.6, 0.6]})
        result = df.select(precision("label", "prob")).to_series()[0]
        expected = sklearn_precision([0, 0, 1, 1], [1, 1, 1, 1])
        assert result == pytest.approx(expected, rel=1e-5)


class TestClassificationMoreGroupBy:
    """group_by tests for recall, accuracy, balanced_accuracy."""

    def _grouped_df(self):
        return pl.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "label": [0, 0, 1, 1, 0, 1, 1, 1],
                "prob": [0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.5, 0.9],
            }
        )

    def test_recall_group_by(self):
        df = self._grouped_df()
        result = df.group_by("group").agg(recall("label", "prob")).sort("group")
        expected_a = sklearn_recall([0, 0, 1, 1], [0, 0, 1, 1])
        expected_b = sklearn_recall([0, 1, 1, 1], [0, 0, 1, 1])
        assert result["recall_label_prob_0.5"][0] == pytest.approx(expected_a, rel=1e-5)
        assert result["recall_label_prob_0.5"][1] == pytest.approx(expected_b, rel=1e-5)

    def test_accuracy_group_by(self):
        df = self._grouped_df()
        result = df.group_by("group").agg(accuracy("label", "prob")).sort("group")
        expected_a = accuracy_score([0, 0, 1, 1], [0, 0, 1, 1])
        expected_b = accuracy_score([0, 1, 1, 1], [0, 0, 1, 1])
        assert result["accuracy_label_prob_0.5"][0] == pytest.approx(expected_a, rel=1e-5)
        assert result["accuracy_label_prob_0.5"][1] == pytest.approx(expected_b, rel=1e-5)

    def test_balanced_accuracy_group_by(self):
        df = self._grouped_df()
        result = df.group_by("group").agg(balanced_accuracy("label", "prob")).sort("group")
        expected_a = balanced_accuracy_score([0, 0, 1, 1], [0, 0, 1, 1])
        expected_b = balanced_accuracy_score([0, 1, 1, 1], [0, 0, 1, 1])
        assert result["balanced_accuracy_label_prob_0.5"][0] == pytest.approx(expected_a, rel=1e-5)
        assert result["balanced_accuracy_label_prob_0.5"][1] == pytest.approx(expected_b, rel=1e-5)


class TestClassificationHypothesis:
    """Property-based tests for classification metrics."""

    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[hypothesis.HealthCheck.differing_executors],
    )
    def test_precision_matches_sklearn(self, data: st.DataObject):
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
        hypothesis.assume(preds.sum() > 0)  # precision undefined otherwise

        df = pl.DataFrame({"label": labels.tolist(), "prob": probs.tolist()})
        result = df.select(precision("label", "prob", threshold=threshold)).to_series()[0]
        expected = sklearn_precision(labels, preds)
        assert result == pytest.approx(expected, rel=1e-5)

    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[hypothesis.HealthCheck.differing_executors],
    )
    def test_recall_matches_sklearn(self, data: st.DataObject):
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

        hypothesis.assume(labels.sum() > 0)  # recall undefined otherwise

        df = pl.DataFrame({"label": labels.tolist(), "prob": probs.tolist()})
        result = df.select(recall("label", "prob", threshold=threshold)).to_series()[0]
        preds = (probs >= threshold).astype(int)
        expected = sklearn_recall(labels, preds)
        assert result == pytest.approx(expected, rel=1e-5)

    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[hypothesis.HealthCheck.differing_executors],
    )
    def test_f1_matches_sklearn(self, data: st.DataObject):
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
        hypothesis.assume(preds.sum() > 0)
        hypothesis.assume(labels.sum() > 0)

        df = pl.DataFrame({"label": labels.tolist(), "prob": probs.tolist()})
        result = df.select(f1_score("label", "prob", threshold=threshold)).to_series()[0]
        expected = sklearn_f1(labels, preds)
        assert result == pytest.approx(expected, rel=1e-5)

    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[hypothesis.HealthCheck.differing_executors],
    )
    def test_accuracy_matches_sklearn(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=2, max_value=200), label="size")
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

        df = pl.DataFrame({"label": labels.tolist(), "prob": probs.tolist()})
        result = df.select(accuracy("label", "prob", threshold=threshold)).to_series()[0]
        preds = (probs >= threshold).astype(int)
        expected = accuracy_score(labels, preds)
        assert result == pytest.approx(expected, rel=1e-5)

    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[hypothesis.HealthCheck.differing_executors],
    )
    def test_balanced_accuracy_matches_sklearn(self, data: st.DataObject):
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

        hypothesis.assume(labels.sum() > 0)
        hypothesis.assume(labels.sum() < len(labels))

        df = pl.DataFrame({"label": labels.tolist(), "prob": probs.tolist()})
        result = df.select(balanced_accuracy("label", "prob", threshold=threshold)).to_series()[0]
        preds = (probs >= threshold).astype(int)
        expected = balanced_accuracy_score(labels, preds)
        assert result == pytest.approx(expected, rel=1e-5)
