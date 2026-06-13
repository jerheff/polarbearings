"""Tests for average precision metric."""

import hypothesis
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import average_precision_score

from polarbear import average_precision


class TestAveragePrecision:
    """Core average precision tests against sklearn."""

    def test_perfect_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9]})
        result = df.select(average_precision("label", "score")).to_series()[0]
        expected = average_precision_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert result == pytest.approx(expected, rel=1e-5)

    def test_worst_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.9, 0.8, 0.2, 0.1]})
        result = df.select(average_precision("label", "score")).to_series()[0]
        expected = average_precision_score([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
        assert result == pytest.approx(expected, rel=1e-5)

    def test_sklearn_example(self):
        """Test case from sklearn docs."""
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.4, 0.35, 0.8]
        df = pl.DataFrame({"label": labels, "score": scores})
        result = df.select(average_precision("label", "score")).to_series()[0]
        expected = average_precision_score(labels, scores)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_tied_scores(self):
        """Tied scores should match sklearn behavior."""
        labels = [1, 0, 1]
        scores = [0.9, 0.9, 0.8]
        df = pl.DataFrame({"label": labels, "score": scores})
        result = df.select(average_precision("label", "score")).to_series()[0]
        expected = average_precision_score(labels, scores)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_random_data(self):
        np.random.seed(42)
        for _ in range(10):
            n = np.random.randint(10, 200)
            labels = np.random.randint(0, 2, n)
            if labels.sum() == 0 or labels.sum() == n:
                continue
            scores = np.random.rand(n)
            df = pl.DataFrame({"label": labels, "score": scores})
            result = df.select(average_precision("label", "score")).to_series()[0]
            expected = average_precision_score(labels, scores)
            assert result == pytest.approx(expected, rel=1e-5)

    def test_large_dataset(self):
        np.random.seed(42)
        n = 10000
        labels = np.random.randint(0, 2, n)
        scores = labels * 0.6 + np.random.randn(n) * 0.3
        df = pl.DataFrame({"label": labels, "score": scores})
        result = df.select(average_precision("label", "score")).to_series()[0]
        expected = average_precision_score(labels, scores)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "label": [0, 0, 1, 1, 0, 1, 1, 1],
                "score": [0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.5, 0.9],
            }
        )
        result = df.group_by("group").agg(average_precision("label", "score")).sort("group")

        group_a = result.filter(pl.col("group") == "A")["average_precision_label_score"][0]
        expected_a = average_precision_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert group_a == pytest.approx(expected_a, rel=1e-5)

        group_b = result.filter(pl.col("group") == "B")["average_precision_label_score"][0]
        expected_b = average_precision_score([0, 1, 1, 1], [0.3, 0.4, 0.5, 0.9])
        assert group_b == pytest.approx(expected_b, rel=1e-5)


class TestAveragePrecisionEdgeCases:
    def test_no_positives(self):
        df = pl.DataFrame({"label": [0, 0, 0], "score": [0.1, 0.5, 0.9]})
        result = df.select(average_precision("label", "score")).to_series()[0]
        assert result is None

    def test_all_positives(self):
        df = pl.DataFrame({"label": [1, 1, 1], "score": [0.1, 0.5, 0.9]})
        result = df.select(average_precision("label", "score")).to_series()[0]
        expected = average_precision_score([1, 1, 1], [0.1, 0.5, 0.9])
        assert result == pytest.approx(expected, rel=1e-5)

    def test_empty_dataframe(self):
        df = pl.DataFrame(
            {"label": pl.Series([], dtype=pl.Int64), "score": pl.Series([], dtype=pl.Float64)}
        )
        result = df.select(average_precision("label", "score")).to_series()[0]
        assert result is None

    def test_single_positive(self):
        df = pl.DataFrame({"label": [0, 0, 1], "score": [0.1, 0.2, 0.9]})
        result = df.select(average_precision("label", "score")).to_series()[0]
        expected = average_precision_score([0, 0, 1], [0.1, 0.2, 0.9])
        assert result == pytest.approx(expected, rel=1e-5)


class TestAveragePrecisionWeighted:
    def test_uniform_weights_match_unweighted(self):
        labels = [0, 0, 1, 1, 0, 1]
        scores = [0.1, 0.4, 0.35, 0.8, 0.3, 0.6]
        df = pl.DataFrame({"label": labels, "score": scores, "w": [1.0] * 6})
        weighted = df.select(average_precision("label", "score", weight="w")).to_series()[0]
        unweighted = df.select(average_precision("label", "score")).to_series()[0]
        assert weighted == pytest.approx(unweighted, rel=1e-5)

    def test_weighted_matches_sklearn(self):
        np.random.seed(42)
        labels = np.random.randint(0, 2, 100)
        scores = np.random.rand(100)
        weights = np.random.rand(100) + 0.5
        df = pl.DataFrame({"label": labels, "score": scores, "w": weights})
        result = df.select(average_precision("label", "score", weight="w")).to_series()[0]
        expected = average_precision_score(labels, scores, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-4)


class TestAveragePrecisionHypothesis:
    """Property-based tests using Hypothesis."""

    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[hypothesis.HealthCheck.differing_executors],
    )
    def test_matches_sklearn(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=2, max_value=500), label="size")
        labels = data.draw(
            arrays(dtype=np.int8, shape=size, elements=st.integers(min_value=0, max_value=1)),
            label="labels",
        )
        scores = data.draw(
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
            label="scores",
        )
        hypothesis.assume(labels.sum() > 0)
        hypothesis.assume(labels.sum() < len(labels))

        df = pl.DataFrame({"label": labels.tolist(), "score": scores.tolist()})
        result = df.select(average_precision("label", "score")).to_series()[0]
        expected = average_precision_score(labels, scores)
        assert result == pytest.approx(expected, rel=1e-5)

    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[hypothesis.HealthCheck.differing_executors],
    )
    def test_tied_scores_match_sklearn(self, data: st.DataObject):
        """Specifically test with many tied scores."""
        size = data.draw(st.integers(min_value=4, max_value=200), label="size")
        labels = data.draw(
            arrays(dtype=np.int8, shape=size, elements=st.integers(min_value=0, max_value=1)),
            label="labels",
        )
        # Only 5 unique score values to force many ties
        scores = data.draw(
            arrays(
                dtype=np.float64,
                shape=size,
                elements=st.sampled_from([0.1, 0.3, 0.5, 0.7, 0.9]),
            ),
            label="scores",
        )
        hypothesis.assume(labels.sum() > 0)
        hypothesis.assume(labels.sum() < len(labels))

        df = pl.DataFrame({"label": labels.tolist(), "score": scores.tolist()})
        result = df.select(average_precision("label", "score")).to_series()[0]
        expected = average_precision_score(labels, scores)
        assert result == pytest.approx(expected, rel=1e-5)
