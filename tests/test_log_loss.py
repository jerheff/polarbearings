"""Tests for log loss metric."""

import hypothesis
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss as sklearn_log_loss

from polarbear import brier_score, log_loss


class TestLogLoss:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.0, 0.0, 1.0, 1.0]})
        result = df.select(log_loss("label", "prob")).to_series()[0]
        assert result < 0.001

    def test_worst_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [1.0, 1.0, 0.0, 0.0]})
        result = df.select(log_loss("label", "prob")).to_series()[0]
        assert result > 10

    def test_random_predictions(self):
        df = pl.DataFrame({"label": [0, 1, 0, 1], "prob": [0.5, 0.5, 0.5, 0.5]})
        result = df.select(log_loss("label", "prob")).to_series()[0]
        assert result == pytest.approx(
            sklearn_log_loss([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), rel=1e-5
        )

    def test_matches_sklearn(self):
        labels = [0, 0, 1, 1, 0, 1, 1, 0]
        probs = [0.1, 0.3, 0.6, 0.9, 0.2, 0.7, 0.8, 0.4]
        df = pl.DataFrame({"label": labels, "prob": probs})
        assert df.select(log_loss("label", "prob")).to_series()[0] == pytest.approx(
            sklearn_log_loss(labels, probs), rel=1e-5
        )

    def test_with_various_probabilities(self):
        np.random.seed(42)
        labels = np.random.randint(0, 2, 100)
        probs = np.random.rand(100)
        df = pl.DataFrame({"label": labels, "prob": probs})
        assert df.select(log_loss("label", "prob")).to_series()[0] == pytest.approx(
            sklearn_log_loss(labels, probs), rel=1e-5
        )

    def test_clipping_behavior(self):
        df = pl.DataFrame({"label": [0, 1, 0, 1], "prob": [0.0, 1.0, 0.1, 0.9]})
        result = df.select(log_loss("label", "prob")).to_series()[0]
        assert np.isfinite(result)

    def test_probabilities_outside_range(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [-0.1, 0.2, 0.8, 1.1]})
        result = df.select(log_loss("label", "prob")).to_series()[0]
        assert np.isfinite(result)
        assert result > 0

    def test_all_same_probability(self):
        labels = [0, 1, 0, 1, 1, 0]
        probs = [0.5] * 6
        df = pl.DataFrame({"label": labels, "prob": probs})
        assert df.select(log_loss("label", "prob")).to_series()[0] == pytest.approx(
            sklearn_log_loss(labels, probs), rel=1e-5
        )


class TestLogLossHypothesis:
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
        probs = data.draw(
            arrays(
                dtype=np.float64,
                shape=size,
                elements=st.floats(
                    min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
                ),
            ),
            label="probs",
        )
        hypothesis.assume(labels.sum() > 0)
        hypothesis.assume(labels.sum() < len(labels))
        df = pl.DataFrame({"label": labels.tolist(), "prob": probs.tolist()})
        result = df.select(log_loss("label", "prob")).to_series()[0]
        assert result == pytest.approx(sklearn_log_loss(labels, probs), rel=1e-5)


class TestProbabilisticMetricsTogether:
    """Cross-metric tests for log_loss and brier_score."""

    def test_multiple_metrics_single_select(self):
        df = pl.DataFrame(
            {
                "label": [0, 0, 1, 1],
                "prob": [0.1, 0.2, 0.8, 0.9],
                "score": [0.15, 0.25, 0.75, 0.85],
            }
        )
        result = df.select(
            log_loss("label", "prob"),
            brier_score("label", "prob"),
            log_loss("label", "score"),
            brier_score("label", "score"),
        )
        assert result.columns == [
            "log_loss_label_prob",
            "brier_score_label_prob",
            "log_loss_label_score",
            "brier_score_label_score",
        ]

    def test_grouped_metrics(self):
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "label": [0, 0, 1, 1, 0, 1, 1, 1],
                "prob": [0.1, 0.2, 0.8, 0.9, 0.3, 0.6, 0.7, 0.9],
            }
        )
        result = (
            df.group_by("group")
            .agg(log_loss("label", "prob"), brier_score("label", "prob"))
            .sort("group")
        )
        labels_a, probs_a = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
        labels_b, probs_b = [0, 1, 1, 1], [0.3, 0.6, 0.7, 0.9]
        assert result["log_loss_label_prob"][0] == pytest.approx(
            sklearn_log_loss(labels_a, probs_a), rel=1e-5
        )
        assert result["log_loss_label_prob"][1] == pytest.approx(
            sklearn_log_loss(labels_b, probs_b), rel=1e-5
        )
        assert result["brier_score_label_prob"][0] == pytest.approx(
            brier_score_loss(labels_a, probs_a), rel=1e-5
        )
        assert result["brier_score_label_prob"][1] == pytest.approx(
            brier_score_loss(labels_b, probs_b), rel=1e-5
        )

    def test_log_loss_random_data(self):
        np.random.seed(42)
        for _ in range(10):
            n = np.random.randint(20, 200)
            labels = np.random.randint(0, 2, n)
            probs = np.clip(np.random.rand(n), 0.01, 0.99)
            df = pl.DataFrame({"label": labels, "prob": probs})
            assert df.select(log_loss("label", "prob")).to_series()[0] == pytest.approx(
                sklearn_log_loss(labels, probs), rel=1e-5
            )

    def test_brier_score_random_data(self):
        np.random.seed(42)
        for _ in range(10):
            n = np.random.randint(20, 200)
            labels = np.random.randint(0, 2, n)
            probs = np.random.rand(n)
            df = pl.DataFrame({"label": labels, "prob": probs})
            assert df.select(brier_score("label", "prob")).to_series()[0] == pytest.approx(
                brier_score_loss(labels, probs), rel=1e-5
            )
