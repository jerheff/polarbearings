"""Tests for additional metrics (log loss, Brier score)."""

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss as sklearn_log_loss

from polarbear import brier_score, log_loss


class TestLogLoss:
    """Tests for log loss metric."""

    def test_perfect_predictions(self):
        """Test log loss with perfect predictions."""
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.0, 0.0, 1.0, 1.0]})
        # With clipping, this should be very close to 0
        result = df.select(log_loss("label", "prob")).to_series()[0]
        assert result < 0.001  # Very small due to eps clipping

    def test_worst_predictions(self):
        """Test log loss with inverted predictions."""
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [1.0, 1.0, 0.0, 0.0]})
        result = df.select(log_loss("label", "prob")).to_series()[0]
        # Should be very large (close to -log(eps))
        assert result > 10

    def test_random_predictions(self):
        """Test log loss with 0.5 probabilities."""
        df = pl.DataFrame({"label": [0, 1, 0, 1], "prob": [0.5, 0.5, 0.5, 0.5]})
        result = df.select(log_loss("label", "prob")).to_series()[0]
        sklearn_result = sklearn_log_loss([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        assert result == pytest.approx(sklearn_result, rel=1e-5)

    def test_matches_sklearn(self):
        """Test that our log loss matches sklearn."""
        labels = [0, 0, 1, 1, 0, 1, 1, 0]
        probs = [0.1, 0.3, 0.6, 0.9, 0.2, 0.7, 0.8, 0.4]

        df = pl.DataFrame({"label": labels, "prob": probs})
        our_result = df.select(log_loss("label", "prob")).to_series()[0]
        sklearn_result = sklearn_log_loss(labels, probs)

        assert our_result == pytest.approx(sklearn_result, rel=1e-5)

    def test_with_various_probabilities(self):
        """Test log loss with various probability values."""
        np.random.seed(42)
        labels = np.random.randint(0, 2, 100)
        probs = np.random.rand(100)

        df = pl.DataFrame({"label": labels, "prob": probs})
        our_result = df.select(log_loss("label", "prob")).to_series()[0]
        sklearn_result = sklearn_log_loss(labels, probs)

        assert our_result == pytest.approx(sklearn_result, rel=1e-5)

    def test_clipping_behavior(self):
        """Test that probabilities are properly clipped."""
        # Test with exact 0 and 1 probabilities
        df = pl.DataFrame({"label": [0, 1, 0, 1], "prob": [0.0, 1.0, 0.1, 0.9]})
        result = df.select(log_loss("label", "prob")).to_series()[0]
        # Should not raise an error and should return finite value
        assert np.isfinite(result)

    def test_probabilities_outside_range(self):
        """Test with probabilities slightly outside [0, 1]."""
        # Polars clip will handle these automatically
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [-0.1, 0.2, 0.8, 1.1]})
        result = df.select(log_loss("label", "prob")).to_series()[0]
        # Should clip and return finite value
        assert np.isfinite(result)
        assert result > 0  # Should still have some loss

    def test_all_same_probability(self):
        """Test log loss when all predictions are the same."""
        df = pl.DataFrame(
            {"label": [0, 1, 0, 1, 1, 0], "prob": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}
        )
        result = df.select(log_loss("label", "prob")).to_series()[0]
        sklearn_result = sklearn_log_loss(
            [0, 1, 0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        )
        assert result == pytest.approx(sklearn_result, rel=1e-5)


class TestBrierScore:
    """Tests for Brier score metric."""

    def test_perfect_predictions(self):
        """Test Brier score with perfect predictions."""
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.0, 0.0, 1.0, 1.0]})
        result = df.select(brier_score("label", "prob")).to_series()[0]
        assert result == pytest.approx(0.0)

    def test_worst_predictions(self):
        """Test Brier score with inverted predictions."""
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [1.0, 1.0, 0.0, 0.0]})
        result = df.select(brier_score("label", "prob")).to_series()[0]
        assert result == pytest.approx(1.0)

    def test_random_predictions(self):
        """Test Brier score with 0.5 probabilities."""
        df = pl.DataFrame({"label": [0, 1, 0, 1], "prob": [0.5, 0.5, 0.5, 0.5]})
        result = df.select(brier_score("label", "prob")).to_series()[0]
        # Brier score for always predicting 0.5 is 0.25
        assert result == pytest.approx(0.25)

    def test_matches_sklearn(self):
        """Test that our Brier score matches sklearn."""
        labels = [0, 0, 1, 1, 0, 1, 1, 0]
        probs = [0.1, 0.3, 0.6, 0.9, 0.2, 0.7, 0.8, 0.4]

        df = pl.DataFrame({"label": labels, "prob": probs})
        our_result = df.select(brier_score("label", "prob")).to_series()[0]
        sklearn_result = brier_score_loss(labels, probs)

        assert our_result == pytest.approx(sklearn_result, rel=1e-5)

    def test_with_various_probabilities(self):
        """Test Brier score with various probability values."""
        np.random.seed(42)
        labels = np.random.randint(0, 2, 100)
        probs = np.random.rand(100)

        df = pl.DataFrame({"label": labels, "prob": probs})
        our_result = df.select(brier_score("label", "prob")).to_series()[0]
        sklearn_result = brier_score_loss(labels, probs)

        assert our_result == pytest.approx(sklearn_result, rel=1e-5)

    def test_brier_score_range(self):
        """Test that Brier score is always in [0, 1]."""
        np.random.seed(42)
        for _ in range(10):
            labels = np.random.randint(0, 2, 50)
            probs = np.random.rand(50)

            df = pl.DataFrame({"label": labels, "prob": probs})
            result = df.select(brier_score("label", "prob")).to_series()[0]

            assert 0.0 <= result <= 1.0

    def test_intermediate_predictions(self):
        """Test Brier score with intermediate probability values."""
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.2, 0.3, 0.7, 0.8]})
        result = df.select(brier_score("label", "prob")).to_series()[0]
        sklearn_result = brier_score_loss([0, 0, 1, 1], [0.2, 0.3, 0.7, 0.8])
        assert result == pytest.approx(sklearn_result, rel=1e-5)

    def test_probabilities_outside_range(self):
        """Test Brier score with probabilities outside [0, 1]."""
        # Values outside [0,1] should still compute MSE correctly
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [-0.1, 0.2, 0.8, 1.1]})
        result = df.select(brier_score("label", "prob")).to_series()[0]
        # Should return finite value
        assert np.isfinite(result)
        # Brier score can be > 1 if probabilities are outside [0, 1]
        assert result >= 0

    def test_all_same_probability(self):
        """Test Brier score when all predictions are the same."""
        df = pl.DataFrame(
            {"label": [0, 1, 0, 1, 1, 0], "prob": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}
        )
        result = df.select(brier_score("label", "prob")).to_series()[0]
        sklearn_result = brier_score_loss(
            [0, 1, 0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        )
        assert result == pytest.approx(sklearn_result, rel=1e-5)


class TestMetricsTogether:
    """Test using multiple metrics together."""

    def test_multiple_metrics_single_select(self):
        """Test computing multiple metrics in one select."""
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

        # Check all metrics were computed
        assert len(result.columns) == 4
        assert "log_loss_label_prob" in result.columns
        assert "brier_score_label_prob" in result.columns
        assert "log_loss_label_score" in result.columns
        assert "brier_score_label_score" in result.columns

    def test_grouped_metrics(self):
        """Test metrics with group_by."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "label": [0, 0, 1, 1, 0, 1, 1, 1],
                "prob": [0.1, 0.2, 0.8, 0.9, 0.3, 0.6, 0.7, 0.9],
            }
        )

        result = df.group_by("group").agg(
            log_loss("label", "prob"),
            brier_score("label", "prob"),
        )

        assert len(result) == 2
        assert "log_loss_label_prob" in result.columns
        assert "brier_score_label_prob" in result.columns
