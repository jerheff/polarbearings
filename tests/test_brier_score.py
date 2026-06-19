"""Tests for Brier score metric."""

import hypothesis
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import brier_score_loss

from polarbearings import brier_score


class TestBrierScore:
    def test_perfect_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.0, 0.0, 1.0, 1.0]})
        result = df.select(brier_score("label", "prob")).to_series()[0]
        assert result == pytest.approx(0.0)

    def test_worst_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [1.0, 1.0, 0.0, 0.0]})
        result = df.select(brier_score("label", "prob")).to_series()[0]
        assert result == pytest.approx(1.0)

    def test_random_predictions(self):
        df = pl.DataFrame({"label": [0, 1, 0, 1], "prob": [0.5, 0.5, 0.5, 0.5]})
        result = df.select(brier_score("label", "prob")).to_series()[0]
        assert result == pytest.approx(0.25)

    def test_matches_sklearn(self):
        labels = [0, 0, 1, 1, 0, 1, 1, 0]
        probs = [0.1, 0.3, 0.6, 0.9, 0.2, 0.7, 0.8, 0.4]
        df = pl.DataFrame({"label": labels, "prob": probs})
        assert df.select(brier_score("label", "prob")).to_series()[0] == pytest.approx(
            brier_score_loss(labels, probs), rel=1e-5
        )

    def test_with_various_probabilities(self):
        np.random.seed(42)
        labels = np.random.randint(0, 2, 100)
        probs = np.random.rand(100)
        df = pl.DataFrame({"label": labels, "prob": probs})
        assert df.select(brier_score("label", "prob")).to_series()[0] == pytest.approx(
            brier_score_loss(labels, probs), rel=1e-5
        )

    def test_brier_score_range(self):
        np.random.seed(42)
        for _ in range(10):
            labels = np.random.randint(0, 2, 50)
            probs = np.random.rand(50)
            df = pl.DataFrame({"label": labels, "prob": probs})
            result = df.select(brier_score("label", "prob")).to_series()[0]
            assert 0.0 <= result <= 1.0

    def test_intermediate_predictions(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.2, 0.3, 0.7, 0.8]})
        assert df.select(brier_score("label", "prob")).to_series()[0] == pytest.approx(
            brier_score_loss([0, 0, 1, 1], [0.2, 0.3, 0.7, 0.8]), rel=1e-5
        )

    def test_probabilities_outside_range(self):
        df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [-0.1, 0.2, 0.8, 1.1]})
        result = df.select(brier_score("label", "prob")).to_series()[0]
        assert np.isfinite(result)
        assert result >= 0

    def test_all_same_probability(self):
        labels = [0, 1, 0, 1, 1, 0]
        probs = [0.5] * 6
        df = pl.DataFrame({"label": labels, "prob": probs})
        assert df.select(brier_score("label", "prob")).to_series()[0] == pytest.approx(
            brier_score_loss(labels, probs), rel=1e-5
        )


class TestBrierScoreHypothesis:
    @given(st.data())
    @settings(
        deadline=None,
        database=None,
        suppress_health_check=[hypothesis.HealthCheck.differing_executors],
    )
    def test_matches_sklearn(self, data: st.DataObject):
        size = data.draw(st.integers(min_value=1, max_value=500), label="size")
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
        df = pl.DataFrame({"label": labels.tolist(), "prob": probs.tolist()})
        assert df.select(brier_score("label", "prob")).to_series()[0] == pytest.approx(
            brier_score_loss(labels, probs), rel=1e-5
        )
