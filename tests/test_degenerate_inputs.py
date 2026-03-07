"""Tests for degenerate inputs: single-class, empty, and single-row DataFrames.

ROC AUC is undefined when only one class is present. In that case we return null
(Polars null / Python None), mirroring sklearn's ValueError but without breaking
expression pipelines — especially important inside group_by().agg().
"""

import math

import polars as pl
import pytest

from polarbear import brier_score, log_loss, roc_auc


class TestRocAucSingleClass:
    """ROC AUC should return null when only one class is present."""

    def test_all_positive_labels(self):
        """All labels are 1 — ROC AUC is undefined, should return null."""
        df = pl.DataFrame({"label": [1, 1, 1, 1], "score": [0.1, 0.4, 0.7, 0.9]})
        result = df.select(roc_auc("label", "score")).to_series()[0]
        assert result is None

    def test_all_negative_labels(self):
        """All labels are 0 — ROC AUC is undefined, should return null."""
        df = pl.DataFrame({"label": [0, 0, 0, 0], "score": [0.1, 0.4, 0.7, 0.9]})
        result = df.select(roc_auc("label", "score")).to_series()[0]
        assert result is None

    def test_all_positive_tied_scores(self):
        """All labels are 1 with tied scores — still undefined, should return null."""
        df = pl.DataFrame({"label": [1, 1, 1], "score": [0.5, 0.5, 0.5]})
        result = df.select(roc_auc("label", "score")).to_series()[0]
        assert result is None

    def test_all_negative_tied_scores(self):
        """All labels are 0 with tied scores — still undefined, should return null."""
        df = pl.DataFrame({"label": [0, 0, 0], "score": [0.5, 0.5, 0.5]})
        result = df.select(roc_auc("label", "score")).to_series()[0]
        assert result is None

    def test_single_class_in_group_by(self):
        """A group where all labels are the same class — the realistic danger case.

        Group A has both classes and should return a valid AUC.
        Group B is all-positive and should return null without breaking Group A.
        """
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B", "B"],
                "label": [0, 0, 1, 1, 1, 1, 1],
                "score": [0.1, 0.2, 0.8, 0.9, 0.3, 0.6, 0.9],
            }
        )
        result = df.group_by("group").agg(roc_auc("label", "score")).sort("group")

        group_a_auc = result.filter(pl.col("group") == "A")["roc_auc_label_score"][0]
        assert group_a_auc == pytest.approx(1.0)

        group_b_auc = result.filter(pl.col("group") == "B")["roc_auc_label_score"][0]
        assert group_b_auc is None

    def test_mixed_groups_single_class(self):
        """Multiple groups, some single-class, some valid."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B", "C", "C", "C"],
                "label": [0, 0, 1, 1, 0, 1, 1],
                "score": [0.3, 0.7, 0.4, 0.8, 0.2, 0.6, 0.9],
            }
        )
        result = df.group_by("group").agg(roc_auc("label", "score")).sort("group")

        assert result.filter(pl.col("group") == "A")["roc_auc_label_score"][0] is None
        assert result.filter(pl.col("group") == "B")["roc_auc_label_score"][0] is None
        assert result.filter(pl.col("group") == "C")["roc_auc_label_score"][0] is not None


class TestEmptyDataFrame:
    """Metrics on empty DataFrames should return null."""

    def test_roc_auc_empty(self):
        df = pl.DataFrame(
            {"label": pl.Series([], dtype=pl.Int64), "score": pl.Series([], dtype=pl.Float64)}
        )
        result = df.select(roc_auc("label", "score")).to_series()[0]
        assert result is None

    def test_log_loss_empty(self):
        df = pl.DataFrame(
            {"label": pl.Series([], dtype=pl.Int64), "prob": pl.Series([], dtype=pl.Float64)}
        )
        result = df.select(log_loss("label", "prob")).to_series()[0]
        assert result is None

    def test_brier_score_empty(self):
        df = pl.DataFrame(
            {"label": pl.Series([], dtype=pl.Int64), "prob": pl.Series([], dtype=pl.Float64)}
        )
        result = df.select(brier_score("label", "prob")).to_series()[0]
        assert result is None


class TestSingleRow:
    """Single-row DataFrames."""

    def test_roc_auc_single_positive(self):
        """Single row means single class — should return null."""
        df = pl.DataFrame({"label": [1], "score": [0.8]})
        result = df.select(roc_auc("label", "score")).to_series()[0]
        assert result is None

    def test_roc_auc_single_negative(self):
        """Single row means single class — should return null."""
        df = pl.DataFrame({"label": [0], "score": [0.3]})
        result = df.select(roc_auc("label", "score")).to_series()[0]
        assert result is None

    def test_log_loss_single_row(self):
        """Log loss on a single row should still produce a valid result."""
        df = pl.DataFrame({"label": [1], "prob": [0.9]})
        result = df.select(log_loss("label", "prob")).to_series()[0]
        assert math.isfinite(result)

    def test_brier_score_single_row(self):
        """Brier score on a single row should still produce a valid result."""
        df = pl.DataFrame({"label": [1], "prob": [0.9]})
        result = df.select(brier_score("label", "prob")).to_series()[0]
        assert result == pytest.approx(0.01)
