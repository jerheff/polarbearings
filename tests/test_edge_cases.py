"""Edge case tests for ROC AUC implementation."""

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from polarbear import roc_auc


def test_single_positive_single_negative():
    """Test with minimal data: 1 positive, 1 negative."""
    df = pl.DataFrame({"label": [0, 1], "score": [0.3, 0.7]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(1.0)


def test_inverted_single_pair():
    """Test with minimal data: inverted scores."""
    df = pl.DataFrame({"label": [0, 1], "score": [0.7, 0.3]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(0.0)


def test_tied_single_pair():
    """Test with minimal data: tied scores."""
    df = pl.DataFrame({"label": [0, 1], "score": [0.5, 0.5]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    sklearn_result = roc_auc_score([0, 1], [0.5, 0.5])
    assert result == pytest.approx(sklearn_result)


def test_many_positives_one_negative():
    """Test imbalanced data: many positives, one negative."""
    df = pl.DataFrame(
        {"label": [1, 1, 1, 1, 0], "score": [0.9, 0.8, 0.7, 0.6, 0.5]}
    )
    result = df.select(roc_auc("label", "score")).to_series()[0]
    sklearn_result = roc_auc_score([1, 1, 1, 1, 0], [0.9, 0.8, 0.7, 0.6, 0.5])
    assert result == pytest.approx(sklearn_result)


def test_one_positive_many_negatives():
    """Test imbalanced data: one positive, many negatives."""
    df = pl.DataFrame(
        {"label": [0, 0, 0, 0, 1], "score": [0.1, 0.2, 0.3, 0.4, 0.9]}
    )
    result = df.select(roc_auc("label", "score")).to_series()[0]
    sklearn_result = roc_auc_score([0, 0, 0, 0, 1], [0.1, 0.2, 0.3, 0.4, 0.9])
    assert result == pytest.approx(sklearn_result)


def test_all_positives_ranked_correctly():
    """Test where all positives score higher than all negatives."""
    df = pl.DataFrame(
        {
            "label": [0, 0, 0, 1, 1, 1],
            "score": [0.1, 0.2, 0.3, 0.6, 0.7, 0.8],
        }
    )
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(1.0)


def test_partial_ties_in_scores():
    """Test with some tied scores."""
    df = pl.DataFrame(
        {
            "label": [0, 0, 1, 1, 1],
            "score": [0.3, 0.3, 0.7, 0.7, 0.9],
        }
    )
    result = df.select(roc_auc("label", "score")).to_series()[0]
    sklearn_result = roc_auc_score(
        [0, 0, 1, 1, 1],
        [0.3, 0.3, 0.7, 0.7, 0.9],
    )
    assert result == pytest.approx(sklearn_result)


def test_negative_scores():
    """Test with negative prediction scores."""
    df = pl.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "score": [-0.5, -0.2, 0.3, 0.8],
        }
    )
    result = df.select(roc_auc("label", "score")).to_series()[0]
    sklearn_result = roc_auc_score([0, 0, 1, 1], [-0.5, -0.2, 0.3, 0.8])
    assert result == pytest.approx(sklearn_result)


def test_scores_greater_than_one():
    """Test with prediction scores > 1."""
    df = pl.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "score": [0.5, 1.2, 2.3, 5.8],
        }
    )
    result = df.select(roc_auc("label", "score")).to_series()[0]
    sklearn_result = roc_auc_score([0, 0, 1, 1], [0.5, 1.2, 2.3, 5.8])
    assert result == pytest.approx(sklearn_result)


def test_very_small_differences():
    """Test with very small differences in scores."""
    df = pl.DataFrame(
        {
            "label": [0, 1],
            "score": [0.500000, 0.500001],
        }
    )
    result = df.select(roc_auc("label", "score")).to_series()[0]
    sklearn_result = roc_auc_score([0, 1], [0.500000, 0.500001])
    assert result == pytest.approx(sklearn_result)


def test_integer_scores():
    """Test with integer scores instead of floats."""
    df = pl.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "score": [1, 2, 3, 4],
        }
    )
    result = df.select(roc_auc("label", "score")).to_series()[0]
    sklearn_result = roc_auc_score([0, 0, 1, 1], [1, 2, 3, 4])
    assert result == pytest.approx(sklearn_result)


def test_unsorted_data():
    """Test with data not sorted by score."""
    df = pl.DataFrame(
        {
            "label": [1, 0, 1, 0, 1],
            "score": [0.9, 0.2, 0.5, 0.3, 0.8],
        }
    )
    result = df.select(roc_auc("label", "score")).to_series()[0]
    sklearn_result = roc_auc_score(
        [1, 0, 1, 0, 1],
        [0.9, 0.2, 0.5, 0.3, 0.8],
    )
    assert result == pytest.approx(sklearn_result)


def test_alternating_labels():
    """Test with alternating positive/negative labels."""
    df = pl.DataFrame(
        {
            "label": [0, 1, 0, 1, 0, 1],
            "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    result = df.select(roc_auc("label", "score")).to_series()[0]
    sklearn_result = roc_auc_score(
        [0, 1, 0, 1, 0, 1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    assert result == pytest.approx(sklearn_result)


def test_large_dataset():
    """Test with a larger dataset."""
    np.random.seed(42)
    n = 10000
    labels = np.random.randint(0, 2, n)
    # Make scores somewhat correlated with labels
    scores = labels * 0.6 + np.random.randn(n) * 0.3

    df = pl.DataFrame({"label": labels, "score": scores})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    sklearn_result = roc_auc_score(labels, scores)
    assert result == pytest.approx(sklearn_result, rel=1e-5)


def test_group_by_auc():
    """Test ROC AUC with group_by aggregation."""
    df = pl.DataFrame(
        {
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "label": [0, 0, 1, 1, 0, 1, 1, 1],
            "score": [0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.5, 0.9],
        }
    )

    result = df.group_by("group").agg(roc_auc("label", "score"))

    # Check group A
    group_a = result.filter(pl.col("group") == "A")
    sklearn_a = roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert group_a["roc_auc_label_score"][0] == pytest.approx(sklearn_a)

    # Check group B
    group_b = result.filter(pl.col("group") == "B")
    sklearn_b = roc_auc_score([0, 1, 1, 1], [0.3, 0.4, 0.5, 0.9])
    assert group_b["roc_auc_label_score"][0] == pytest.approx(sklearn_b)


def test_with_nulls_filtered():
    """Test that filtering nulls before AUC works correctly."""
    df = pl.DataFrame(
        {
            "label": [0, 0, 1, 1, None],
            "score": [0.1, 0.2, 0.8, 0.9, 0.5],
        }
    )

    # Filter out nulls first
    result = (
        df.drop_nulls()
        .select(roc_auc("label", "score"))
        .to_series()[0]
    )
    assert result == pytest.approx(1.0)


def test_multiple_auc_calculations():
    """Test multiple AUC calculations in one select."""
    df = pl.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "score1": [0.1, 0.2, 0.8, 0.9],
            "score2": [0.3, 0.3, 0.7, 0.7],
        }
    )

    result = df.select(
        roc_auc("label", "score1"),
        roc_auc("label", "score2"),
    )

    assert result["roc_auc_label_score1"][0] == pytest.approx(1.0)
    sklearn_result = roc_auc_score([0, 0, 1, 1], [0.3, 0.3, 0.7, 0.7])
    assert result["roc_auc_label_score2"][0] == pytest.approx(sklearn_result)
