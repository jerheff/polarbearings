"""Tests for ROC AUC metric."""

import hypothesis
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import roc_auc_score

from polarbearings import roc_auc


def test_perfect_classification():
    df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(1.0)


def test_worst_classification():
    df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.9, 0.8, 0.2, 0.1]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(0.0)


@given(
    labels=arrays(
        dtype=np.int8,
        shape=st.integers(min_value=4, max_value=100),
        elements=st.integers(min_value=0, max_value=1),
    )
)
@settings(
    deadline=None, database=None, suppress_health_check=[hypothesis.HealthCheck.differing_executors]
)
def test_constant_score_property(labels):
    labels = np.array([0, 1, 0, 1])
    scores = np.full_like(labels, 0.5, dtype=float)
    df = pl.DataFrame({"label": labels.tolist(), "score": scores.tolist()})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(0.5)


@given(size=st.integers(min_value=10000, max_value=50000))
@settings(
    deadline=None, database=None, suppress_health_check=[hypothesis.HealthCheck.differing_executors]
)
def test_random_noise_classification_property(size):
    seed = np.random.randint(0, 2**32 - 1)
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, 2, size=size)
    hypothesis.assume(np.sum(labels) > 0 and np.sum(labels) < size)
    scores = rng.rand(size)
    df = pl.DataFrame({"label": labels.tolist(), "score": scores.tolist()})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(0.5, rel=0.05)


@given(st.data())
@settings(
    deadline=None, database=None, suppress_health_check=[hypothesis.HealthCheck.differing_executors]
)
def test_auc_matches_sklearn(data: st.DataObject):
    size = data.draw(st.integers(min_value=1, max_value=1000), label="size")
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
    hypothesis.assume(np.min(labels) != np.max(labels))
    df = pl.DataFrame({"label": labels.tolist(), "score": scores.tolist()})
    our_auc = df.select(roc_auc("label", "score")).to_series()[0]
    sklearn_auc = roc_auc_score(labels, scores)
    assert our_auc == pytest.approx(sklearn_auc, rel=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_positive_single_negative():
    df = pl.DataFrame({"label": [0, 1], "score": [0.3, 0.7]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(1.0)


def test_inverted_single_pair():
    df = pl.DataFrame({"label": [0, 1], "score": [0.7, 0.3]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(0.0)


def test_tied_single_pair():
    df = pl.DataFrame({"label": [0, 1], "score": [0.5, 0.5]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(roc_auc_score([0, 1], [0.5, 0.5]))


def test_tiny_distinct_scores_not_treated_as_tied():
    # Regression: scores so small their squared deviations underflow to 0.0 made
    # the old var()==0 tie check falsely report a tie. The scores are distinct, so
    # the AUC must be the real rank-based value (0.0 here), not 0.5. (Found by the
    # thorough Hypothesis profile in CI.)
    for scores in ([5.7223975e-303, 0.0], [1e-310, 0.0]):
        df = pl.DataFrame({"label": [0, 1], "score": scores})
        result = df.select(roc_auc("label", "score")).to_series()[0]
        assert result == pytest.approx(roc_auc_score([0, 1], scores))
        assert result == pytest.approx(0.0)


def test_many_positives_one_negative():
    df = pl.DataFrame({"label": [1, 1, 1, 1, 0], "score": [0.9, 0.8, 0.7, 0.6, 0.5]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(roc_auc_score([1, 1, 1, 1, 0], [0.9, 0.8, 0.7, 0.6, 0.5]))


def test_one_positive_many_negatives():
    df = pl.DataFrame({"label": [0, 0, 0, 0, 1], "score": [0.1, 0.2, 0.3, 0.4, 0.9]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(roc_auc_score([0, 0, 0, 0, 1], [0.1, 0.2, 0.3, 0.4, 0.9]))


def test_all_positives_ranked_correctly():
    df = pl.DataFrame({"label": [0, 0, 0, 1, 1, 1], "score": [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(1.0)


def test_partial_ties_in_scores():
    df = pl.DataFrame({"label": [0, 0, 1, 1, 1], "score": [0.3, 0.3, 0.7, 0.7, 0.9]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(roc_auc_score([0, 0, 1, 1, 1], [0.3, 0.3, 0.7, 0.7, 0.9]))


def test_negative_scores():
    df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [-0.5, -0.2, 0.3, 0.8]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(roc_auc_score([0, 0, 1, 1], [-0.5, -0.2, 0.3, 0.8]))


def test_scores_greater_than_one():
    df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.5, 1.2, 2.3, 5.8]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(roc_auc_score([0, 0, 1, 1], [0.5, 1.2, 2.3, 5.8]))


def test_very_small_differences():
    df = pl.DataFrame({"label": [0, 1], "score": [0.500000, 0.500001]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(roc_auc_score([0, 1], [0.500000, 0.500001]))


def test_integer_scores():
    df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [1, 2, 3, 4]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(roc_auc_score([0, 0, 1, 1], [1, 2, 3, 4]))


def test_unsorted_data():
    df = pl.DataFrame({"label": [1, 0, 1, 0, 1], "score": [0.9, 0.2, 0.5, 0.3, 0.8]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(roc_auc_score([1, 0, 1, 0, 1], [0.9, 0.2, 0.5, 0.3, 0.8]))


def test_alternating_labels():
    df = pl.DataFrame({"label": [0, 1, 0, 1, 0, 1], "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(
        roc_auc_score([0, 1, 0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    )


def test_large_dataset():
    np.random.seed(42)
    n = 10000
    labels = np.random.randint(0, 2, n)
    scores = labels * 0.6 + np.random.randn(n) * 0.3
    df = pl.DataFrame({"label": labels, "score": scores})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(roc_auc_score(labels, scores), rel=1e-5)


def test_no_uint32_overflow_above_131k_rows():
    # Regression: the Mann-Whitney U path multiplies positive/negative counts.
    # `sum()` of a boolean is UInt32, so total_pos * total_neg (and
    # total_pos * (total_pos + 1)) overflowed UInt32 once total_pos exceeded
    # ~65k rows (n > ~131k), corrupting the AUC. 200k rows trips the old bug.
    np.random.seed(42)
    n = 200_000
    labels = np.random.randint(0, 2, n)
    scores = labels * 0.6 + np.random.randn(n) * 0.3
    df = pl.DataFrame({"label": labels, "score": scores})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert 0.0 <= result <= 1.0
    assert result == pytest.approx(roc_auc_score(labels, scores), rel=1e-9)


def test_group_by_auc():
    df = pl.DataFrame(
        {
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "label": [0, 0, 1, 1, 0, 1, 1, 1],
            "score": [0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.5, 0.9],
        }
    )
    result = df.group_by("group").agg(roc_auc("label", "score"))
    group_a = result.filter(pl.col("group") == "A")
    assert group_a["roc_auc_label_score"][0] == pytest.approx(
        roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    )
    group_b = result.filter(pl.col("group") == "B")
    assert group_b["roc_auc_label_score"][0] == pytest.approx(
        roc_auc_score([0, 1, 1, 1], [0.3, 0.4, 0.5, 0.9])
    )


def test_with_nulls_filtered():
    df = pl.DataFrame({"label": [0, 0, 1, 1, None], "score": [0.1, 0.2, 0.8, 0.9, 0.5]})
    result = df.drop_nulls().select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(1.0)


def test_multiple_auc_calculations():
    df = pl.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "score1": [0.1, 0.2, 0.8, 0.9],
            "score2": [0.3, 0.3, 0.7, 0.7],
        }
    )
    result = df.select(roc_auc("label", "score1"), roc_auc("label", "score2"))
    assert result["roc_auc_label_score1"][0] == pytest.approx(1.0)
    assert result["roc_auc_label_score2"][0] == pytest.approx(
        roc_auc_score([0, 0, 1, 1], [0.3, 0.3, 0.7, 0.7])
    )


def test_expression_column_inputs_match_string_form():
    """Passing pl.Expr column references should match the string-column form."""
    df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9]})

    string_result = df.select(roc_auc("label", "score")).to_series()[0]

    # Bare pl.col expressions for both columns.
    expr_result = df.select(roc_auc(pl.col("label"), pl.col("score"))).to_series()[0]
    assert expr_result == pytest.approx(string_result)

    # A derived score expression: scaling by 2 is monotonic, so ROC AUC is unchanged.
    derived_result = df.select(roc_auc("label", pl.col("score") * 2)).to_series()[0]
    assert derived_result == pytest.approx(string_result)
