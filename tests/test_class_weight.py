"""Tests for balanced sample weights."""

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from polarbearings.class_weight import balanced_class_weights, balanced_sample_weight


def _assert_matches_sklearn(labels: list, target: str = "y") -> None:
    """Check balanced_sample_weight row-by-row against sklearn."""
    df = pl.DataFrame({target: labels})
    result = df.select(balanced_sample_weight(target)).to_series().to_list()
    expected = compute_sample_weight("balanced", np.asarray(labels))
    assert len(result) == len(expected)
    for got, want in zip(result, expected, strict=True):
        assert got == pytest.approx(want)


class TestBalancedSampleWeight:
    def test_balanced_classes(self):
        _assert_matches_sklearn([0, 0, 1, 1])

    def test_imbalanced_classes(self):
        _assert_matches_sklearn([0, 0, 0, 0, 1])

    def test_three_classes(self):
        _assert_matches_sklearn([0, 0, 0, 1, 1, 2])

    def test_string_labels(self):
        _assert_matches_sklearn(["cat", "cat", "dog", "bird", "bird", "bird"])

    def test_boolean_labels(self):
        _assert_matches_sklearn([True, True, False])

    def test_alias(self):
        df = pl.DataFrame({"label": [0, 1]})
        out = df.select(balanced_sample_weight("label"))
        assert out.columns == ["balanced_sample_weight_label"]

    def test_dtype_is_float64(self):
        df = pl.DataFrame({"y": [0, 0, 1]})
        out = df.select(balanced_sample_weight("y"))
        assert out.schema["balanced_sample_weight_y"] == pl.Float64

    def test_single_class(self):
        # n_classes == 1 -> every weight is 1.0, matching sklearn.
        _assert_matches_sklearn([5, 5, 5])

    def test_null_label_excluded(self):
        # A null label is missing, not a class: its row gets a null weight and it
        # does not dilute the real classes (which match sklearn on the non-null y).
        df = pl.DataFrame({"y": [1, 1, 0, None]})
        result = df.select(balanced_sample_weight("y")).to_series().to_list()
        expected = compute_sample_weight("balanced", np.asarray([1, 1, 0]))
        assert result[-1] is None
        for got, want in zip(result[:-1], expected, strict=True):
            assert got == pytest.approx(want)

    def test_used_in_downstream_weighting(self):
        # Weighted sum of a constant column equals n_classes-balanced total.
        df = pl.DataFrame({"y": [0, 0, 0, 1], "value": [1.0, 1.0, 1.0, 1.0]})
        weighted = df.with_columns(balanced_sample_weight("y")).select(
            (pl.col("value") * pl.col("balanced_sample_weight_y")).sum().alias("total")
        )
        total = weighted.to_series()[0]
        # Each class contributes n_samples / n_classes = 4 / 2 = 2.0; two classes -> 4.0.
        assert total == pytest.approx(4.0)

    def test_expression_column_matches_string(self):
        # Passing pl.col(...) (an expression) equals passing the column name.
        df = pl.DataFrame({"label": [0, 0, 0, 1, 2]})
        from_expr = df.select(balanced_sample_weight(pl.col("label")))
        from_str = df.select(balanced_sample_weight("label"))
        assert from_expr.columns == from_str.columns == ["balanced_sample_weight_label"]
        np.testing.assert_allclose(
            from_expr.to_series().to_list(),
            from_str.to_series().to_list(),
        )

    def test_select_with_other_columns(self):
        # Composes inside select alongside other columns over the whole frame.
        df = pl.DataFrame({"g": ["A", "A", "B"], "y": [0, 0, 1]})
        out = df.select("g", balanced_sample_weight("y"))
        assert out.columns == ["g", "balanced_sample_weight_y"]
        np.testing.assert_allclose(
            out["balanced_sample_weight_y"].to_list(),
            compute_sample_weight("balanced", np.array([0, 0, 1])),
        )


class TestBalancedClassWeights:
    def test_three_classes(self):
        labels = [0, 0, 0, 1, 1, 2]
        result = balanced_class_weights(pl.Series("y", labels))
        classes = np.unique(labels)
        expected = compute_class_weight("balanced", classes=classes, y=np.asarray(labels))
        for cls, want in zip(classes, expected, strict=True):
            assert result[int(cls)] == pytest.approx(want)

    def test_string_classes(self):
        labels = ["a", "a", "b"]
        result = balanced_class_weights(pl.Series("y", labels))
        classes = np.unique(labels)
        expected = compute_class_weight("balanced", classes=classes, y=np.asarray(labels))
        for cls, want in zip(classes, expected, strict=True):
            assert result[cls] == pytest.approx(want)

    def test_null_labels_dropped(self):
        # Nulls are missing, not a class: no null key, and the real classes match
        # sklearn computed on the non-null labels.
        result = balanced_class_weights(pl.Series("y", [1, 1, 0, None]))
        assert set(result) == {0, 1}
        expected = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array([1, 1, 0]))
        assert result[0] == pytest.approx(expected[0])
        assert result[1] == pytest.approx(expected[1])


@given(
    size=st.integers(min_value=1, max_value=50),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    n_classes=st.integers(min_value=1, max_value=5),
)
@settings(
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.differing_executors],
)
def test_against_sklearn(size, seed, n_classes):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=size)
    # Ensure at least one of each used class is present via sklearn's own view.
    expected = compute_sample_weight("balanced", labels)

    df = pl.DataFrame({"y": labels})
    result = df.select(balanced_sample_weight("y")).to_series().to_list()
    np.testing.assert_allclose(result, expected, rtol=1e-9)
