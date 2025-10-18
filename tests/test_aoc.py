import hypothesis
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import roc_auc_score

from polarbear import roc_auc


def test_perfect_classification():
    # Perfect separation ⇒ AUC = 1.0
    df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9]})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(1.0)


def test_worst_classification():
    # Inverted scores ⇒ AUC = 0.0
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
@settings(deadline=None)
def test_constant_score_property(labels):
    # Property: For any labels, constant scores should give AUC = 0.5
    labels = np.array([0, 1, 0, 1])  # Ensure we have both classes
    constant_score = 0.5
    scores = np.full_like(labels, constant_score, dtype=float)

    df = pl.DataFrame({"label": labels.tolist(), "score": scores.tolist()})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(0.5)


@given(size=st.integers(min_value=10000, max_value=50000))
@settings(deadline=None)
def test_random_noise_classification_property(size):
    # Property: Randomly generated labels and scores should have AUC ≈ 0.5
    seed = np.random.randint(0, 2**32 - 1)
    rng = np.random.RandomState(seed)

    # Generate random binary labels
    labels = rng.randint(0, 2, size=size)
    # Ensure we have both classes
    hypothesis.assume(np.sum(labels) > 0 and np.sum(labels) < size)

    # Generate uncorrelated random scores
    scores = rng.rand(size)

    df = pl.DataFrame({"label": labels.tolist(), "score": scores.tolist()})
    result = df.select(roc_auc("label", "score")).to_series()[0]
    assert result == pytest.approx(0.5, rel=0.05)


@given(st.data())
@settings(deadline=None)
def test_auc_matches_sklearn(data: st.DataObject):
    # Choose a size for both arrays
    size = data.draw(st.integers(min_value=1, max_value=1000), label="size")

    # Generate arrays of that exact size
    labels = data.draw(
        arrays(dtype=np.int8, shape=size, elements=st.integers(min_value=0, max_value=1)),
        label="labels",
    )
    scores = data.draw(
        arrays(
            dtype=np.float64,
            shape=size,
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        label="scores",
    )

    # Skip test cases where all labels are the same (sklearn raises ValueError in this case)
    hypothesis.assume(np.min(labels) != np.max(labels))

    df = pl.DataFrame({"label": labels.tolist(), "score": scores.tolist()})

    # Calculate AUC using our implementation
    our_auc = df.select(roc_auc("label", "score")).to_series()[0]

    # Calculate AUC using sklearn
    sklearn_auc = roc_auc_score(labels, scores)

    # Compare the results
    assert our_auc == pytest.approx(sklearn_auc, rel=1e-5)
