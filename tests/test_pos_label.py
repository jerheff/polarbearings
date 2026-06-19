"""Tests for the pos_label parameter — arbitrary positive-class values.

Oracle: computing a metric on data encoded with an arbitrary positive label
(int 100, "cancer", True) must exactly equal the same metric on the data
relabeled to {0, 1}, since it is the identical computation with only the label
values swapped. Selecting the *other* class as positive must in turn match the
*inverted* {0, 1} relabeling — proving pos_label genuinely selects the class
rather than just happening to work for one value.
"""

import functools

import polars as pl
import pytest

from polarbearings import (
    accuracy,
    average_precision,
    balanced_accuracy,
    brier_score,
    cohens_kappa,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision,
    recall,
    roc_auc,
    specificity,
)

# fbeta needs a beta; bind one so it shares the metric(target, value, ...) shape.
fbeta2 = functools.partial(fbeta_score, beta=2.0)

# Every binary/classification metric that takes pos_label.
METRICS = [
    roc_auc,
    average_precision,
    log_loss,
    brier_score,
    precision,
    recall,
    f1_score,
    fbeta2,
    accuracy,
    balanced_accuracy,
    specificity,
    matthews_corrcoef,
    cohens_kappa,
]


def _name(metric) -> str:
    """Display name for a metric callable (handles functools.partial)."""
    return getattr(metric, "__name__", None) or metric.func.__name__


LABELS01 = [0, 0, 1, 1, 0, 1, 1, 0]
VALUES = [0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.7, 0.45]

# (id, {0: negative_value, 1: positive_value})
ENCODINGS = [
    ("int", {0: 200, 1: 100}),
    ("str", {0: "healthy", 1: "cancer"}),
    ("bool", {0: False, 1: True}),
]


def _value(df: pl.DataFrame, expr: pl.Expr) -> float:
    return df.select(expr).to_series()[0]


@pytest.mark.parametrize("metric", METRICS, ids=_name)
@pytest.mark.parametrize("enc", ENCODINGS, ids=lambda e: e[0])
def test_pos_label_selects_either_class(metric, enc):
    _, mapping = enc
    pos_val, neg_val = mapping[1], mapping[0]
    encoded = [mapping[v] for v in LABELS01]
    df = pl.DataFrame({"t": encoded, "v": VALUES})

    # Selecting the original positive class reproduces the canonical labeling.
    got_pos = _value(df, metric("t", "v", pos_label=pos_val))
    ref_pos = _value(pl.DataFrame({"t": LABELS01, "v": VALUES}), metric("t", "v"))
    assert got_pos == pytest.approx(ref_pos), f"{_name(metric)} pos_label={pos_val!r}"

    # Selecting the *other* class reproduces the inverted labeling — proving
    # pos_label chooses which value is positive, not just "works for one value".
    inverted = [1 - v for v in LABELS01]
    got_neg = _value(df, metric("t", "v", pos_label=neg_val))
    ref_neg = _value(pl.DataFrame({"t": inverted, "v": VALUES}), metric("t", "v"))
    assert got_neg == pytest.approx(ref_neg), f"{_name(metric)} pos_label={neg_val!r}"


def test_pos_label_choice_changes_result():
    # Cleanly separable data: class 100 scores high, class 200 scores low.
    df = pl.DataFrame({"t": [200, 200, 200, 100, 100, 100], "v": [0.1, 0.2, 0.3, 0.6, 0.8, 0.9]})

    # roc_auc is anti-symmetric: the opposite positive class complements the score.
    auc_pos = _value(df, roc_auc("t", "v", pos_label=100))
    auc_neg = _value(df, roc_auc("t", "v", pos_label=200))
    assert auc_pos == pytest.approx(1.0)
    assert auc_neg == pytest.approx(0.0)
    assert auc_pos != pytest.approx(auc_neg)
    assert auc_pos == pytest.approx(1 - auc_neg)

    # recall is asymmetric too: the choice of positive class changes the value.
    assert _value(df, recall("t", "v", pos_label=100)) == pytest.approx(1.0)
    assert _value(df, recall("t", "v", pos_label=200)) == pytest.approx(0.0)


def test_weighted_pos_label_matches_relabeled():
    weights = [1.0, 2.0, 0.5, 1.5, 1.0, 0.7, 1.2, 0.9]
    encoded = [{0: 200, 1: 100}[v] for v in LABELS01]
    df_enc = pl.DataFrame({"t": encoded, "v": VALUES, "w": weights})
    df_ref = pl.DataFrame({"t": LABELS01, "v": VALUES, "w": weights})
    for metric in METRICS:
        got = _value(df_enc, metric("t", "v", weight="w", pos_label=100))
        ref = _value(df_ref, metric("t", "v", weight="w"))
        assert got == pytest.approx(ref), _name(metric)


def test_alias_suffix_only_for_nondefault_pos_label():
    # Default pos_label=1 leaves the alias unchanged.
    assert precision("t", "v").meta.output_name() == "precision_t_v_0.5"
    assert roc_auc("t", "v").meta.output_name() == "roc_auc_t_v"
    # Non-default pos_label appends a _pos<value> suffix.
    assert precision("t", "v", pos_label=100).meta.output_name() == "precision_t_v_0.5_pos100"
    assert roc_auc("t", "v", pos_label="cancer").meta.output_name() == "roc_auc_t_v_poscancer"
    assert brier_score("t", "v", pos_label=200).meta.output_name() == "brier_score_t_v_pos200"
