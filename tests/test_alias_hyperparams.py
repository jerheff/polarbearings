"""A metric's hyper-parameter must reach its output name.

Two calls to the same metric that differ only in a hyper-parameter used to produce
the same alias and collide with a ``DuplicateError`` inside one ``select`` — which
is exactly how a quantile sweep, a Tweedie power comparison, or a bin-count
sensitivity check is written. Each metric below is checked both ways: sweeping the
parameter yields distinct columns, and the *default* call keeps its historical name
so no existing column moves.
"""

import polars as pl
import pytest

from polarbearings import (
    d2_pinball_score,
    d2_tweedie_score,
    dcg_score,
    expected_calibration_error,
    fbeta_score,
    huber_loss,
    maximum_calibration_error,
    mean_pinball_loss,
    mean_tweedie_deviance,
    ndcg_score,
)


@pytest.fixture
def reg():
    # The last residual (4.0) is deliberately large: Huber's `delta` only changes the
    # answer for residuals that straddle it, so an all-small-residual frame would make
    # delta=0.5 and delta=2.0 agree and the sweep assertions vacuous.
    return pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "p": [1.2, 1.8, 3.3, 3.7, 9.0]})


@pytest.fixture
def clf():
    return pl.DataFrame(
        {
            "y": [0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
            "p": [0.1, 0.25, 0.3, 0.45, 0.5, 0.6, 0.7, 0.75, 0.85, 0.9],
        }
    )


# (metric, kwarg name, default value, two non-default values, expected default alias)
_REGRESSION_CASES = [
    (mean_pinball_loss, "alpha", 0.5, (0.1, 0.9), "mean_pinball_loss_y_p"),
    (d2_pinball_score, "alpha", 0.5, (0.1, 0.9), "d2_pinball_score_y_p"),
    (huber_loss, "delta", 1.0, (0.5, 2.0), "huber_loss_y_p"),
    (mean_tweedie_deviance, "power", 0.0, (1.0, 2.0), "mean_tweedie_deviance_y_p"),
    (d2_tweedie_score, "power", 0.0, (1.0, 2.0), "d2_tweedie_score_y_p"),
]


@pytest.mark.parametrize(
    ("metric", "kwarg", "default", "values", "default_alias"), _REGRESSION_CASES
)
def test_regression_hyperparam_sweeps_in_one_select(
    reg, metric, kwarg, default, values, default_alias
):
    """Two non-default values plus the default coexist as three distinct columns."""
    lo, hi = values
    out = reg.select(
        metric("y", "p", **{kwarg: lo}),
        metric("y", "p", **{kwarg: hi}),
        metric("y", "p", **{kwarg: default}),
    )
    assert len(set(out.columns)) == 3, out.columns
    # The default call keeps the historical, parameter-free name.
    assert default_alias in out.columns
    # ...and the two swept columns actually hold different numbers.
    assert out.row(0)[0] != out.row(0)[1]


@pytest.mark.parametrize(
    ("metric", "kwarg", "default", "values", "default_alias"), _REGRESSION_CASES
)
def test_regression_default_alias_is_unchanged(reg, metric, kwarg, default, values, default_alias):
    """Passing the default explicitly names the column exactly as omitting it does."""
    assert reg.select(metric("y", "p")).columns == [default_alias]
    assert reg.select(metric("y", "p", **{kwarg: default})).columns == [default_alias]


def test_regression_hyperparam_composes_with_weight(reg):
    """The parameter fragment sits before the weight suffix, and both survive."""
    frame = reg.with_columns(w=pl.lit(2.0))
    out = frame.select(
        mean_pinball_loss("y", "p", alpha=0.9, weight="w"),
        mean_pinball_loss("y", "p", weight="w"),
    )
    assert out.columns == ["mean_pinball_loss_a0.9_y_p_w", "mean_pinball_loss_y_p_w"]


@pytest.mark.parametrize("metric", [expected_calibration_error, maximum_calibration_error])
def test_calibration_binning_sweeps_in_one_select(clf, metric):
    """Bin count, strategy, and explicit edges each discriminate the output name."""
    name = metric.__name__
    out = clf.select(
        metric("y", "p"),  # default: 10 uniform bins
        metric("y", "p", n_bins=4),
        metric("y", "p", n_bins=4, strategy="quantile"),
        metric("y", "p", strategy="quantile"),
        metric("y", "p", bins=[0.0, 0.5, 1.0]),
    )
    assert out.columns == [
        f"{name}_y_p",
        f"{name}_b4_y_p",
        f"{name}_b4_q_y_p",
        f"{name}_q_y_p",
        f"{name}_e2_y_p",
    ]
    # Spelling the default explicitly must land on the same name as omitting it, so
    # those two really are duplicates and cannot share a select.
    assert clf.select(metric("y", "p", n_bins=10)).columns == [f"{name}_y_p"]
    with pytest.raises(pl.exceptions.DuplicateError):
        clf.select(metric("y", "p"), metric("y", "p", n_bins=10))


@pytest.mark.parametrize("metric", [expected_calibration_error, maximum_calibration_error])
def test_calibration_default_alias_is_unchanged(clf, metric):
    """The default 10-bin uniform binning keeps its parameter-free name."""
    assert clf.select(metric("y", "p")).columns == [f"{metric.__name__}_y_p"]


@pytest.mark.parametrize("metric", [dcg_score, ndcg_score])
def test_ranking_log_base_sweeps_in_one_select(clf, metric):
    """``log_base`` discriminates the name, and still composes with ``k``."""
    out = clf.select(
        metric("y", "p"),
        metric("y", "p", log_base=10.0),
        metric("y", "p", log_base=10.0, k=3),
        metric("y", "p", k=3),
    )
    assert len(set(out.columns)) == 4, out.columns
    name = "dcg" if metric is dcg_score else "ndcg"
    assert out.columns == [
        f"{name}_y_p",
        f"{name}_lb10_y_p",
        f"{name}_lb10_y_p_k3",
        f"{name}_y_p_k3",
    ]


def test_fbeta_precedent_still_holds(clf):
    """``fbeta_score`` already encoded its parameter; that shape is unchanged."""
    out = clf.select(fbeta_score("y", "p", beta=1.0), fbeta_score("y", "p", beta=2.0))
    assert out.columns == ["fbeta_1_y_p_0.5", "fbeta_2_y_p_0.5"]


def test_sweeping_a_hyperparam_used_to_be_impossible(reg):
    """Regression guard: the pre-fix behaviour was a hard DuplicateError."""
    # Same alpha twice genuinely *is* a duplicate, and must still raise.
    with pytest.raises(pl.exceptions.DuplicateError):
        reg.select(mean_pinball_loss("y", "p", alpha=0.9), mean_pinball_loss("y", "p", alpha=0.9))
