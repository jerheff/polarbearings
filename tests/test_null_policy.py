"""Universal null/NaN policy contract, asserted per scalar metric.

Every scalar metric must return ``null`` when *any* of its ``target``, value
(score/prob/pred), or ``weight`` inputs contains a null or NaN — the "loud
missing" policy documented in the README. The metric correctness tests all run on
clean data and so execute the ``guarded()`` wrapper regardless of whether it
actually fires; the 100%-line-coverage gate therefore cannot catch a metric that
forgets ``guarded()`` (or a routing regression for one metric). This test makes the
shared-helper assumption an explicit, per-metric guarantee, and the partition check
below forces every future ``__all__`` export to be classified as scalar or not.
"""

from functools import partial

import polars as pl
import pytest

import polarbearings as pb

NAN = float("nan")

# Representative, in-domain clean data per metric family. Chosen so the baseline
# result is genuinely non-null (positive/varying regression targets satisfy the
# log/deviance/quantile domains; binary data gives perfect threshold predictions).
_DATA = {
    "binary": ([0.0, 1.0, 0.0, 1.0, 1.0], [0.2, 0.8, 0.3, 0.9, 0.6]),
    "regression": ([1.0, 2.0, 3.0, 4.0, 2.5], [1.5, 2.5, 2.0, 4.5, 3.0]),
    "ranking": ([3.0, 2.0, 0.0, 1.0, 2.0], [0.9, 0.4, 0.1, 0.7, 0.5]),
}

# (name, callable, data-family, accepts-weight)
_SCALAR_METRICS = [
    # probabilistic / ranking-AUC
    ("roc_auc", pb.roc_auc, "binary", True),
    ("average_precision", pb.average_precision, "binary", True),
    ("log_loss", pb.log_loss, "binary", True),
    ("brier_score", pb.brier_score, "binary", True),
    # classification (threshold-based)
    ("precision", pb.precision, "binary", True),
    ("recall", pb.recall, "binary", True),
    ("f1_score", pb.f1_score, "binary", True),
    ("fbeta_score", partial(pb.fbeta_score, beta=2.0), "binary", True),
    ("jaccard_score", pb.jaccard_score, "binary", True),
    ("accuracy", pb.accuracy, "binary", True),
    ("balanced_accuracy", pb.balanced_accuracy, "binary", True),
    ("specificity", pb.specificity, "binary", True),
    ("matthews_corrcoef", pb.matthews_corrcoef, "binary", True),
    ("cohens_kappa", pb.cohens_kappa, "binary", True),
    # calibration scalars
    (
        "expected_calibration_error",
        partial(pb.expected_calibration_error, n_bins=4),
        "binary",
        True,
    ),
    ("maximum_calibration_error", partial(pb.maximum_calibration_error, n_bins=4), "binary", True),
    # gini (continuous magnitude target)
    ("gini_coefficient", pb.gini_coefficient, "regression", True),
    # ranking quality (unweighted)
    ("dcg_score", pb.dcg_score, "ranking", False),
    ("ndcg_score", pb.ndcg_score, "ranking", False),
    # regression (weighted)
    ("mae", pb.mae, "regression", True),
    ("mse", pb.mse, "regression", True),
    ("rmse", pb.rmse, "regression", True),
    ("r2_score", pb.r2_score, "regression", True),
    ("mape", pb.mape, "regression", True),
    ("mean_squared_log_error", pb.mean_squared_log_error, "regression", True),
    ("root_mean_squared_log_error", pb.root_mean_squared_log_error, "regression", True),
    ("explained_variance_score", pb.explained_variance_score, "regression", True),
    ("smape", pb.smape, "regression", True),
    ("log_cosh_loss", pb.log_cosh_loss, "regression", True),
    ("mean_pinball_loss", pb.mean_pinball_loss, "regression", True),
    ("huber_loss", pb.huber_loss, "regression", True),
    ("mean_tweedie_deviance", pb.mean_tweedie_deviance, "regression", True),
    ("mean_poisson_deviance", pb.mean_poisson_deviance, "regression", True),
    ("mean_gamma_deviance", pb.mean_gamma_deviance, "regression", True),
    ("d2_tweedie_score", pb.d2_tweedie_score, "regression", True),
    # regression (intentionally unweighted)
    ("max_error", pb.max_error, "regression", False),
    ("median_absolute_error", pb.median_absolute_error, "regression", False),
    ("d2_absolute_error_score", pb.d2_absolute_error_score, "regression", False),
    ("d2_pinball_score", pb.d2_pinball_score, "regression", False),
]

# Public exports that are deliberately NOT scalar metrics: curve generators (return
# frames), the confusion-matrix struct, threshold/class-weight utilities, bootstrap
# helpers, and the data-splitting expressions. Kept explicit so adding a new export
# without classifying it fails the partition test below.
_NON_SCALAR = {
    "confusion_matrix",
    "confusion_curve",
    "calibration_curve",
    "roc_curve",
    "pr_curve",
    "det_curve",
    "expected_cost",
    "threshold_sweep",
    "quantiles",
    "equal_width",
    "linspace",
    "resolve_thresholds",
    "balanced_sample_weight",
    "balanced_class_weights",
    "BootstrapCI",
    "bootstrap",
    "bootstrap_ci",
    "bootstrap_weight",
    "ci_from_distribution",
    "hash_uniform",
    "hash_split",
    "hash_fold",
    "hash_splits",
}


def test_scalar_metric_registry_partitions_all() -> None:
    """Every ``__all__`` export is either a registered scalar metric or excluded.

    Forces a new export to be classified (scalar -> add to ``_SCALAR_METRICS``;
    otherwise -> ``_NON_SCALAR``) rather than silently skipping the null contract.
    """
    scalar_names = {name for name, *_ in _SCALAR_METRICS}
    assert scalar_names.isdisjoint(_NON_SCALAR)
    assert scalar_names | _NON_SCALAR == set(pb.__all__)


@pytest.mark.parametrize(
    ("name", "fn", "kind", "weighted"),
    _SCALAR_METRICS,
    ids=[m[0] for m in _SCALAR_METRICS],
)
def test_scalar_metric_null_policy(name, fn, kind, weighted) -> None:
    t, v = _DATA[kind]
    n = len(t)

    # Baseline: clean data must be non-null, or the missing-value checks are vacuous.
    base = pl.DataFrame({"t": t, "v": v}).select(fn("t", "v")).to_series()[0]
    assert base is not None, "non-null baseline expected (test would be vacuous)"

    # A null target -> null.
    got = pl.DataFrame({"t": [None, *t[1:]], "v": v}).select(fn("t", "v")).to_series()[0]
    assert got is None, "expected null on a null target"

    # A NaN in the value column -> null.
    got = pl.DataFrame({"t": t, "v": [NAN, *v[1:]]}).select(fn("t", "v")).to_series()[0]
    assert got is None, "expected null on a NaN value"

    if weighted:
        # A null weight -> null...
        got = (
            pl.DataFrame({"t": t, "v": v, "w": [None, *([1.0] * (n - 1))]})
            .select(fn("t", "v", weight="w"))
            .to_series()[0]
        )
        assert got is None, "expected null on a null weight"

        # ...while a clean all-ones weight stays non-null.
        got = (
            pl.DataFrame({"t": t, "v": v, "w": [1.0] * n})
            .select(fn("t", "v", weight="w"))
            .to_series()[0]
        )
        assert got is not None, "expected non-null on clean weighted data"

        # A zero total weight makes the metric undefined -> null, never a 0/0 = NaN.
        got = (
            pl.DataFrame({"t": t, "v": v, "w": [0.0] * n})
            .select(fn("t", "v", weight="w"))
            .to_series()[0]
        )
        assert got is None, "expected null on all-zero weights (undefined, not NaN)"
