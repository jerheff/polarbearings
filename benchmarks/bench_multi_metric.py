"""Benchmarks for computing a *suite* of metrics in a single Polars pass.

The single-metric ``bench_*.py`` files isolate per-metric speed. This file tests
the *composability* thesis instead: because every polarbearings metric is a plain
``pl.Expr``, dropping a whole suite of them into ONE ``df.select([...])`` lets
Polars parallelize across the independent output expressions and share the
column scans / common subexpressions — the same structural win that powers the
``group_by`` story, but along the "many metrics at once" axis.

Three approaches are compared at each size on the shared binary-classification
fixtures:

* ``polarbearings_one_select`` — every metric in a single ``df.select([...])`` (the
  headline).
* ``polarbearings_n_selects`` — the same metrics, each in its own ``df.select(...)``,
  summed. Isolates the fuse/parallel benefit from raw per-metric speed by paying
  N separate query-plan + scan overheads.
* ``sklearn_sequence`` — the corresponding sklearn functions called one after
  another, the way a typical evaluation script accumulates a report.

Metric suite (12 metrics, all with a sklearn analog so the three approaches are
genuinely comparable):

* threshold-based (at 0.5): precision, recall, f1, accuracy, balanced_accuracy,
  specificity (= recall of the negative class), mcc, cohens_kappa;
* probability-based: roc_auc, average_precision, log_loss, brier_score.

polarbearings also exposes ``fbeta_score`` and ``gini_coefficient``; they are left
out of the comparable suite because fbeta duplicates f1 at beta=1 and gini has no
direct sklearn function (it is ``2 * roc_auc - 1``), so including them would make
the three approaches no longer line up one-to-one.
"""

from typing import Any

import polars as pl
from pytest_benchmark.fixture import BenchmarkFixture
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.metrics import (
    f1_score as sklearn_f1,
)
from sklearn.metrics import (
    log_loss as sklearn_log_loss,
)
from sklearn.metrics import (
    matthews_corrcoef as sklearn_mcc,
)
from sklearn.metrics import (
    precision_score as sklearn_precision,
)
from sklearn.metrics import (
    recall_score as sklearn_recall,
)

from polarbearings import (
    accuracy,
    average_precision,
    balanced_accuracy,
    brier_score,
    cohens_kappa,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision,
    recall,
    roc_auc,
    specificity,
)

THRESHOLD = 0.5

# (name, builder) for the threshold-based metrics; each builder returns a named
# pl.Expr over the "label"/"prob" columns at THRESHOLD.
_THRESHOLD_METRICS = [
    ("precision", lambda: precision("label", "prob", THRESHOLD)),
    ("recall", lambda: recall("label", "prob", THRESHOLD)),
    ("f1", lambda: f1_score("label", "prob", THRESHOLD)),
    ("accuracy", lambda: accuracy("label", "prob", THRESHOLD)),
    ("balanced_accuracy", lambda: balanced_accuracy("label", "prob", THRESHOLD)),
    ("specificity", lambda: specificity("label", "prob", THRESHOLD)),
    ("mcc", lambda: matthews_corrcoef("label", "prob", THRESHOLD)),
    ("cohens_kappa", lambda: cohens_kappa("label", "prob", THRESHOLD)),
]

# (name, builder) for the probability-based metrics (no threshold).
_PROB_METRICS = [
    ("roc_auc", lambda: roc_auc("label", "prob")),
    ("average_precision", lambda: average_precision("label", "prob")),
    ("log_loss", lambda: log_loss("label", "prob")),
    ("brier_score", lambda: brier_score("label", "prob")),
]

_ALL_METRICS = _THRESHOLD_METRICS + _PROB_METRICS


def _sklearn_suite(labels: Any, probs: Any) -> dict[str, float]:
    """Compute the comparable 12-metric suite the sklearn way, in sequence."""
    preds = (probs >= THRESHOLD).astype(int)
    return {
        "precision": sklearn_precision(labels, preds, zero_division=0),
        "recall": sklearn_recall(labels, preds, zero_division=0),
        "f1": sklearn_f1(labels, preds, zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "specificity": sklearn_recall(labels, preds, pos_label=0, zero_division=0),
        "mcc": sklearn_mcc(labels, preds),
        "cohens_kappa": cohen_kappa_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
        "average_precision": average_precision_score(labels, probs),
        "log_loss": sklearn_log_loss(labels, probs),
        "brier_score": brier_score_loss(labels, probs),
    }


class TestMultiMetricSuite:
    """A 12-metric suite three ways, on the shared ``binary_probs`` fixture.

    Probabilities are independent of the label here (``binary_probs``), so the
    metrics are well-defined but unremarkable in value — the point is the cost of
    *computing* them, not their numbers.
    """

    def test_polarbearings_one_select(
        self, benchmark: BenchmarkFixture, binary_probs: tuple[Any, Any, int]
    ) -> None:
        labels, probs, n = binary_probs
        benchmark.group = f"Multi-metric suite (12) n={n}"
        df = pl.DataFrame({"label": labels, "prob": probs})
        exprs = [build() for _, build in _ALL_METRICS]

        def compute() -> Any:
            return df.select(exprs).row(0)

        benchmark(compute)

    def test_polarbearings_n_selects(
        self, benchmark: BenchmarkFixture, binary_probs: tuple[Any, Any, int]
    ) -> None:
        labels, probs, n = binary_probs
        benchmark.group = f"Multi-metric suite (12) n={n}"
        df = pl.DataFrame({"label": labels, "prob": probs})

        def compute() -> dict[str, float]:
            return {name: df.select(build()).to_series()[0] for name, build in _ALL_METRICS}

        benchmark(compute)

    def test_sklearn_sequence(
        self, benchmark: BenchmarkFixture, binary_probs: tuple[Any, Any, int]
    ) -> None:
        labels, probs, n = binary_probs
        benchmark.group = f"Multi-metric suite (12) n={n}"

        def compute() -> dict[str, float]:
            return _sklearn_suite(labels, probs)

        benchmark(compute)
