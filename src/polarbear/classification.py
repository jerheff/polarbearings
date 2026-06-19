"""Threshold-based classification metrics implemented as Polars expressions."""

from typing import Protocol

import polars as pl

# A positive-class label may be any scalar value comparable to the target column
# (e.g. 1, 100, "cancer", True). Defaults to 1 for backward compatibility.
_PosLabel = int | float | str | bool


class _MetricFn(Protocol):
    def __call__(
        self,
        target: str,
        prob: str,
        threshold: float = ...,
        weight: str | None = ...,
        pos_label: _PosLabel = ...,
    ) -> pl.Expr: ...


def _confusion_components(
    target: str,
    prob: str,
    threshold: float,
    weight: str | None = None,
    pos_label: _PosLabel = 1,
) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """Compute TP, FP, FN, TN as Polars expressions.

    The positive class is any row where ``target == pos_label``; every other
    value is treated as negative (one-vs-rest).
    """
    is_pos = (pl.col(target) == pos_label).cast(pl.Float64)
    predicted = (pl.col(prob) >= threshold).cast(pl.Float64)

    if weight is not None:
        w = pl.col(weight).cast(pl.Float64)
        tp = (is_pos * predicted * w).sum()
        fp = ((1 - is_pos) * predicted * w).sum()
        fn = (is_pos * (1 - predicted) * w).sum()
        tn = ((1 - is_pos) * (1 - predicted) * w).sum()
    else:
        tp = (is_pos * predicted).sum()
        fp = ((1 - is_pos) * predicted).sum()
        fn = (is_pos * (1 - predicted)).sum()
        tn = ((1 - is_pos) * (1 - predicted)).sum()

    return tp, fp, fn, tn


def _alias(
    name: str,
    target: str,
    prob: str,
    threshold: float,
    weight: str | None,
    pos_label: _PosLabel = 1,
) -> str:
    """Build a consistent alias string.

    A ``_pos{pos_label}`` suffix is appended only when ``pos_label`` is not the
    default ``1``, so existing alias strings are unchanged for the common case.
    """
    alias = f"{name}_{target}_{prob}_{threshold:g}"
    if weight is not None:
        alias += f"_{weight}"
    if pos_label != 1:
        alias += f"_pos{pos_label}"
    return alias


def precision(
    target: str,
    prob: str,
    threshold: float = 0.5,
    weight: str | None = None,
    pos_label: _PosLabel = 1,
) -> pl.Expr:
    """Compute precision at a decision threshold.

    Precision = TP / (TP + FP). Returns null when no positive predictions are made.

    Args:
        target: Column with class labels.
        prob: Column with predicted probabilities.
        threshold: Decision threshold (predict positive if prob >= threshold).
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
    """
    tp, fp, _fn, _tn = _confusion_components(target, prob, threshold, weight, pos_label)
    denom = tp + fp
    result = pl.when(denom == 0).then(None).otherwise(tp / denom)
    return result.alias(_alias("precision", target, prob, threshold, weight, pos_label))


def recall(
    target: str,
    prob: str,
    threshold: float = 0.5,
    weight: str | None = None,
    pos_label: _PosLabel = 1,
) -> pl.Expr:
    """Compute recall at a decision threshold.

    Recall = TP / (TP + FN). Returns null when no actual positives exist.

    Args:
        target: Column with class labels.
        prob: Column with predicted probabilities.
        threshold: Decision threshold (predict positive if prob >= threshold).
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
    """
    tp, _fp, fn, _tn = _confusion_components(target, prob, threshold, weight, pos_label)
    denom = tp + fn
    result = pl.when(denom == 0).then(None).otherwise(tp / denom)
    return result.alias(_alias("recall", target, prob, threshold, weight, pos_label))


def f1_score(
    target: str,
    prob: str,
    threshold: float = 0.5,
    weight: str | None = None,
    pos_label: _PosLabel = 1,
) -> pl.Expr:
    """Compute F1 score at a decision threshold.

    F1 = 2·TP / (2·TP + FP + FN). Returns null when undefined.

    Args:
        target: Column with class labels.
        prob: Column with predicted probabilities.
        threshold: Decision threshold (predict positive if prob >= threshold).
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
    """
    tp, fp, fn, _tn = _confusion_components(target, prob, threshold, weight, pos_label)
    denom = 2 * tp + fp + fn
    result = pl.when(denom == 0).then(None).otherwise(2 * tp / denom)
    return result.alias(_alias("f1_score", target, prob, threshold, weight, pos_label))


def accuracy(
    target: str,
    prob: str,
    threshold: float = 0.5,
    weight: str | None = None,
    pos_label: _PosLabel = 1,
) -> pl.Expr:
    """Compute accuracy at a decision threshold.

    Accuracy = (TP + TN) / (TP + TN + FP + FN). Returns null on empty data.

    Args:
        target: Column with class labels.
        prob: Column with predicted probabilities.
        threshold: Decision threshold (predict positive if prob >= threshold).
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
    """
    tp, fp, fn, tn = _confusion_components(target, prob, threshold, weight, pos_label)
    total = tp + tn + fp + fn
    result = pl.when(total == 0).then(None).otherwise((tp + tn) / total)
    return result.alias(_alias("accuracy", target, prob, threshold, weight, pos_label))


def balanced_accuracy(
    target: str,
    prob: str,
    threshold: float = 0.5,
    weight: str | None = None,
    pos_label: _PosLabel = 1,
) -> pl.Expr:
    """Compute balanced accuracy at a decision threshold.

    Balanced accuracy = (TPR + TNR) / 2 = (recall + specificity) / 2.
    Returns null when either class is absent.

    Args:
        target: Column with class labels.
        prob: Column with predicted probabilities.
        threshold: Decision threshold (predict positive if prob >= threshold).
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
    """
    tp, fp, fn, tn = _confusion_components(target, prob, threshold, weight, pos_label)
    pos_total = tp + fn
    neg_total = tn + fp
    undefined = (pos_total == 0) | (neg_total == 0)
    tpr = tp / pos_total
    tnr = tn / neg_total
    result = pl.when(undefined).then(None).otherwise((tpr + tnr) / 2)
    return result.alias(_alias("balanced_accuracy", target, prob, threshold, weight, pos_label))


def specificity(
    target: str,
    prob: str,
    threshold: float = 0.5,
    weight: str | None = None,
    pos_label: _PosLabel = 1,
) -> pl.Expr:
    """Compute specificity (true negative rate) at a decision threshold.

    Specificity = TN / (TN + FP). Returns null when no actual negatives exist.

    Args:
        target: Column with class labels.
        prob: Column with predicted probabilities.
        threshold: Decision threshold (predict positive if prob >= threshold).
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
    """
    _tp, fp, _fn, tn = _confusion_components(target, prob, threshold, weight, pos_label)
    denom = tn + fp
    result = pl.when(denom == 0).then(None).otherwise(tn / denom)
    return result.alias(_alias("specificity", target, prob, threshold, weight, pos_label))


def fbeta_score(
    target: str,
    prob: str,
    beta: float,
    threshold: float = 0.5,
    weight: str | None = None,
    pos_label: _PosLabel = 1,
) -> pl.Expr:
    """Compute F-beta score at a decision threshold.

    F_beta = (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP).
    Generalizes F1 (beta=1). Use beta < 1 to weight precision higher,
    beta > 1 to weight recall higher. Returns null when undefined.

    Args:
        target: Column with class labels.
        prob: Column with predicted probabilities.
        beta: Weight of recall relative to precision.
        threshold: Decision threshold (predict positive if prob >= threshold).
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
    """
    tp, fp, fn, _tn = _confusion_components(target, prob, threshold, weight, pos_label)
    beta_sq = beta**2
    denom = (1 + beta_sq) * tp + beta_sq * fn + fp
    result = pl.when(denom == 0).then(None).otherwise((1 + beta_sq) * tp / denom)
    alias = f"fbeta_{beta:g}_{target}_{prob}_{threshold:g}"
    if weight is not None:
        alias += f"_{weight}"
    if pos_label != 1:
        alias += f"_pos{pos_label}"
    return result.alias(alias)


def matthews_corrcoef(
    target: str,
    prob: str,
    threshold: float = 0.5,
    weight: str | None = None,
    pos_label: _PosLabel = 1,
) -> pl.Expr:
    """Compute Matthews correlation coefficient at a decision threshold.

    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)).
    Returns null when any marginal total is zero. Range: [-1, 1].

    Args:
        target: Column with class labels.
        prob: Column with predicted probabilities.
        threshold: Decision threshold (predict positive if prob >= threshold).
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
    """
    tp, fp, fn, tn = _confusion_components(target, prob, threshold, weight, pos_label)
    numerator = tp * tn - fp * fn
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    undefined = ((tp + fp) == 0) | ((tp + fn) == 0) | ((tn + fp) == 0) | ((tn + fn) == 0)
    result = pl.when(undefined).then(None).otherwise(numerator / denom_sq.sqrt())
    return result.alias(_alias("mcc", target, prob, threshold, weight, pos_label))


def cohens_kappa(
    target: str,
    prob: str,
    threshold: float = 0.5,
    weight: str | None = None,
    pos_label: _PosLabel = 1,
) -> pl.Expr:
    """Compute Cohen's kappa at a decision threshold.

    Kappa = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and
    p_e is expected agreement by chance. Returns null when undefined.

    Args:
        target: Column with class labels.
        prob: Column with predicted probabilities.
        threshold: Decision threshold (predict positive if prob >= threshold).
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
    """
    tp, fp, fn, tn = _confusion_components(target, prob, threshold, weight, pos_label)
    # Compact 2x2 form, algebraically identical to (p_o - p_e) / (1 - p_e) but a
    # far shallower expression tree (avoids polars' >512-element depth warning on
    # the nested p_o/p_e formulation). Since denom == total**2 * (1 - p_e),
    # denom == 0 captures both undefined cases: empty input (total == 0) and
    # perfect expected agreement (p_e == 1).
    numerator = 2 * (tp * tn - fp * fn)
    denom = (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)
    result = pl.when(denom == 0).then(None).otherwise(numerator / denom)
    return result.alias(_alias("cohens_kappa", target, prob, threshold, weight, pos_label))


def threshold_sweep(
    metric_fn: _MetricFn,
    target: str,
    prob: str,
    thresholds: list[float],
    weight: str | None = None,
    pos_label: _PosLabel = 1,
) -> list[pl.Expr]:
    """Generate metric expressions across multiple thresholds.

    Convenience function for sweeping a metric across thresholds.
    Each threshold produces a separate expression with a unique alias.

    Args:
        metric_fn: One of precision, recall, f1_score, accuracy, balanced_accuracy.
        target: Column with class labels.
        prob: Column with predicted probabilities.
        thresholds: List of decision thresholds to evaluate.
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).

    Returns:
        List of Polars expressions, one per threshold.

    Examples:
        >>> import polars as pl
        >>> from polarbear import f1_score, threshold_sweep
        >>>
        >>> df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.4, 0.6, 0.9]})
        >>> df.select(*threshold_sweep(f1_score, "label", "prob", [0.3, 0.5, 0.7]))
    """
    return [
        metric_fn(target, prob, threshold=t, weight=weight, pos_label=pos_label) for t in thresholds
    ]


def percentile_thresholds(series: pl.Series, percentiles: list[float]) -> list[float]:
    """Compute threshold values from percentiles of a score distribution.

    Args:
        series: A Polars Series of prediction scores/probabilities.
        percentiles: List of percentile values (0-100).

    Returns:
        List of threshold values corresponding to the given percentiles.

    Examples:
        >>> import polars as pl
        >>> from polarbear import percentile_thresholds
        >>>
        >>> scores = pl.Series([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
        >>> percentile_thresholds(scores, [25, 50, 75])
    """
    thresholds: list[float] = []
    for p in percentiles:
        q = series.quantile(p / 100, interpolation="linear")
        if q is None:
            raise ValueError("Cannot compute percentile thresholds on an empty series.")
        thresholds.append(float(q))
    return thresholds
