"""Threshold-based classification metrics implemented as Polars expressions."""

from typing import Final, Protocol

import polars as pl

from polarbearings._common import (
    IntoExpr,
    PosLabel,
    ThresholdValue,
    WeightInput,
    any_missing,
    col_expr,
    col_name,
    resolve_weight,
    weight_suffix,
)
from polarbearings.thresholds import ThresholdsLike, quantiles, resolve_thresholds


class _MetricFn(Protocol):
    def __call__(
        self,
        target: IntoExpr,
        prob: IntoExpr,
        *,
        threshold: ThresholdValue = ...,
        weight: WeightInput = ...,
        pos_label: PosLabel = ...,
    ) -> pl.Expr: ...


def _threshold_token(threshold: ThresholdValue) -> str:
    """Alias token for a threshold — its value for floats, a placeholder for exprs.

    Expression thresholds (e.g. a data-derived quantile) have no static value, so
    they get a fixed token; ``threshold_sweep`` overrides the alias with the spec's
    own label.
    """
    return "expr" if isinstance(threshold, pl.Expr) else f"{threshold:g}"


def _pos_suffix(pos_label: PosLabel) -> str:
    """Alias suffix for a non-default positive class (empty for the default 1)."""
    return f"_pos{pos_label}" if pos_label != 1 else ""


def _confusion_components(
    target: IntoExpr,
    prob: IntoExpr,
    threshold: ThresholdValue,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """Compute TP, FP, FN, TN as Polars expressions.

    The positive class is any row where ``target == pos_label``; every other
    value is treated as negative (one-vs-rest).
    """
    # Keep the masks boolean and count them, rather than casting to Float64 and
    # summing four full-length products. Counting bitmasks touches far less
    # memory and skips a float multiply per row, so the confusion cells (which
    # feed every threshold metric) are several times faster at scale — without
    # changing the result (verified to exact / float-rounding equivalence).
    is_pos = col_expr(target) == pos_label
    predicted = col_expr(prob) >= threshold

    w = resolve_weight(weight)
    if w is not None:
        # filter().sum() over an empty selection returns 0.0 (not null), which
        # matches the all-one-class degenerate semantics of the product form.
        tp = w.filter(is_pos & predicted).sum()
        fp = w.filter(~is_pos & predicted).sum()
        fn = w.filter(is_pos & ~predicted).sum()
        tn = w.filter(~is_pos & ~predicted).sum()
    else:
        tp = (is_pos & predicted).sum().cast(pl.Float64)
        fp = (~is_pos & predicted).sum().cast(pl.Float64)
        fn = (is_pos & ~predicted).sum().cast(pl.Float64)
        tn = (~is_pos & ~predicted).sum().cast(pl.Float64)

    # Null every cell — so every derived metric is null — when any input is missing
    # in the current context (frame or group). Detected on the raw prob/target/weight
    # before the >= / == comparisons can launder a NaN into a real prediction/class.
    miss = any_missing(values=[prob], labels=[target], weight=weight)
    tp, fp, fn, tn = (pl.when(miss).then(None).otherwise(c) for c in (tp, fp, fn, tn))
    return tp, fp, fn, tn


def _alias(
    name: str,
    target: IntoExpr,
    prob: IntoExpr,
    threshold: ThresholdValue,
    weight: WeightInput,
    pos_label: PosLabel = 1,
) -> str:
    """Build a consistent alias string.

    A ``_pos{pos_label}`` suffix is appended only when ``pos_label`` is not the
    default ``1``, so existing alias strings are unchanged for the common case.
    """
    token = _threshold_token(threshold)
    tname, pname = col_name(target), col_name(prob)
    return f"{name}_{tname}_{pname}_{token}{weight_suffix(weight)}{_pos_suffix(pos_label)}"


def confusion_matrix(
    target: IntoExpr,
    prob: IntoExpr,
    *,
    threshold: ThresholdValue = 0.5,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
) -> pl.Expr:
    """Compute the binary confusion matrix at a decision threshold as a struct.

    Returns a single struct value with fields ``threshold``, ``tp``, ``fp``,
    ``fn``, ``tn`` — the building block every other threshold metric is derived
    from. Exposing it directly lets you read all four cells (or compute custom
    rates) in one pass, and it composes inside ``group_by().agg(...)`` for a
    per-segment confusion matrix. Unlike fixed-label implementations, the cells
    honour both sample weights and an arbitrary positive class.

    The leading ``threshold`` field is the decision threshold the cells were
    computed at (its actual value, even when ``threshold`` is a data-derived
    expression such as a quantile). It makes a swept set of structs
    self-describing: ``pl.concat_list(...).explode(...).unnest(...)`` yields a tidy
    frame with the threshold already attached to each row (see ``threshold_sweep``).

    The cell fields are ``Int64`` counts for unweighted data and ``Float64`` summed
    weights when ``weight`` is given; ``threshold`` is always ``Float64``.

    Args:
        target: Column with class labels.
        prob: Column with predicted probabilities (or scores).
        threshold: Decision threshold (predict positive if prob >= threshold).
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).

    Returns:
        A Polars expression yielding a struct ``{threshold, tp, fp, fn, tn}``.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import confusion_matrix
        >>>
        >>> df = pl.DataFrame({
        ...     "label": [0, 0, 1, 1],
        ...     "score": [0.2, 0.8, 0.6, 0.9],
        ... })
        >>> df.select(confusion_matrix("label", "score")).unnest("confusion_matrix_label_score_0.5")
        shape: (1, 5)
        ┌───────────┬─────┬─────┬─────┬─────┐
        │ threshold ┆ tp  ┆ fp  ┆ fn  ┆ tn  │
        │ ---       ┆ --- ┆ --- ┆ --- ┆ --- │
        │ f64       ┆ i64 ┆ i64 ┆ i64 ┆ i64 │
        ╞═══════════╪═════╪═════╪═════╪═════╡
        │ 0.5       ┆ 2   ┆ 1   ┆ 0   ┆ 1   │
        └───────────┴─────┴─────┴─────┴─────┘
    """
    tp, fp, fn, tn = _confusion_components(target, prob, threshold, weight, pos_label)
    if weight is None:
        tp, fp, fn, tn = (cell.cast(pl.Int64) for cell in (tp, fp, fn, tn))
    # Carry the decision threshold (its actual value, even when it's a data-derived
    # expression such as a quantile) as the struct's first field, so a swept set of
    # structs is self-describing: concat_list(...).explode().unnest() yields a tidy
    # frame with the threshold already attached to each row.
    thr = (threshold if isinstance(threshold, pl.Expr) else pl.lit(threshold)).cast(pl.Float64)
    cells = pl.struct(
        thr.alias("threshold"),
        tp.alias("tp"),
        fp.alias("fp"),
        fn.alias("fn"),
        tn.alias("tn"),
    )
    return cells.alias(_alias("confusion_matrix", target, prob, threshold, weight, pos_label))


def precision(
    target: IntoExpr,
    prob: IntoExpr,
    *,
    threshold: ThresholdValue = 0.5,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
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
    target: IntoExpr,
    prob: IntoExpr,
    *,
    threshold: ThresholdValue = 0.5,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
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
    target: IntoExpr,
    prob: IntoExpr,
    *,
    threshold: ThresholdValue = 0.5,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
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


def jaccard_score(
    target: IntoExpr,
    prob: IntoExpr,
    *,
    threshold: ThresholdValue = 0.5,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
) -> pl.Expr:
    """Compute the Jaccard index (intersection over union) at a decision threshold.

    Jaccard = TP / (TP + FP + FN), the size of the intersection of predicted and
    actual positives over the size of their union. Returns null in the degenerate
    case where TP + FP + FN == 0 (no sample is predicted or labelled positive).

    Note:
        This null convention follows this package's precision/recall behaviour and
        diverges from ``sklearn.metrics.jaccard_score``, which returns ``0.0`` (and
        emits an ``UndefinedMetricWarning``) for that degenerate case.

    Args:
        target: Column with class labels.
        prob: Column with predicted probabilities.
        threshold: Decision threshold (predict positive if prob >= threshold).
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).
    """
    tp, fp, fn, _tn = _confusion_components(target, prob, threshold, weight, pos_label)
    denom = tp + fp + fn
    result = pl.when(denom == 0).then(None).otherwise(tp / denom)
    return result.alias(_alias("jaccard", target, prob, threshold, weight, pos_label))


def accuracy(
    target: IntoExpr,
    prob: IntoExpr,
    *,
    threshold: ThresholdValue = 0.5,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
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
    target: IntoExpr,
    prob: IntoExpr,
    *,
    threshold: ThresholdValue = 0.5,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
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
    target: IntoExpr,
    prob: IntoExpr,
    *,
    threshold: ThresholdValue = 0.5,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
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
    target: IntoExpr,
    prob: IntoExpr,
    *,
    beta: float,
    threshold: ThresholdValue = 0.5,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
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
    return result.alias(_alias(f"fbeta_{beta:g}", target, prob, threshold, weight, pos_label))


def matthews_corrcoef(
    target: IntoExpr,
    prob: IntoExpr,
    *,
    threshold: ThresholdValue = 0.5,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
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
    target: IntoExpr,
    prob: IntoExpr,
    *,
    threshold: ThresholdValue = 0.5,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
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


# Probe threshold used only to learn a metric's alias shape; never emitted.
_ALIAS_PROBE: Final = 0.123457


def threshold_sweep(
    metric_fn: _MetricFn,
    target: IntoExpr,
    prob: IntoExpr,
    thresholds: ThresholdsLike | None = None,
    *,
    weight: WeightInput = None,
    pos_label: PosLabel = 1,
) -> list[pl.Expr]:
    """Generate metric expressions across many thresholds in a single pass.

    ``thresholds`` is an ``int`` ``N`` (shorthand for ``quantiles(N)``), a list of
    fixed values, or a *threshold spec* such as
    :func:`~polarbearings.thresholds.quantiles` or
    :func:`~polarbearings.thresholds.equal_width`. An int or a spec resolves against
    the ``prob`` column inside the query graph, so quantile / equal-width thresholds
    are computed in-engine — and under ``group_by().agg(...)`` each group is swept
    at its own thresholds. A plain ``list[float]`` keeps the original column names.

    Args:
        metric_fn: A threshold metric — precision, recall, f1_score, accuracy,
            balanced_accuracy, specificity, matthews_corrcoef, cohens_kappa,
            jaccard_score, or confusion_matrix.
        target: Column with class labels.
        prob: Column with predicted probabilities.
        thresholds: An int ``N`` (``quantiles(N)``), fixed thresholds, or a spec.
            Defaults to ``quantiles(100)``.
        weight: Optional column with sample weights.
        pos_label: Value in ``target`` treated as the positive class (default 1).

    Returns:
        List of Polars expressions, one per threshold, each uniquely aliased.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import f1_score, threshold_sweep
        >>> from polarbearings.thresholds import quantiles
        >>>
        >>> df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.4, 0.6, 0.9]})
        >>> df.select(*threshold_sweep(f1_score, "label", "prob", [0.3, 0.5, 0.7]))
        shape: (1, 3)
        ┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
        │ f1_score_label_prob_0.3 ┆ f1_score_label_prob_0.5 ┆ f1_score_label_prob_0.7 │
        │ ---                     ┆ ---                     ┆ ---                     │
        │ f64                     ┆ f64                     ┆ f64                     │
        ╞═════════════════════════╪═════════════════════════╪═════════════════════════╡
        │ 0.8                     ┆ 1.0                     ┆ 0.666667                │
        └─────────────────────────┴─────────────────────────┴─────────────────────────┘
        >>> # `thresholds` also accepts a spec, e.g. decile cut points:
        >>> _ = df.select(*threshold_sweep(f1_score, "label", "prob", quantiles(10)))
    """
    spec: ThresholdsLike = quantiles(100) if thresholds is None else thresholds
    resolved = resolve_thresholds(spec, prob)

    # Learn this metric's alias shape from a probe call so every threshold gets a
    # unique, metric-named column even when its value is an expression (which has no
    # static token). The stripped tail is exactly what _alias appends after the
    # "{name}_{target}_{prob}" prefix, so the float path reproduces the old names.
    suffix = weight_suffix(weight) + _pos_suffix(pos_label)
    tail = f"_{_threshold_token(_ALIAS_PROBE)}{suffix}"
    probe_name = metric_fn(
        target, prob, threshold=_ALIAS_PROBE, weight=weight, pos_label=pos_label
    ).meta.output_name()
    prefix = probe_name.removesuffix(tail)

    exprs: list[pl.Expr] = []
    for label, value in resolved:
        expr = metric_fn(target, prob, threshold=value, weight=weight, pos_label=pos_label)
        exprs.append(expr.alias(f"{prefix}_{label}{suffix}"))
    return exprs
