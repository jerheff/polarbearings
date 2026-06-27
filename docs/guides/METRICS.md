# Metrics Reference

The complete catalogue of polarbearings metrics, with per-metric semantics,
edge cases, and scikit-learn correspondences. For a quick overview and one
example per family, see the [README](../../README.md).

Cross-cutting behaviour shared by every metric — **sample weights**, **custom
positive class** (`pos_label`), and **missing-value handling** — is documented
once at the [end of this file](#cross-cutting-behaviour), not repeated per metric.

## Contents

- [Ranking metrics](#ranking-metrics) — ROC AUC, average precision, Gini, NDCG/DCG
- [Probabilistic metrics](#probabilistic-metrics) — log loss, Brier score
- [Classification metrics](#classification-metrics-threshold-based) — threshold metrics, confusion matrix, curves, threshold sweep
- [Regression metrics](#regression-metrics)
- [Calibration](#calibration)
- [Class weights](#class-weights)
- [Confidence intervals (bootstrap)](#confidence-intervals-bootstrap)
- [Data splitting](#data-splitting-deterministic-id-keyed)
- [Cross-cutting behaviour](#cross-cutting-behaviour) — weights, `pos_label`, missing values

## Ranking Metrics

### ROC AUC

Receiver Operating Characteristic Area Under the Curve for binary classification.

```python
from polarbearings import roc_auc

df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9]})
df.select(roc_auc("label", "score"))  # Returns: 1.0
```

- Uses the Mann-Whitney U statistic for correct tie handling
- Matches scikit-learn's `roc_auc_score` exactly

### Average Precision

Non-interpolated average precision score for binary classification.

```python
from polarbearings import average_precision

df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.4, 0.35, 0.8]})
df.select(average_precision("label", "score"))
```

- Matches scikit-learn's `average_precision_score`
- Handles tied scores correctly

### Gini Coefficient

Normalized Gini coefficient for ranking non-negative targets (e.g. fraud losses).

```python
from polarbearings import gini_coefficient

df = pl.DataFrame({"loss": [1.0, 2.0, 3.0, 4.0], "score": [1.0, 2.0, 3.0, 4.0]})
df.select(gini_coefficient("loss", "score"))  # Returns: 1.0 for perfect ordering

# Binary target via pos_label -> 2*AUC - 1; works for string/categorical labels too:
df2 = pl.DataFrame({"y": ["fraud", "ok", "fraud", "ok"], "score": [0.9, 0.1, 0.8, 0.2]})
df2.select(gini_coefficient("y", "score", pos_label="fraud"))
```

- Returns values between ``-1.0`` and ``1.0``.
- ``1.0`` means the score ordering is optimal for the observed target distribution.
- ``0.0`` means the score is no better than random.
- Supports optional sample weights. **Caveat:** the binary ``pos_label`` identity
  ``Gini = 2·AUC − 1`` holds only for *unweighted* data — the weighted normalized
  Gini uses a per-unit-weight perfect-ordering baseline, so weighted binary Gini is
  **not** ``2·weighted_AUC − 1``. Use `roc_auc` for a weighted AUC.
- ``pos_label`` (default ``None``) maps a class label (``target == pos_label``) to a
  0/1 indicator before computing Gini, for binary/string labels; with ``None`` the
  target is used directly as a numeric magnitude (continuous or already 0/1).

### NDCG / DCG

Ranking quality for graded relevance, on **long-format** data: one row per
(query, document). Evaluate one ranking over the whole frame, or many at once with
`group_by(query).agg(...)`.

```python
from polarbearings import dcg_score, ndcg_score

df = pl.DataFrame({"relevance": [3, 2, 3, 0, 1], "score": [3.0, 2.2, 3.5, 0.1, 1.0]})
df.select(ndcg_score("relevance", "score"))          # normalized, in [0, 1]
df.select(dcg_score("relevance", "score", k=3))      # raw DCG, top-3 only

# One NDCG per query in a single pass:
events.group_by("query_id").agg(ndcg_score("relevance", "score"))
```

- `gain / log_base(rank + 2)` discounting; `k` truncates to the top-k, `log_base`
  defaults to 2.
- Matches scikit-learn's `dcg_score` / `ndcg_score` with `ignore_ties=True` **for
  distinct scores**. Under *tied* scores, ties are broken by row order (not
  gain-averaged), so the result is order-dependent and diverges from scikit-learn;
  sort or break ties upstream if you need reproducibility under ties.
- `ndcg_score` returns `null` when every document is irrelevant (ideal DCG is 0).

## Probabilistic Metrics

### Log Loss (Binary Cross-Entropy)

```python
from polarbearings import log_loss

df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
df.select(log_loss("label", "prob"))
```

### Brier Score

```python
from polarbearings import brier_score

df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
df.select(brier_score("label", "prob"))
```

## Classification Metrics (Threshold-Based)

All classification metrics accept an optional `threshold` parameter (default 0.5).

```python
from polarbearings import precision, recall, f1_score, fbeta_score, specificity
from polarbearings import accuracy, balanced_accuracy, matthews_corrcoef, cohens_kappa
from polarbearings import jaccard_score

df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.4, 0.6, 0.9]})

df.select(
    precision("label", "prob"),
    recall("label", "prob"),
    f1_score("label", "prob"),
    specificity("label", "prob"),
    accuracy("label", "prob"),
    balanced_accuracy("label", "prob"),
    matthews_corrcoef("label", "prob"),
    cohens_kappa("label", "prob"),
    jaccard_score("label", "prob"),
)

# F-beta with custom beta (0.5 weights precision higher, 2.0 weights recall higher)
df.select(fbeta_score("label", "prob", beta=2.0))

# Custom threshold
df.select(precision("label", "prob", threshold=0.7))
```

**Metric reference:**

| Metric | Formula | Returns `null` when |
|---|---|---|
| `precision` | TP / (TP + FP) | No positive predictions |
| `recall` | TP / (TP + FN) | No actual positives |
| `specificity` | TN / (TN + FP) | No actual negatives |
| `f1_score` | 2·TP / (2·TP + FP + FN) | Undefined denominator |
| `fbeta_score` | (1+β²)·TP / ((1+β²)·TP + β²·FN + FP) | Undefined denominator |
| `accuracy` | (TP + TN) / total | Empty data |
| `balanced_accuracy` | (TPR + TNR) / 2 | Either class absent |
| `matthews_corrcoef` | (TP·TN − FP·FN) / √(...) | Any marginal total is zero |
| `cohens_kappa` | (p_o − p_e) / (1 − p_e) | All predictions one class |
| `jaccard_score` | TP / (TP + FP + FN) | Empty union (no positives predicted or actual) |

> `fbeta_score` takes `beta` as a required keyword argument (e.g. `fbeta_score("label", "prob", beta=2.0)`). `balanced_accuracy` and `matthews_corrcoef` are more robust to class imbalance than `accuracy`.

### Confusion Matrix

`confusion_matrix` returns the four cells every threshold metric is built from as
a single struct — read them all in one pass, or derive any custom rate yourself.
The cell fields are `Int64` counts (or `Float64` summed weights when `weight` is
given); a leading `threshold` field (`Float64`) records the decision threshold the
cells were computed at. It honours `weight` and `pos_label` like every other metric:

```python
from polarbearings import confusion_matrix

df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.2, 0.8, 0.6, 0.9]})

# One struct column {threshold, tp, fp, fn, tn}; unnest to spread into columns.
df.select(confusion_matrix("label", "score")).unnest("confusion_matrix_label_score_0.5")
# ┌───────────┬─────┬─────┬─────┬─────┐
# │ threshold ┆ tp  ┆ fp  ┆ fn  ┆ tn  │
# │    0.5    ┆  2  ┆  1  ┆  0  ┆  1  │
# └───────────┴─────┴─────┴─────┴─────┘

# Composes inside group_by for a per-segment confusion matrix in one pass:
df.group_by("segment").agg(confusion_matrix("label", "score").alias("cm")).unnest("cm")
```

Because each struct carries its own `threshold`, sweeping it across thresholds and
reshaping to a tidy frame is one pass — every row is self-describing:

```python
from polarbearings import threshold_sweep, quantiles

# wide (1 row, one struct column per threshold) -> tidy (one row per threshold)
df.select(pl.concat_list(threshold_sweep(confusion_matrix, "label", "score", quantiles(100))).alias("cm")) \
  .explode("cm").unnest("cm")
# -> columns: threshold, tp, fp, fn, tn  (then tpr/fpr/precision/recall are column math)
```

### Diagnostic curves: ROC, PR, DET, cost

One call each, returning tidy, plot-ready `LazyFrame`s:

```python
from polarbearings import roc_curve, pr_curve, det_curve, expected_cost

roc_curve(df, "label", "score").collect()        # -> threshold, fpr, tpr
pr_curve(df, "label", "score").collect()         # -> threshold, precision, recall
det_curve(df, "label", "score").collect()        # -> threshold, fpr, fnr
expected_cost(df, "label", "score", {"fp": 1.0, "fn": 5.0}).collect()  # -> threshold, cost
```

All four take `weight=`, `pos_label=`, `by=` (a separate curve per segment in one
pass), and `thresholds=` (below). They are thin column-math wrappers over
`confusion_curve`.

### `confusion_curve` — the primitive

`confusion_curve` gives the confusion cells `threshold, tp, fp, fn, tn`. By default
it reports **every distinct score** in one sorted pass (`O(n log n)`) — the exact
step function (matches scikit-learn's `roc_curve`), scaling to millions of rows.
Pass `thresholds=` for a fixed **grid** of comparable operating points instead — an
`int` (that many score quantiles), a spec (`quantiles(n)`, `equal_width(n)`,
`linspace(n)`), or a `list[float]`:

```python
from polarbearings import confusion_curve

confusion_curve(df, "label", "score").collect()                  # exact, every distinct score
confusion_curve(df, "label", "score", thresholds=20).collect()   # 20 quantile operating points
confusion_curve(df, "label", "score", by="segment").collect()    # a separate curve per segment
# -> threshold, tp, fp, fn, tn   (then fpr = fp/(fp+tn), tpr = tp/(tp+fn), ... are column math)
```

- Takes a `DataFrame` **or** `LazyFrame` and returns a `LazyFrame` — fully lazy and
  single-pass either way; `.collect()` to materialize, or compose in a larger query.
- `endpoints=True` (default) prepends a `threshold = +inf` row so a derived curve
  starts at the origin; `weight` and `pos_label` behave like every other metric.

The `thresholds=` argument is forwarded by every curve wrapper, so `roc_curve(df,
"label", "score", thresholds=20)` gives a 20-point grid ROC just the same.

### Threshold Sweep

Compute any classification metric across many thresholds in a **single pass**.
`thresholds` accepts a fixed list, or a **threshold spec** — `quantiles(n)`,
`equal_width(n)`, or `linspace(n)` — resolved *inside the query graph*: quantile
and equal-width cut points are computed in-engine, so under `group_by` each group
is swept at **its own** thresholds. The default is `quantiles(100)`.

```python
from polarbearings import f1_score, threshold_sweep, quantiles

df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.4, 0.6, 0.9]})

df.select(*threshold_sweep(f1_score, "label", "prob", [0.3, 0.5, 0.7]))  # fixed values
df.select(*threshold_sweep(f1_score, "label", "prob", quantiles(10)))     # 10 quantile cuts
df.select(*threshold_sweep(f1_score, "label", "prob"))                    # default quantiles(100)

# Per-segment thresholds in one pass — each group uses its own score quantiles:
df.group_by("segment").agg(*threshold_sweep(f1_score, "label", "prob", quantiles(20)))
```

Metrics also accept a Polars **expression** as the threshold directly, e.g.
`precision("label", "prob", threshold=pl.col("prob").quantile(0.9))`.

### Percentile Thresholds

`percentile_thresholds` materializes concrete threshold values from a score
series (eager) — complements the in-graph `quantiles` spec when you need the
numbers themselves:

```python
from polarbearings import f1_score, percentile_thresholds, threshold_sweep

scores = df["prob"]
thresholds = percentile_thresholds(scores, [10, 25, 50, 75, 90])
df.select(*threshold_sweep(f1_score, "label", "prob", thresholds))
```

## Regression Metrics

```python
from polarbearings import (
    d2_absolute_error_score,
    d2_pinball_score,
    d2_tweedie_score,
    explained_variance_score,
    huber_loss,
    log_cosh_loss,
    mae,
    mape,
    max_error,
    mean_gamma_deviance,
    mean_pinball_loss,
    mean_poisson_deviance,
    mean_squared_log_error,
    mean_tweedie_deviance,
    median_absolute_error,
    mse,
    r2_score,
    rmse,
    root_mean_squared_log_error,
    smape,
)

df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "pred": [1.1, 2.2, 2.8, 4.5]})

df.select(
    mae("y", "pred"),
    mse("y", "pred"),
    rmse("y", "pred"),
    r2_score("y", "pred"),
    mape("y", "pred"),
    mean_squared_log_error("y", "pred"),       # MSLE (inputs must be >= 0)
    root_mean_squared_log_error("y", "pred"),  # RMSLE
    max_error("y", "pred"),                     # worst-case |residual|
    median_absolute_error("y", "pred"),         # robust central error
    explained_variance_score("y", "pred"),
    mean_pinball_loss("y", "pred", alpha=0.5),  # quantile loss
    smape("y", "pred"),                          # symmetric MAPE
    huber_loss("y", "pred", delta=1.0),          # robust, MSE/MAE hybrid
    log_cosh_loss("y", "pred"),                  # smooth, numerically stable
    mean_poisson_deviance("y", "pred"),          # Tweedie power=1 (counts)
    mean_gamma_deviance("y", "pred"),            # Tweedie power=2 (positive, skewed)
    mean_tweedie_deviance("y", "pred", power=1.5),
    d2_tweedie_score("y", "pred", power=1),      # "explained deviance" (R²-like)
    d2_absolute_error_score("y", "pred"),        # D² around the median
    d2_pinball_score("y", "pred", alpha=0.5),    # D² around a quantile
)
```

The full regression set, and which accept an optional `weight`:

| Metric | sklearn analog | Weighted? |
| --- | --- | --- |
| `mae`, `mse`, `rmse`, `r2_score`, `mape` | yes | yes |
| `mean_squared_log_error` (MSLE) | `mean_squared_log_error` | yes |
| `root_mean_squared_log_error` (RMSLE) | `root_mean_squared_log_error` | yes |
| `explained_variance_score` | `explained_variance_score` | yes |
| `mean_pinball_loss` (quantile loss, `alpha`) | `mean_pinball_loss` | yes |
| `smape` (symmetric MAPE) | — | yes |
| `huber_loss` (`delta`) | — | yes |
| `log_cosh_loss` | — | yes |
| `mean_tweedie_deviance` (`power`) | `mean_tweedie_deviance` | yes |
| `mean_poisson_deviance` | `mean_poisson_deviance` | yes |
| `mean_gamma_deviance` | `mean_gamma_deviance` | yes |
| `d2_tweedie_score` (`power`) | `d2_tweedie_score` | yes |
| `max_error` | `max_error` | **no** — scaling samples doesn't change the single worst residual, so weighting is undefined (sklearn's `max_error` also takes no `sample_weight`). |
| `median_absolute_error` | `median_absolute_error` | **no** — sklearn's weighted form is a *weighted percentile*, which can't be expressed correctly as one pure Polars expression; shipping a wrong weighted median was avoided. |
| `d2_absolute_error_score` | `d2_absolute_error_score` | **no** — its baseline is the median (a weighted percentile), same constraint as `median_absolute_error`. |
| `d2_pinball_score` (`alpha`) | `d2_pinball_score` | **no** — baseline is the `alpha`-quantile (weighted percentile). |

> **MAPE note:** Rows where `target == 0` are **excluded** (the percentage error is undefined there). This differs from scikit-learn's `mean_absolute_percentage_error`, which keeps those rows using an epsilon floor and can return very large values. All other metrics match scikit-learn.
>
> **MSLE / RMSLE note:** inputs must be non-negative (the log is otherwise undefined). Negative inputs yield NaN rather than raising, mirroring (but not re-raising) scikit-learn's domain error.
>
> **sMAPE note:** uses the `mean(2·|y−p| / (|y|+|p|))` form (range `[0, 2]`); the `0/0` case (both `y` and `p` zero) contributes `0` to avoid division-by-zero blow-up.

## Calibration

`calibration_curve` returns plot-ready reliability-diagram data — one row per
non-empty bin with the mean predicted probability vs. the observed positive
fraction. It mirrors scikit-learn's `calibration_curve` for the `"uniform"` and
`"quantile"` strategies, and additionally accepts **explicit bin edges** (handy
for fixed, comparable score bands across models).

```python
from polarbearings import calibration_curve

calibration_curve(df, "label", "prob", n_bins=10, strategy="quantile").collect()
calibration_curve(df, "label", "prob", bins=[0.0, 0.25, 0.5, 0.75, 1.0]).collect()  # custom edges
calibration_curve(df, "label", "prob", n_bins=10, by="segment").collect()  # a curve per segment
# -> columns: [*by], bin, bin_lower, bin_upper, count, prob_pred, prob_true
```

Like `confusion_curve`, it accepts a `DataFrame` or `LazyFrame` and returns a
`LazyFrame` (call `.collect()` to materialize). `by=` computes one curve per group
in a single pass, with **shared bins** across groups so the segments stay directly
comparable.

For the scalar summaries, `expected_calibration_error` (ECE) and
`maximum_calibration_error` (MCE) are plain **expressions** — the count-weighted
average and the worst bin's `|mean predicted − observed|` gap — so they drop into
`select` and `group_by` next to any other metric:

```python
from polarbearings import expected_calibration_error, maximum_calibration_error

df.select(
    ece=expected_calibration_error("label", "prob", n_bins=10),
    mce=maximum_calibration_error("label", "prob", n_bins=10),
)
df.group_by("segment").agg(expected_calibration_error("label", "prob"))  # ECE per segment
```

They share `n_bins` / `strategy` / `bins` / `weight` / `pos_label` with
`calibration_curve`; 0 is perfectly calibrated.

## Class Weights

`balanced_sample_weight` produces per-row weights inversely proportional to class
frequency (`n_samples / (n_classes · count)`), matching
`sklearn.utils.class_weight.compute_sample_weight("balanced", y)` — feed them
straight into any weighted metric. `balanced_class_weights(series)` returns the
per-class `{label: weight}` mapping (mirrors `compute_class_weight`).

```python
from polarbearings import balanced_sample_weight, roc_auc

weighted = df.with_columns(balanced_sample_weight("label").alias("w"))
weighted.select(roc_auc("label", "score", weight="w"))
```

## Confidence intervals (bootstrap)

Because nearly every metric accepts a `weight`, a bootstrap replicate is just the metric
under random weights — `polarbearings` uses the **Bayesian bootstrap**, generated
in-engine (no Python resampling loop). `bootstrap_ci` returns `{estimate, low,
high}` for the whole frame, or one row per group with `by=`:

```python
from polarbearings import bootstrap_ci, roc_auc

bootstrap_ci(df, roc_auc, "label", "score", n_resamples=1000, method="bc")
bootstrap_ci(df, roc_auc, "label", "score", by="segment")   # one CI per segment
```

`bootstrap_weight` exposes a single replicate as a weight **expression**, so it
composes with *any* metric or curve — e.g. a bootstrapped ROC band is `roc_curve`
under many replicate weights. Hashing a stable id column makes it reproducible
across runs and safe inside `group_by`:

```python
from polarbearings import bootstrap_weight, roc_curve

boot = df.with_row_index("id")
roc_curve(boot, "label", "score", weight=bootstrap_weight("id", seed=0), thresholds=50)
```

It supports `kind="bayesian"` (default, `Exp(1)`) or `kind="poisson"` integer
multiplicities, and `weight_kind="frequency"` for de-duplicated case counts (a row
for `w` cases draws `Gamma(w, 1)`, not `w · Exp(1)` — correct variance, not
over-dispersed).

## Data splitting (deterministic, id-keyed)

Hashing a stable record id to a uniform gives split assignments that are
**reproducible** across runs and row orderings and **leak nothing** — the same id
always lands in the same split. All are plain expressions (drop into
`with_columns`, `group_by`, lazy):

`seed` is the **first, required** argument — it *is* the split's identity, and
independent splits must use different seeds (sharing one correlates them, since the
assignment is a pure function of `id` and `seed`):

```python
from polarbearings import hash_split, hash_fold, hash_splits

df.with_columns(holdout=hash_split(1, "id", fraction=0.2))  # bool: ~20% holdout
df.with_columns(fold=hash_fold(0, "id", k=5))               # CV fold id 0..4
df.with_columns(                                            # named multi-way
    split=hash_splits(2, "id", [("test", 0.15), ("val", 0.15)], remainder="train")
)
```

`hash_split` is **consistent**: growing `fraction` only *adds* rows to the holdout
(no churn). `hash_splits` gives each split its own seed and a residual-conditional
fraction, so resizing one split leaves the **upstream** splits' membership unchanged
— unlike cumulative-threshold schemes that reshuffle neighbours. Order is priority;
put the long-lived holdout first.

**Stratification** is free in expectation (the hash ignores labels, so each class is
sampled at `~fraction`). For an *exact* per-class split, rank within the stratum on
the shared uniform:

```python
from polarbearings import hash_uniform

u = hash_uniform(1, "id")
df.with_columns(holdout=u.rank("ordinal").over("class") <= (0.2 * pl.len()).over("class"))
```

## Cross-cutting behaviour

These three behaviours are shared by the metrics above rather than specific to any one.

### Sample weights

*Nearly every* metric supports optional sample weights via a `weight` column —
including ones that are awkward or unsupported elsewhere, like ROC AUC, log loss,
MCC, and Cohen's kappa. Six metrics omit it — `dcg_score`, `ndcg_score`,
`max_error`, `median_absolute_error`, `d2_absolute_error_score`, and
`d2_pinball_score` — where a weighted form is undefined, non-standard, or not
cleanly expressible as one Polars expression (the rationale is in the ranking and
regression sections above).

```python
df.select(roc_auc("label", "score", weight="sample_weight"))
df.select(matthews_corrcoef("label", "prob", weight="w"))
df.select(mae("y", "pred", weight="w"))
```

### Custom positive class

The positive class defaults to `1`, but `pos_label` lets it be any value —
integers, strings, or booleans — with no need to remap your labels to `0`/`1`:

```python
# String labels
df = pl.DataFrame({"outcome": ["cancer", "healthy", "cancer"], "p": [0.9, 0.2, 0.7]})
df.select(precision("outcome", "p", pos_label="cancer"))

df.select(roc_auc("y", "score", pos_label=100))     # integer labels {100, 200}
df.select(f1_score("flag", "p", pos_label=True))     # boolean labels
```

Supported by the classification and binary metrics (ROC AUC, average precision,
log loss, Brier score, precision/recall/F1/F-beta, accuracy, balanced accuracy,
specificity, MCC, Cohen's kappa). `gini_coefficient` also takes an optional
`pos_label` (to treat its target as a binary label instead of a magnitude).
Regression metrics take continuous targets, so they don't have `pos_label`.

### Missing values

Missing data is treated **loudly**: if any `target`, score, or `weight` value is
`null` **or** `NaN`, a metric returns `null` rather than silently dropping rows or
miscounting. Detection happens on the raw columns (before any `pos_label`/threshold
comparison), and is **scoped to the evaluation context** — the whole frame in a
`select`, or just the affected group under `group_by().agg()`:

```python
df = pl.DataFrame({"y": [0, 1, 1], "p": [0.2, None, 0.8]})
df.select(precision("y", "p"))            # -> null (one missing score)

g.group_by("seg").agg(roc_auc("y", "s"))  # only segments with a missing value are null
```

`null` and `NaN` are treated identically (both "missing"), matching what scikit-learn
does the moment a Polars column with nulls crosses into NumPy. For **complete-case**
behavior instead, drop missing rows yourself first — `df.drop_nulls([...])` (and
`drop_nans` for float columns). The **curve helpers are the one exception**:
`confusion_curve` and `calibration_curve` *drop* incomplete rows and compute on the
rest, since a curve is a set of operating points rather than one number.
