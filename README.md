# Polarbear =;D

High-performance machine learning metrics implemented as native Polars expressions.

## Features

- **Fast where it counts**: Native Polars expressions — large wins on grouped and probabilistic metrics (see [Performance](#performance))
- **Weighted**: Sample weights on *every* metric
- **Flexible labels**: Any positive class — `1`, `100`, `"cancer"`, `True` — via `pos_label`
- **Correct**: scikit-learn-faithful, verified with property-based testing
- **Composable**: Plain Polars expressions — drop into `group_by`, `over`, and lazy pipelines
- **Type-safe**: Full type hints, checked with [ty](https://github.com/astral-sh/ty)

## Installation

> **Note:** Not yet published to PyPI. Install from source until the first release.

```bash
pip install git+https://github.com/jerheff/polarbear.git
```

Or, once published:

```bash
pip install polarbear   # or: uv add polarbear
```

## Quick Start

```python
import polars as pl
from polarbear import roc_auc

# Create your data
df = pl.DataFrame({
    "actual": [0, 0, 1, 1, 1],
    "predicted_score": [0.1, 0.4, 0.35, 0.8, 0.9]
})

# Calculate ROC AUC as a Polars expression
result = df.select(roc_auc("actual", "predicted_score"))
print(result)
# shape: (1, 1)
# ┌──────────────────────────────────┐
# │ roc_auc_actual_predicted_score   │
# │ ---                              │
# │ f64                              │
# ╞══════════════════════════════════╡
# │ 0.833333                         │
# └──────────────────────────────────┘
```

## Where Polarbear Excels

Every metric is a *pure Polars expression*. That design gives polarbear
strengths a scikit-learn wrapper or a compiled plugin can't easily match:

- **Group-wise metrics at scale** — metrics drop straight into `group_by().agg()`
  and Polars parallelizes across groups (15–65x faster than a Python loop calling
  scikit-learn per segment).
- **Sample weights almost everywhere** — nearly all 27 metrics accept an optional
  `weight` column, including ROC AUC, log loss, MCC, and Cohen's kappa. (The two
  exceptions are `max_error` and `median_absolute_error`, where a weighted form is
  undefined or not cleanly expressible — see the regression section.)
- **Any positive class** — `pos_label` accepts `1`, `100`, `"cancer"`, `True`, …
  no remapping your labels to 0/1.
- **scikit-learn-faithful** — names and edge-case semantics mirror scikit-learn
  (e.g. `null` for undefined cases), so migration is ~1:1.
- **Composable & lazy-friendly** — pure expressions fold into lazy query plans,
  `over()` windows, and the rest of your Polars pipeline.
- **Zero build, one dependency** — pure Python emitting Polars expressions; no
  compiled extension, installs anywhere Polars runs, supports Polars 1.0.0+.

## Available Metrics

### Ranking Metrics

#### ROC AUC

Receiver Operating Characteristic Area Under the Curve for binary classification.

```python
from polarbear import roc_auc

df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9]})
df.select(roc_auc("label", "score"))  # Returns: 1.0
```

- Uses the Mann-Whitney U statistic for correct tie handling
- Matches scikit-learn's `roc_auc_score` exactly

#### Average Precision

Non-interpolated average precision score for binary classification.

```python
from polarbear import average_precision

df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.4, 0.35, 0.8]})
df.select(average_precision("label", "score"))
```

- Matches scikit-learn's `average_precision_score`
- Handles tied scores correctly

#### Gini Coefficient

Normalized Gini coefficient for ranking non-negative targets (e.g. fraud losses).

```python
from polarbear import gini_coefficient

df = pl.DataFrame({"loss": [1.0, 2.0, 3.0, 4.0], "score": [1.0, 2.0, 3.0, 4.0]})
df.select(gini_coefficient("loss", "score"))  # Returns: 1.0 for perfect ordering
```

- Returns values between ``-1.0`` and ``1.0``.
- ``1.0`` means the score ordering is optimal for the observed target distribution.
- ``0.0`` means the score is no better than random.
- Supports optional sample weights.

### Probabilistic Metrics

#### Log Loss (Binary Cross-Entropy)

```python
from polarbear import log_loss

df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
df.select(log_loss("label", "prob"))
```

#### Brier Score

```python
from polarbear import brier_score

df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
df.select(brier_score("label", "prob"))
```

### Classification Metrics (Threshold-Based)

All classification metrics accept an optional `threshold` parameter (default 0.5).

```python
from polarbear import precision, recall, f1_score, fbeta_score, specificity
from polarbear import accuracy, balanced_accuracy, matthews_corrcoef, cohens_kappa

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

> `fbeta_score` takes `beta` as a required positional argument before `threshold`. `balanced_accuracy` and `matthews_corrcoef` are more robust to class imbalance than `accuracy`.

#### Threshold Sweep

Compute any classification metric across multiple thresholds in a single pass:

```python
from polarbear import f1_score, threshold_sweep

df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.4, 0.6, 0.9]})
df.select(*threshold_sweep(f1_score, "label", "prob", [0.3, 0.5, 0.7]))
```

#### Percentile Thresholds

Compute threshold values from percentiles of the score distribution, useful for selecting thresholds relative to model output ranges rather than absolute values:

```python
from polarbear import f1_score, percentile_thresholds, threshold_sweep

scores = df["prob"]
thresholds = percentile_thresholds(scores, [10, 25, 50, 75, 90])
df.select(*threshold_sweep(f1_score, "label", "prob", thresholds))
```

### Regression Metrics

```python
from polarbear import (
    explained_variance_score,
    huber_loss,
    log_cosh_loss,
    mae,
    mape,
    max_error,
    mean_pinball_loss,
    mean_squared_log_error,
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
| `max_error` | `max_error` | **no** — scaling samples doesn't change the single worst residual, so weighting is undefined (sklearn's `max_error` also takes no `sample_weight`). |
| `median_absolute_error` | `median_absolute_error` | **no** — sklearn's weighted form is a *weighted percentile*, which can't be expressed correctly as one pure Polars expression; shipping a wrong weighted median was avoided. |

> **MAPE note:** Rows where `target == 0` are **excluded** (the percentage error is undefined there). This differs from scikit-learn's `mean_absolute_percentage_error`, which keeps those rows using an epsilon floor and can return very large values. All other metrics match scikit-learn.
>
> **MSLE / RMSLE note:** inputs must be non-negative (the log is otherwise undefined). Negative inputs yield NaN rather than raising, mirroring (but not re-raising) scikit-learn's domain error.
>
> **sMAPE note:** uses the `mean(2·|y−p| / (|y|+|p|))` form (range `[0, 2]`); the `0/0` case (both `y` and `p` zero) contributes `0` to avoid division-by-zero blow-up.

### Sample Weights

*Every* metric supports optional sample weights via a `weight` column — including
ones that are awkward or unsupported elsewhere, like ROC AUC, log loss, MCC, and
Cohen's kappa:

```python
df.select(roc_auc("label", "score", weight="sample_weight"))
df.select(matthews_corrcoef("label", "prob", weight="w"))
df.select(mae("y", "pred", weight="w"))
```

### Custom Positive Class

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
specificity, MCC, Cohen's kappa). Regression metrics take continuous targets, so
they don't have `pos_label`.

## Use Cases

Polarbear is perfect for:

1. **Large-scale model evaluation**: Evaluate millions of predictions efficiently
2. **Real-time metrics**: Calculate metrics in streaming pipelines
3. **Group-wise metrics**: Leverage Polars' `group_by` for segment analysis
4. **Memory efficiency**: Process data that doesn't fit in memory with Polars LazyFrames

### Example: Group-wise ROC AUC

```python
import polars as pl
from polarbear import roc_auc

# Calculate ROC AUC per customer segment
df = pl.DataFrame({
    "segment": ["A", "A", "A", "B", "B", "B"],
    "label": [0, 1, 1, 0, 0, 1],
    "score": [0.2, 0.8, 0.9, 0.1, 0.3, 0.85]
})

result = df.group_by("segment").agg(
    roc_auc("label", "score")
)
print(result)
# shape: (2, 2)
# ┌─────────┬─────────────────────┐
# │ segment │ roc_auc_label_score │
# │ ---     │ ---                 │
# │ str     │ f64                 │
# ╞═════════╪═════════════════════╡
# │ A       │ 1.0                 │
# │ B       │ 0.5                 │
# └─────────┴─────────────────────┘
```

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and [just](https://github.com/casey/just) for task running.

### Quick Commands (with just)

```bash
just              # List all commands
just test         # Run tests
just test-fast    # Quick test run
just quality      # Run linting and type checking
just ci           # Run all CI checks locally
just pre-commit   # Quick pre-commit check
```

### Manual Commands

```bash
# Install dependencies
uv sync --all-groups

# Run tests
uv run pytest

# Run type checking
uv run ty check

# Run linting
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/
```

## Testing

Polarbear uses a comprehensive testing strategy:

- **Unit tests**: Basic functionality and edge cases
- **Property-based tests**: Random data generation with Hypothesis
- **Compatibility tests**: Verified against scikit-learn on multiple Polars versions

```bash
# Run all tests
just test  # or: uv run pytest

# Test multiple Polars versions
just test-compat

# Run benchmarks
just bench
```

## Performance

Polarbear runs every metric as a native Polars expression. The advantage is
**large and real where there's work to parallelize, and honest where there
isn't.** Numbers below are speedup vs scikit-learn (scikit-learn time ÷
polarbear time; higher = faster), median of clean benchmark runs.

**Where polarbear wins big** — grouped metrics and probabilistic/ranking metrics:

| Metric | 100k rows | 10M rows |
|--------|:---:|:---:|
| Grouped metrics (per segment) | **15–65x** | — |
| Brier Score | ~18x | ~8.6x |
| Log Loss | ~5.7x | ~4–5x |
| ROC AUC | ~4.5x | ~4.5x |
| Precision / F1 | ~3.5–5x | ~2–3x |

Grouped metrics are the standout: Polars parallelizes across groups while
scikit-learn loops in Python.

**Where it's at parity or slower** — trivial reductions (MAE, MSE, MAPE, R²) are
roughly even with scikit-learn at small-to-mid sizes and *slower* on a single
very large array, where NumPy's tight single-threaded loop beats one Polars
expression. Reach for polarbear on grouped/composed pipelines and the
probabilistic metrics; for a one-off MAE over a giant array, NumPy is fine.

See [docs/technical/PERFORMANCE.md](docs/technical/PERFORMANCE.md) for the full
per-metric breakdown, the size-scaling curve, and the ceteris-paribus Polars
version comparison.

## Requirements

- **Python**: 3.11+
- **Polars**: 1.0.0+

## Roadmap

- [x] ROC AUC for binary classification
- [x] Log Loss / Binary Cross-Entropy
- [x] Brier Score
- [x] Average Precision Score
- [x] Precision, Recall, F1 Score, Accuracy, Balanced Accuracy
- [x] Specificity, F-beta Score, Matthews Correlation Coefficient, Cohen's Kappa
- [x] R-squared, MAPE
- [x] Weighted variants for all metrics
- [x] Custom positive class label (`pos_label`: int, string, or bool)
- [x] Multi-version Polars support (1.0.0+)
- [x] Gini coefficient (normalized for non-negative targets)
- [ ] Multi-class ROC AUC (one-vs-rest, one-vs-one)
- [ ] Calibration metrics (ECE, MCE)

## Contributing

Contributions are welcome! Please:

1. Run `just ci` to verify your changes locally
2. Submit a Pull Request with a clear description

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with [Polars](https://www.pola.rs/)
- Tested against [scikit-learn](https://scikit-learn.org/)
- Property-based testing with [Hypothesis](https://hypothesis.readthedocs.io/)
