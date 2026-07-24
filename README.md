# Polarbearings 🐻‍❄️🧭

High-performance machine learning metrics implemented as native Polars expressions.

## Features

- **Fast where it counts**: Native Polars expressions — large wins on grouped and probabilistic metrics (see [Performance](#performance))
- **Weighted**: Optional sample weights on nearly every metric
- **Flexible labels**: Any positive class — `1`, `100`, `"cancer"`, `True` — via `pos_label`
- **Correct**: scikit-learn-faithful, verified with property-based testing
- **Composable**: Plain Polars expressions — drop into `group_by`, `over`, and lazy pipelines
- **Type-safe**: Full type hints

## Installation

```bash
uv add polarbearings
# or: pip install polarbearings
```

The only runtime dependency is Polars.

## Quick Start

```python
import polars as pl
from polarbearings import roc_auc

df = pl.DataFrame({"label": [0, 0, 1, 1, 1], "score": [0.1, 0.4, 0.35, 0.8, 0.9]})

# Every metric is a Polars expression — use it anywhere an expression is allowed.
df.select(roc_auc("label", "score"))
# shape: (1, 1)
# ┌─────────────────────┐
# │ roc_auc_label_score │
# │ f64                 │
# ╞═════════════════════╡
# │ 0.833333            │
# └─────────────────────┘
```

## Why Polarbearings

Every metric is a *pure Polars expression*. That single design choice is where the
strengths come from — things a scikit-learn wrapper or a compiled plugin can't
easily match:

- **Group-wise metrics at scale** — metrics drop straight into `group_by().agg()`
  and Polars parallelizes across groups (15–65x faster than a Python loop calling
  scikit-learn per segment).
- **A whole metric suite in one pass** — bundle every metric into a single
  `df.select([...])`; Polars shares the column scans and parallelizes across the
  independent outputs. 13 metrics on 10M rows run in **~1.05 s vs scikit-learn's
  ~6.1 s (5.8x)** — see [Performance](#performance).
- **Sample weights almost everywhere** — nearly every metric accepts an optional
  `weight` column, including ROC AUC, log loss, MCC, and Cohen's kappa (a few
  exceptions are noted per metric).
- **Any positive class** — `pos_label` accepts `1`, `100`, `"cancer"`, `True`, …
  no remapping your labels to 0/1.
- **scikit-learn-faithful** — names and edge-case semantics mirror scikit-learn
  (e.g. `null` for undefined cases), so migration is ~1:1.
- **Composable & lazy-friendly** — pure expressions fold into lazy query plans,
  `over()` windows, and the rest of your Polars pipeline.
- **Zero build, one dependency** — pure Python emitting Polars expressions; no
  compiled extension, installs anywhere Polars runs, supports Polars 1.0.0+.

### A whole evaluation report in one `select`

Because every metric is just an expression, a full report is one `df.select(...)`
— Polars reads each column once and fans the work across the independent outputs:

<!--- invisible-code-block: python
df = pl.DataFrame({
    "label": [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    "prob": [0.2, 0.45, 0.6, 0.55, 0.75, 0.35, 0.3, 0.8, 0.65, 0.5, 0.7, 0.4],
})
--->

```python
from polarbearings import (
    precision,
    recall,
    f1_score,
    roc_auc,
    average_precision,
    log_loss,
    brier_score,
    confusion_matrix,
)

df.select(
    precision("label", "prob"),
    recall("label", "prob"),
    f1_score("label", "prob"),
    roc_auc("label", "prob"),
    average_precision("label", "prob"),
    log_loss("label", "prob"),
    brier_score("label", "prob"),
    confusion_matrix("label", "prob"),  # struct {threshold, tp, fp, fn, tn}
)
# One tidy row, one column per metric (8 columns; abbreviated):
# shape: (1, 8)
# ┌──────────────────────────┬───────────────────────┬─────┬────────────────────────┐
# │ precision_label_prob_0.5 ┆ recall_label_prob_0.5 ┆  …  ┆ brier_score_label_prob │
# │ f64                      ┆ f64                   ┆     ┆ f64                    │
# ╞══════════════════════════╪═══════════════════════╪═════╪════════════════════════╡
# │ 0.714286                 ┆ 0.833333              ┆  …  ┆ 0.161944               │
# └──────────────────────────┴───────────────────────┴─────┴────────────────────────┘
```

`benchmarks/bench_multi_metric.py` reproduces this timing across sizes from 100 to
1M rows.

## Metrics

~40 metrics across ranking, probabilistic, classification, calibration, and
regression families, plus curve generators, bootstrap CIs, deterministic splitting,
and threshold utilities. The **[full metrics reference →
docs/guides/METRICS.md](https://github.com/jerheff/polarbearings/blob/main/docs/guides/METRICS.md)** documents every metric's
semantics, edge cases, and scikit-learn correspondence.

| Family | What's in it |
| --- | --- |
| **Ranking** | `roc_auc`, `average_precision`, `gini_coefficient`, `dcg_score`, `ndcg_score` |
| **Probabilistic** | `log_loss`, `brier_score` |
| **Classification** | `precision`, `recall`, `f1_score`, `fbeta_score`, `specificity`, `accuracy`, `balanced_accuracy`, `matthews_corrcoef`, `cohens_kappa`, `jaccard_score`, `confusion_matrix` |
| **Thresholds** | `threshold_sweep`, `quantiles` / `equal_width` / `linspace` specs |
| **Curves** | `roc_curve`, `pr_curve`, `det_curve`, `expected_cost`, `confusion_curve` |
| **Regression** | `mae`, `mse`, `rmse`, `r2_score`, `mape`, `smape`, MSLE/RMSLE, `huber_loss`, `log_cosh_loss`, pinball, Tweedie/Poisson/gamma deviance, D² scores, `max_error`, `median_absolute_error` |
| **Calibration** | `calibration_curve`, `expected_calibration_error` (ECE), `maximum_calibration_error` (MCE) |
| **Weights & CIs** | `balanced_sample_weight`, `balanced_class_weights`, `bootstrap_ci`, `bootstrap_weight` |
| **Splitting** | `hash_split`, `hash_splits`, `hash_fold`, `hash_uniform` |

<!--- invisible-code-block: python
reg = pl.DataFrame({
    "y": [1.0, 2.0, 3.0, 4.0, 5.0],
    "pred": [1.1, 1.9, 3.2, 3.8, 5.5],
    "w": [1.0, 1.0, 1.0, 2.0, 1.0],
})
--->

```python
from polarbearings import roc_auc, f1_score, mae, calibration_curve

# Classification + probabilistic metrics are plain expressions:
df.select(roc_auc("label", "prob"), f1_score("label", "prob", threshold=0.7))

# Regression metrics too:
reg.select(mae("y", "pred", weight="w"))

# Curve helpers take a (Lazy)Frame and return a plot-ready LazyFrame:
calibration_curve(df, "label", "prob", n_bins=10, strategy="quantile").collect()
```

### Cross-cutting behaviour

Four behaviours are shared by the metrics rather than specific to one — full
details in the [metrics reference](https://github.com/jerheff/polarbearings/blob/main/docs/guides/METRICS.md#cross-cutting-behaviour):

- **Output names** — each expression is pre-aliased `<metric>_<target>_<score>`
  (plus suffixes for a weight, non-default `pos_label`, or threshold), so
  `roc_auc("label", "score")` yields a `roc_auc_label_score` column. Chain
  `.alias("auc")` to rename it.
- **Sample weights** — pass `weight="col"` to nearly any metric. Six omit it
  (`dcg_score`, `ndcg_score`, `max_error`, `median_absolute_error`,
  `d2_absolute_error_score`, `d2_pinball_score`) where a weighted form is undefined
  or not cleanly expressible.
- **Custom positive class** — `pos_label` accepts ints, strings, or booleans; no
  remapping to 0/1.
- **Missing values** — any `null`/`NaN` in `target`, score, or `weight` makes a
  metric return `null` (loud, not silent), scoped to the evaluation context. Drop
  rows yourself for complete-case behaviour. Curve helpers are the exception — they
  drop incomplete rows.

### Diagnostic plots

`notebooks/diagnostics.ipynb` gives examples of ROC/PR/DET/calibration curves and bootstrap
confidence bands with these helpers and visualizing them with Plotly.

## Use Cases

Polarbearings is a good fit for:

1. **Large-scale model evaluation** — millions of predictions, efficiently
2. **Group-wise metrics** — per-segment analysis via Polars' `group_by`
3. **Streaming / out-of-core** — lazy expressions over `LazyFrame`s
4. **Composed evaluation reports** — a whole metric suite in one pass

<!--- invisible-code-block: python
df = pl.DataFrame({
    "segment": ["A", "A", "B", "B"],
    "label": [0, 1, 0, 1],
    "score": [0.1, 0.9, 0.5, 0.5],
})
--->

```python
from polarbearings import roc_auc

# ROC AUC per customer segment, in one parallelized pass:
df.group_by("segment").agg(roc_auc("label", "score"))
# shape: (2, 2)
# ┌─────────┬─────────────────────┐
# │ segment │ roc_auc_label_score │
# ╞═════════╪═════════════════════╡
# │ A       │ 1.0                 │
# │ B       │ 0.5                 │
# └─────────┴─────────────────────┘
```

## Performance

Polarbearings runs every metric as a native Polars expression. The advantage is
**large and real where there's work to parallelize, and honest where there
isn't.** Numbers below are speedup vs scikit-learn (scikit-learn time ÷
polarbearings time; higher = faster), median of clean benchmark runs.

**Where polarbearings wins big** — grouped, probabilistic, and ranking metrics.
Ranges span polars 1.0.0 and 1.41.2; see
[PERFORMANCE.md](https://github.com/jerheff/polarbearings/blob/main/docs/technical/PERFORMANCE.md) for the per-version, per-size matrix.

| Metric | 100k rows | 10M rows |
|--------|:---:|:---:|
| Grouped metrics (per segment) | **~15–60x** | — |
| Precision / F1 | ~5–22x | ~14–23x |
| Brier Score | ~9–10x | ~7–8x |
| Log Loss | ~5–6x | ~4–5x |
| ROC AUC | ~4.5–6x | ~5x |
| 13 metrics in one `select` | — | **5.8x** |

**Where it's at parity or slower** — trivial reductions (MAE, MSE, MAPE, R²) are
roughly even with scikit-learn at small-to-mid sizes and *slower* on a single
very large array, where NumPy's tight single-threaded loop beats one Polars
expression. Reach for polarbearings on grouped/composed pipelines and the
probabilistic metrics; for a one-off MAE over a giant array, NumPy is fine.

See [docs/technical/PERFORMANCE.md](https://github.com/jerheff/polarbearings/blob/main/docs/technical/PERFORMANCE.md) for the full
per-metric breakdown, the size-scaling curve, and the ceteris-paribus Polars
version comparison.

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management
and [just](https://github.com/casey/just) for task running. Run `just` to list all
recipes.

```bash
uv sync --all-groups   # install dependencies
just test              # run tests        (or: uv run pytest)
just quality           # lint + type-check
just check             # fast local check: lint + type-check + tests (not full CI)
just test-compat       # test against min / mid / latest Polars
just bench             # benchmarks vs scikit-learn
```

Testing combines unit tests, property-based tests (Hypothesis), and
scikit-learn compatibility tests across multiple Polars versions — see
[docs/guides/TESTING.md](https://github.com/jerheff/polarbearings/blob/main/docs/guides/TESTING.md).

## Requirements

- **Python**: 3.11+
- **Polars**: 1.0.0+

## Roadmap

- [x] Publish to PyPI (first tagged release)
- [ ] KS statistic & lift/gain curves (churn / credit workflows)
- [ ] Ranking metrics: precision@k, recall@k, MRR, MAP
- [ ] Weighted (linear / quadratic) Cohen's kappa for ordinal targets
- [ ] `.metrics` Polars expression namespace (`pl.col(...).metrics.roc_auc(...)`)

More candidate metrics and their implementation notes live in
[docs/FUTURE_IDEAS.md](https://github.com/jerheff/polarbearings/blob/main/docs/FUTURE_IDEAS.md).

## Contributing

Contributions are welcome! Please:

1. Run `just check` to verify your changes locally (lint + type-check + tests)
2. Submit a Pull Request with a clear description

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with [Polars](https://www.pola.rs/)
- Tested against [scikit-learn](https://scikit-learn.org/)
- Property-based testing with [Hypothesis](https://hypothesis.readthedocs.io/)
