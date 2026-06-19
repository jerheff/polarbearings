# Polars Releases: What Changed in the Last Year (March 2025 – June 2026)

A deep look at the capabilities added to Polars across versions **1.24 through 1.41**, organized by theme, with emphasis on how each change affects common data science workflows and ML metrics computation.

> Sections 1–10 cover **1.24–1.39**. [Section 11](#11-update-releases-140-141-aprilmay-2026) extends the analysis through the latest release at the time of writing, **polars 1.41.2** (May 29, 2026).

**Sources:**

- [Polars GitHub Releases](https://github.com/pola-rs/polars/releases)
- [Polars Blog](https://pola.rs/posts/)
- [Polars in Aggregate (Dec 2025)](https://pola.rs/posts/polars-in-aggregate-dec25/)
- [Polars Benchmarks (May 2025)](https://pola.rs/posts/benchmarks/)

---

## Table of Contents

1. [New Streaming Engine](#1-new-streaming-engine)
2. [New Data Types](#2-new-data-types)
3. [Categorical / Enum Overhaul](#3-categorical--enum-overhaul)
4. [New Expressions for Data Science & Metrics](#4-new-expressions-for-data-science--metrics)
5. [Performance Improvements](#5-performance-improvements)
6. [GPU Acceleration](#6-gpu-acceleration)
7. [I/O and Ecosystem](#7-io-and-ecosystem)
8. [Breaking Changes and Deprecations](#8-breaking-changes-and-deprecations)
9. [Implications for Polarbear](#9-implications-for-polarbear)
10. [Improvement Plan: Metrics Leveraging New Polars Features](#10-improvement-plan-polarbear-metrics-leveraging-new-polars-features)
11. [Update: Releases 1.40–1.41 (April–May 2026)](#11-update-releases-140-141-aprilmay-2026)

---

## 1. New Streaming Engine

**The single largest change of the year.** The new streaming engine, based on [morsel-driven parallelism](https://db.in.tum.de/~leis/papers/morsels.pdf) research, matured from experimental to production-ready and replaced the legacy engine entirely.

- **Docs:** [Streaming User Guide](https://docs.pola.rs/user-guide/concepts/streaming/)
- **Blog coverage:** [Polars in Aggregate (Dec 2025)](https://pola.rs/posts/polars-in-aggregate-dec25/)

### Timeline

| Version | What happened |
|---------|---------------|
| v1.26 | IO-plugin support added; `streaming=False` parameter deprecated |
| v1.31 | **Legacy streaming engine removed.** New engine gained `int_range`, `arg_unique`, `arg_where`, `shift`, `diff`, cumulative ops, order-preserving groupby |
| v1.32 | Streaming `Expr.slice`, `any()`, `all()`, row-separable functions, bitwise aggregation, categorical/enum min/max |
| v1.35 | Streaming `ewm_mean()`, `approx_n_unique`; fixed `group_by_dynamic` slowness on sparse data |
| v1.38 | Streaming merge-join, streaming NDJSON decompression, elementwise CSE optimization |
| v1.39 | Streaming `AsOf` join, streaming `arg_min`/`arg_max`, streaming cloud CSV/NDJSON downloads, parallel in-memory sinks |

### How to use it

The streaming engine is now the default path for sink operations and will soon be the default for all `collect()` calls. You can opt in explicitly:

```python
# Explicit streaming collect
result = lf.collect(engine="streaming")

# Set as session default
pl.Config.set_engine_affinity("streaming")

# Sink operations always stream
lf.sink_parquet("output.parquet")
lf.sink_csv("output.csv")
```

**Batch processing** — new in v1.35+ — lets you consume streaming results incrementally instead of materializing the full DataFrame:

```python
# Generator of sub-DataFrames, each with ~50k rows
for batch in lf.collect_batches(chunk_size=50_000):
    process(batch)

# Custom callback on streaming chunks
lf.sink_batches(callback=my_func)
```

- **API:** [`LazyFrame.collect_batches()`](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect_batches.html)

### Performance impact

The new engine delivers **3–7x speedups** over the in-memory engine on datasets that exceed L3 cache. It achieves this through:

- **Morsel-driven parallelism**: work is divided into small "morsels" that are dynamically scheduled across cores, avoiding the load-imbalance problems of partition-based parallelism.
- **Pipeline fusion**: multiple operators are fused into a single pipeline, reducing materialization of intermediate results.
- **Streaming I/O**: data is read, processed, and written in a single pass without loading the full dataset into memory.

### What this means for data science

Before the new engine, you had two modes: eager (immediate, in-memory) and lazy (query-planned, but still fully materialized on `collect()`). The streaming engine adds a third mode: lazy evaluation that processes data in constant memory. This changes the calculus for:

- **Feature engineering on large datasets**: Window functions, rolling stats, and group-by aggregations that used to OOM on 50GB+ datasets now stream through in bounded memory.
- **Model evaluation at scale**: Computing metrics over billions of predictions no longer requires holding the full prediction DataFrame in RAM.
- **ETL pipelines**: `scan_parquet() → transform → sink_parquet()` pipelines now operate end-to-end in streaming mode, making Polars competitive with Spark for medium-scale batch jobs (single-node, <1TB).

---

## 2. New Data Types

### Decimal (stabilized in v1.35)

The `Decimal` type moved from `unstable` to stable. It uses 128-bit fixed-point arithmetic with up to 38 significant digits and a fixed scale.

- **API:** [`polars.datatypes.Decimal`](https://docs.pola.rs/api/python/stable/reference/api/polars.datatypes.Decimal.html)

```python
# Create a Decimal column
df = pl.DataFrame({
    "price": [pl.lit("0.1").cast(pl.Decimal(precision=10, scale=2)),
              pl.lit("0.2").cast(pl.Decimal(precision=10, scale=2))]
})

# 0.1 + 0.2 == 0.3, not 0.30000000000000004
```

**How this matters for data science:**

- **Financial modeling**: Portfolio returns, transaction amounts, and fee calculations need exact arithmetic. Decimal eliminates the class of bugs where `sum(weights) = 0.9999999999999998` instead of `1.0`.
- **Metric precision**: When computing metrics like log loss where values near 0 and 1 are clipped, Decimal avoids the floating-point accumulation errors that can drift results on very large datasets.
- **Reproducibility**: Decimal arithmetic is deterministic across platforms — no more platform-dependent float rounding differences in CI.

As of v1.35+, Decimal works with `search_sorted`, `product`, `Expr.sign`, and standard aggregations.

### Int128 / UInt128 (v1.34)

A signed 128-bit integer type ranging from approximately ±1.7 × 10³⁸ — roughly 18 quintillion times the range of Int64.

**Use cases in data science:**
- Hash-based identifiers (UUIDs stored as integers)
- Exact large-integer arithmetic for combinatorial computations
- Interop with databases that use 128-bit keys

### Float16 (v1.36)

Half-precision (16-bit) float support. Uses 2 bytes per value instead of 4 (Float32) or 8 (Float64).

**Use cases:**
- Loading model weights or embeddings stored in fp16 format
- Memory-efficient storage of low-precision features
- Interop with ML frameworks (PyTorch, JAX) that default to fp16/bf16

### Extension Types (v1.36)

User-defined data types can now be registered with Polars. This opens the door for domain-specific types (e.g., geospatial coordinates, currency-tagged decimals, probability distributions) that carry metadata and custom display logic.

---

## 3. Categorical / Enum Overhaul

**Version:** v1.32 — **This is a breaking change.**

- **Blog:** [Understanding the New Categoricals](https://pola.rs/posts/categoricals-refactor/)
- **API:** [`polars.datatypes.Categories`](https://docs.pola.rs/api/python/dev/reference/api/polars.datatypes.Categories.html) (dev docs)

### What changed

The old `StringCache` (both scoped and global) was replaced with a new `Categories` object:

```python
# Old way (removed)
with pl.StringCache():
    df1 = pl.DataFrame({"col": pl.Series(["a", "b"]).cast(pl.Categorical)})
    df2 = pl.DataFrame({"col": pl.Series(["b", "c"]).cast(pl.Categorical)})

# New way
cats = pl.Categories(name="my_categories")
df1 = pl.DataFrame({"col": pl.Series(["a", "b"]).cast(pl.Categorical(cats))})
df2 = pl.DataFrame({"col": pl.Series(["b", "c"]).cast(pl.Categorical(cats))})
```

### Key differences

| Aspect | Old | New |
|--------|-----|-----|
| Scope | Global or context-manager `StringCache` | Explicit `Categories` object, passed where needed |
| Physical type | Always `UInt32` | Configurable: `UInt8` (≤255 cats), `UInt16` (≤65K), `UInt32` (default) |
| Comparison | Configurable (`physical` or `lexical`) | Always **lexical** (alphabetical) |
| Enum | Separate concept | Frozen/immutable variant of `Categories` with deterministic ordering |
| Thread safety | Global state → race conditions | Object-scoped → no global state |

### How this affects data science workflows

- **Memory efficiency**: Using `UInt8` physical type for low-cardinality columns (e.g., `["male", "female", "other"]`) cuts memory from 4 bytes to 1 byte per row. On a 100M-row dataset, that's 300MB saved per column.
- **Sorting is now predictable**: Lexical ordering means sorted categoricals always produce alphabetical order. The old `physical` ordering (based on insertion order) was a common source of nondeterministic bugs in group-by results.
- **Enum for ML targets**: If your classification target has a fixed, known set of classes, `Enum` guarantees those classes are always in the same physical order — useful for confusion matrix layouts and label encoding.
- **Cross-DataFrame joins**: The explicit `Categories` object makes it clear which DataFrames share a category mapping, eliminating the "forgot to use StringCache" class of bugs.

---

## 4. New Expressions for Data Science & Metrics

### `rolling_rank()` (v1.35)

Computes the rank of each value within a rolling window.

- **API:** [`Expr.rolling_rank()`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.rolling_rank.html)

```python
df.with_columns(
    pl.col("score").rolling_rank(window_size=100).alias("rank_in_window")
)
```

**How to use it in data science:**
- **Time-series feature engineering**: Rolling rank captures relative position within a local window — useful for momentum signals, anomaly detection, and non-parametric trend indicators.
- **Online learning evaluation**: Track how a model's predictions rank against recent ground truth in a sliding window.
- **Percentile-based features**: Rolling rank divided by window size gives a rolling percentile, which is a robust alternative to z-scores for non-normal distributions.

### `Expr.item()` (v1.35)

Strict single-value extraction. Raises an error if the expression doesn't resolve to exactly one value.

- **API:** [`Expr.item()`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.item.html)

```python
# Before: silent bugs if multiple values
auc = df.select(roc_auc("target", "score")).to_series().to_list()[0]

# After: explicit, fails loudly if something is wrong
auc = df.select(roc_auc("target", "score")).item()
```

**Why this matters:** Metric functions return single scalar values. `.item()` is both more readable and safer than indexing into a list or series, because it will error if the expression unexpectedly returns multiple rows.

### `is_close()` (v1.32)

Approximate floating-point equality, analogous to `numpy.isclose()`.

- **API:** [`Expr.is_close()`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.is_close.html)

```python
df.filter(pl.col("predicted").is_close(pl.col("actual"), rtol=1e-5, atol=1e-8))
```

**Use cases:**
- Numerical validation in tests: assert that two metric implementations agree within tolerance
- Data quality checks: find rows where two independently computed features should be equal but diverge due to floating-point issues
- Deduplication: identify near-duplicate rows based on numerical similarity

### `min_by` / `max_by` (v1.37)

Returns the value of one column at the row where another column is minimized/maximized.

- **API:** [`Expr.min_by()`](https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.min_by.html), [`Expr.max_by()`](https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.max_by.html) (dev docs)

```python
# Find the threshold that maximizes F1 score
best_threshold = (
    threshold_sweep_df
    .select(pl.col("threshold").max_by(pl.col("f1_score")))
    .item()
)
```

**How this changes threshold optimization:** Previously, finding the optimal threshold required either a sort + tail or an `arg_max` + indexing two-step. `max_by` collapses this into a single, readable expression that works inside group-by and streaming contexts.

### Multi-Quantile Computation (v1.38)

The `quantile()` expression gained the ability to compute multiple quantiles in a single pass over the data.

- **API:** [`Expr.quantile()`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.quantile.html)

```python
# Compute multiple quantiles efficiently
quantiles = df.select(
    pl.col("score").quantile([0.1, 0.25, 0.5, 0.75, 0.9])
)
```

**Why this matters for metrics:** Percentile-based thresholds (e.g., "classify as positive if score is above the 90th percentile") previously required one pass per quantile. Multi-quantile computes them all at once — important when you're sweeping across many percentile thresholds to build a precision-recall curve.

### `list.filter()` (v1.30)

Filter elements within list columns using a boolean mask or predicate.

- **API:** [`Expr.list.filter()`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.list.filter.html)

```python
# Keep only positive predictions within each group's prediction list
df.with_columns(
    pl.col("predictions").list.filter(pl.element() > 0.5).alias("positive_preds")
)
```

**Data science use case:** When working with recommendation systems or multi-label classification, predictions are often stored as list columns. `list.filter()` lets you apply thresholds, remove nulls, or select top-k within each row's list without exploding the DataFrame.

### `list.agg()` / `arr.agg()` (v1.35)

Run arbitrary aggregation expressions over elements in list or fixed-size array columns.

- **API:** [`Expr.list.agg()`](https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.list.agg.html) (dev docs)

```python
# Compute mean and std of each row's prediction list
df.with_columns(
    pl.col("predictions").list.agg(pl.element().mean()).alias("mean_pred"),
    pl.col("predictions").list.agg(pl.element().std()).alias("std_pred"),
)
```

**Use case:** Per-row aggregation over variable-length lists — common in ensemble models (aggregate multiple model outputs), time-series (aggregate irregular observations), and NLP (aggregate token-level scores to document-level).

### `LazyFrame.pivot()` (v1.36)

Pivot (reshape from long to wide) is now available in lazy mode, meaning it works with the streaming engine and query optimizer.

- **API:** [`LazyFrame.pivot()`](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.pivot.html)

```python
# Build a confusion matrix lazily
confusion = (
    predictions_lf
    .with_columns((pl.col("score") > threshold).cast(pl.Int8).alias("predicted"))
    .group_by("actual", "predicted")
    .agg(pl.len().alias("count"))
    .pivot(on="predicted", index="actual", values="count")
    .collect(engine="streaming")
)
```

**Impact:** Confusion matrices, contingency tables, and cross-tabulations can now be computed in streaming mode on arbitrarily large datasets.

### `.over()` without `partition_by` (v1.30)

Window functions no longer require an explicit partition column — they default to computing over the entire DataFrame.

```python
# Before
df.with_columns(pl.col("score").rank().over(pl.lit(1)).alias("global_rank"))

# After
df.with_columns(pl.col("score").rank().over().alias("global_rank"))
```

### `Expr.log()` with expression base (v1.33)

`log()` now accepts another expression as the base, not just a literal.

```python
# Information gain with variable base
df.with_columns(
    pl.col("probability").log(pl.col("base_value")).alias("log_prob")
)
```

Useful for information-theoretic metrics where the log base varies (bits vs nats vs custom).

---

## 5. Performance Improvements

Beyond the streaming engine, Polars shipped significant performance work across the stack.

### Common Subplan Elimination (v1.31+)

When running `pl.collect_all()` on multiple LazyFrames that share common branches (e.g., reading the same Parquet file, applying the same filters), the optimizer now detects shared subplans and executes them only once.

```python
# Both queries read and filter the same data — the shared scan+filter runs once
lf_base = pl.scan_parquet("data.parquet").filter(pl.col("year") == 2025)
lf_metrics = lf_base.select(compute_metrics())
lf_features = lf_base.select(compute_features())

results = pl.collect_all([lf_metrics, lf_features])  # ~35% faster than separate collects
```

**Data science impact:** It's common to compute multiple metric sets or feature groups from the same base data. CSE makes the "fan-out" pattern essentially free.

### Hash Table and Memory

- **foldhash** replaced ahash (v1.26) — faster hash computation for joins and group-by
- **Transparent Huge Pages** enabled by default (v1.26) — reduces TLB misses on large allocations
- **Dedicated `arg_max`/`arg_min` kernel** for group-by (v1.38) — avoids materializing the full column just to find the index of the extremum

### Parquet I/O

| Optimization | Version | Effect |
|-------------|---------|--------|
| Dictionary encoding for floats | v1.36 | Smaller Parquet files when float columns have repeated values |
| ZSTD decompression context caching | v1.32 | Avoids re-initializing the decompressor per row group |
| String regex prefiltering | v1.36 | Skips row groups whose dictionary doesn't match the regex |
| Row group skipping for floats | v1.39 | Uses min/max statistics to skip irrelevant row groups |
| Concurrent file schema resolution | v1.38 | Faster `scan_parquet("*.parquet")` on many files |
| Row group prefetch increased to 8 | v1.39 | Better I/O pipelining |

### Other

- **Filter pushdown past joins** (v1.32) — filters applied after a join are pushed before it, reducing the data flowing through the join
- **Rolling quantile complexity reduction** (v1.32) — faster `rolling_quantile()` for large windows
- **Boolean casting 2x faster** (v1.39)
- **Duration/interval string parsing 2–5x faster** (v1.35)
- **Numerical formatting** switched from ryu to zmij (v1.37) — faster `.to_pandas()` and CSV writes

### Benchmarks

The [May 2025 benchmarks post](https://pola.rs/posts/benchmarks/) shows Polars outperforming DuckDB, pandas, and Spark on the standard TPC-H and DB-Benchmark suites, with the streaming engine widening the gap further on larger scale factors.

---

## 6. GPU Acceleration

Polars GPU acceleration, powered by NVIDIA RAPIDS cuDF, reached open beta.

- **Docs:** [GPU Support Guide](https://docs.pola.rs/user-guide/gpu-support/)
- **Blog:** [Larger-than-RAM GPU Processing with UVM](https://pola.rs/posts/uvm-larger-than-ram-gpu/)

### How to use it

```python
# Single line change — automatic fallback to CPU for unsupported ops
result = lf.collect(engine="gpu")
```

### Performance

- Up to **13x faster** than CPU Polars for grouped aggregations and joins
- Unified Virtual Memory (UVM) enables processing datasets larger than VRAM by transparently spilling to system RAM
- The GPU engine supports most common operations: filters, projections, joins, group-by, sorts, and window functions

### What this means for data science

- **Hyperparameter sweeps**: When computing metrics across thousands of threshold values or model configurations, GPU acceleration turns minutes into seconds.
- **Feature engineering**: Group-by aggregations over high-cardinality columns (user-level features over millions of users) are the sweet spot for GPU speedup.
- **Real-time scoring pipelines**: Sub-second latency on large batch scoring jobs.

**Requirements:** NVIDIA GPU with CUDA support, `pip install polars[gpu]`.

---

## 7. I/O and Ecosystem

### New I/O Targets

| Feature | Version | Description |
|---------|---------|-------------|
| `sink_iceberg()` | v1.39 (unstable) | Stream results directly to Apache Iceberg tables |
| `sink_delta()` | v1.37 | Stream results to Delta Lake tables |
| `scan_lines()` | v1.38 | Read raw text lines as a LazyFrame |
| PyTorch Tensor init | v1.29 | `pl.DataFrame(torch_tensor)` — direct construction from tensors |

### Polars Cloud (launched September 2025)

Polars Cloud provides distributed execution for Polars queries, targeting the gap between single-node Polars and full-blown Spark clusters.

- **Docs:** [Polars Cloud](https://docs.pola.rs/polars-cloud/)
- **Blog:** [What We Are Building](https://pola.rs/posts/polars-cloud-what-we-are-building/), [Launch Announcement](https://pola.rs/posts/polars-cloud-launch/)

The key insight: since Polars queries are already expressed as lazy plans, the same plan can be shipped to a cloud runtime for distributed execution without rewriting any code.

### SQL Enhancements

Polars' SQL context gained: `LPAD`, `RPAD`, `QUALIFY`, `FETCH`, FROM-first SELECT syntax, `ARRAY` initialization, and window functions (`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LEAD`, `LAG`). This makes Polars more viable as a drop-in replacement for SQL-based analytics workflows.

### Infrastructure

- **Free-threading safety** marked in Python 3.13+ (v1.37) — Polars is ready for Python's no-GIL future
- **Pyright type completeness** reached 95%+ (v1.31) — better IDE support and type inference
- **Runtime packages refactored** (v1.34–v1.35): `polars-lts-cpu` → `polars[rtcompat]`, `polars-u64-index` → `polars[rt64]`

---

## 8. Breaking Changes and Deprecations

| Change | Version | Migration |
|--------|---------|-----------|
| Legacy streaming engine removed | v1.31 | Use `engine="streaming"` or default |
| `streaming=False` deprecated | v1.26 | Remove the parameter; streaming is automatic for sinks |
| `StringCache` replaced by `Categories` | v1.32 | See [categoricals refactor blog](https://pola.rs/posts/categoricals-refactor/) |
| Categorical ordering always lexical | v1.32 | Remove `ordering` parameter; sort gives alphabetical order |
| `rolling` renamed to `overlapping` | v1.35 | Find-and-replace in your codebase |
| Python 3.9 dropped | v1.37 | Minimum is now Python 3.10+ |
| `Expr.agg_groups()` deprecated | v1.35 | Use `Expr.implode()` or `Expr.list` |
| `read_csv_batched()` deprecated | v1.39 | Use `scan_csv().collect_batches()` |
| `allow_missing_columns` deprecated | v1.31 | Use `missing_columns` parameter |
| `retries=n` deprecated | v1.38 | Use `storage_options={"max_retries": n}` |
| Flat `list.gather` deprecated | v1.29 | Use elementwise `list.gather` |

---

## 9. Implications for Polarbear

Polarbear currently requires `polars>=1.0.0` and tests against versions 1.0.0, 1.24.0, and 1.41.2. Many of the features below are available in the tested compatibility matrix.

### Direct opportunities

| Polars Feature | Polarbear Impact | How |
|---------------|-----------------|-----|
| **Streaming engine** | All lazy metric expressions get 3–7x speedups for free | Users just call `.collect(engine="streaming")` on polarbear results |
| **`rolling_rank()`** | Could enable new rolling/windowed ranking metrics | Build on top of native `rolling_rank()` instead of custom implementations |
| **`min_by` / `max_by`** | Cleaner threshold sweep implementation | Replace sort + tail pattern in `threshold_sweep()` with `max_by` |
| **Multi-quantile** | Faster `percentile_thresholds()` | Compute all percentile thresholds in a single pass |
| **`Expr.item()`** | Safer scalar extraction in examples and docs | Recommend `.item()` over `.to_list()[0]` in documentation |
| **`is_close()`** | Better test assertions | Use in property-based tests for approximate equality |
| **Lazy `pivot()`** | Streaming confusion matrices | Confusion matrix computation can work on datasets larger than RAM |
| **`list.filter()` / `list.agg()`** | New multi-label and ensemble metric patterns | Enable metrics over list-typed prediction columns |
| **Decimal** | High-precision metrics option | Offer Decimal-compatible metric variants for financial use cases |

### Version strategy considerations

- If polarbear wants to use `min_by`/`max_by` (v1.37+), `list.agg()` (v1.35+), or `rolling_rank()` (v1.35+), the minimum Polars version would need to increase, or these features would need to be gated behind version checks.
- The categorical overhaul (v1.32) doesn't directly impact polarbear today since metrics operate on numeric columns, but it's worth noting for any future categorical metric support.
- The `rolling` → `overlapping` rename (v1.35) could affect downstream users if polarbear ever exposes rolling window APIs.

### Documentation and examples

With the streaming engine mature, polarbear's benchmarks and documentation should highlight the streaming path:

```python
import polars as pl
import polarbear as pb

# Compute AUC on a billion-row dataset in streaming mode
result = (
    pl.scan_parquet("predictions/*.parquet")
    .select(pb.roc_auc("target", "score"))
    .collect(engine="streaming")
    .item()
)
```

This pattern — scan, compute metric expression, stream-collect, extract scalar — is now the idiomatic way to compute metrics on large datasets with Polars + polarbear.

---

## 10. Improvement Plan: Polarbear Metrics Leveraging New Polars Features

This section maps specific changes to each of polarbear's 18 metrics and utility functions, grouped by effort and the minimum Polars version they would require. Each improvement references the current implementation and explains exactly what would change.

### Version constraint summary

| Minimum Polars | Features unlocked | Impact on polarbear |
|---------------|-------------------|---------------------|
| **1.0.0** (current) | Everything that exists today works | No changes needed |
| **1.30.0** | `list.filter()`, `.over()` without partition_by | Enables list-based metric patterns, simpler window expressions |
| **1.32.0** | `is_close()`, lazy `arr.mean()`, filter pushdown past joins | Better tests, faster joined metric pipelines |
| **1.33.0** | `Expr.log()` with expression base, `map_columns()` | More flexible log loss, batch metric computation |
| **1.35.0** | `rolling_rank()`, `Expr.item()`, `list.agg()`, Decimal stable, `ewm_mean` streaming | Rolling metrics, safer scalar extraction, high-precision mode |
| **1.36.0** | `LazyFrame.pivot()`, Float16 support | Streaming confusion matrices, fp16 input handling |
| **1.37.0** | `min_by()`/`max_by()`, `PartitionBy` | Cleaner threshold optimization, per-group metrics |
| **1.38.0** | Multi-quantile, streaming merge-join, dedicated `arg_max` kernel | Faster percentile thresholds, faster threshold sweep |

### Tier 1 — No version bump needed (polars >= 1.0.0)

These improvements use only features available since Polars 1.0.0 and require no dependency change.

#### 1.1 Streaming engine documentation and benchmarks

**What:** Update all documentation, examples, and benchmarks to show the streaming execution path. Add a "Large-scale usage" section to the README.

**Why:** The streaming engine is available in all supported versions (it just wasn't the default until later). Users on any version can opt in with `engine="streaming"`.

**Current state:** No documentation mentions streaming at all.

**Change:**
```python
# Before (current docs)
df.select(pb.roc_auc("target", "score"))

# After (recommended pattern for large data)
(
    pl.scan_parquet("predictions/*.parquet")
    .select(pb.roc_auc("target", "score"))
    .collect(engine="streaming")
    .item()
)
```

#### 1.2 Batch metric computation with `pl.collect_all()` + CSE

**What:** Document and test the pattern of computing multiple independent metrics in a single `pl.collect_all()` call, which triggers Common Subplan Elimination.

**Why:** Users frequently compute ROC AUC, average precision, log loss, and Brier score on the same data. Today each `.select()` scans the data independently. With `collect_all()`, the shared scan and filter execute once.

**Current state:** Each metric is called independently in examples.

**Change:**
```python
# Compute 4 metrics with a single data scan
lf = pl.scan_parquet("predictions.parquet")
results = pl.collect_all([
    lf.select(pb.roc_auc("target", "score")),
    lf.select(pb.average_precision("target", "score")),
    lf.select(pb.log_loss("target", "prob")),
    lf.select(pb.brier_score("target", "prob")),
])
```

### Tier 2 — Moderate version bump (polars >= 1.35.0)

These changes require raising the minimum Polars version from 1.0.0 to 1.35.0. This is the sweet spot: v1.35 is 9+ months old, covers the most impactful new features, and is close to what polarbear already tests against (1.41.2).

#### 2.1 Rewrite `percentile_thresholds()` to be lazy-native

**What:** Replace the current eager, loop-based implementation with a single lazy expression using multi-quantile computation.

**Why:** The current implementation (`percentile_thresholds()` in `classification.py:260-280`) takes a materialized `pl.Series` and calls `.quantile()` in a Python loop — one pass per percentile. This is both eager-only and O(n × p) where p is the number of percentiles.

**Current code:**
```python
def percentile_thresholds(series: pl.Series, percentiles: list[float]) -> list[float]:
    return [
        float(series.quantile(p / 100, interpolation="linear"))
        for p in percentiles
    ]
```

**Proposed change:** A new expression-based version that computes all quantiles in one pass and works in lazy/streaming mode.

```python
def percentile_thresholds_expr(
    col_name: str, percentiles: list[float]
) -> pl.Expr:
    """Compute threshold values from percentiles as a lazy expression.

    Returns a struct expression with one field per percentile.
    """
    return pl.struct([
        pl.col(col_name).quantile(p / 100, interpolation="linear").alias(f"p{p:g}")
        for p in percentiles
    ]).alias("percentile_thresholds")
```

**Minimum Polars:** 1.35.0 (multi-quantile optimization for single-pass computation). The expression itself works on earlier versions but won't get the single-pass optimization.

**Constraint:** This changes the return type from `list[float]` to `pl.Expr`. The old function would need to be kept for backwards compatibility or the API would need a breaking change. Recommend keeping the old function and adding the new one as `percentile_thresholds_expr()`.

#### 2.2 Use `Expr.item()` for scalar extraction in threshold sweep

**What:** Add a `threshold_sweep_df()` function that returns a tidy DataFrame (one row per threshold) instead of a list of expressions, using `.item()` for safe scalar extraction.

**Why:** The current `threshold_sweep()` returns `list[pl.Expr]`, which is flexible but requires the user to unpack results manually. A DataFrame-returning variant is more ergonomic for the common case of comparing metrics across thresholds.

**Current code (`classification.py:228-257`):** Returns a list of expressions, one per threshold.

**Proposed addition:**
```python
def threshold_sweep_df(
    metric_fn: _MetricFn,
    df: pl.DataFrame | pl.LazyFrame,
    target: str,
    prob: str,
    thresholds: list[float],
    weight: str | None = None,
) -> pl.DataFrame:
    """Sweep a metric across thresholds and return a tidy DataFrame.

    Returns a DataFrame with columns: threshold, metric_value.
    """
    exprs = threshold_sweep(metric_fn, target, prob, thresholds, weight)
    if isinstance(df, pl.LazyFrame):
        row = df.select(*exprs).collect()
    else:
        row = df.select(*exprs)
    return pl.DataFrame({
        "threshold": thresholds,
        "value": [row[0, i] for i in range(len(thresholds))],
    })
```

**Minimum Polars:** 1.35.0 (for `.item()` availability in downstream user code and documentation).

#### 2.3 Add `optimal_threshold()` using `max_by`

**What:** A new function that finds the threshold maximizing a given metric, returning both the threshold and the metric value.

**Why:** Finding the optimal threshold is one of the most common follow-ups after `threshold_sweep()`. Today users must sort the results and pick the best row manually. With `max_by` (v1.37), this can be a single expression.

**Proposed addition:**
```python
def optimal_threshold(
    metric_fn: _MetricFn,
    df: pl.DataFrame | pl.LazyFrame,
    target: str,
    prob: str,
    thresholds: list[float],
    weight: str | None = None,
) -> tuple[float, float]:
    """Find the threshold that maximizes the given metric.

    Returns (best_threshold, best_metric_value).
    """
    sweep_df = threshold_sweep_df(metric_fn, df, target, prob, thresholds, weight)
    best = sweep_df.select(
        pl.col("threshold").max_by("value"),
        pl.col("value").max(),
    )
    return best["threshold"].item(), best["value"].item()
```

**Minimum Polars:** 1.37.0 (for `max_by`).

**Fallback for 1.35.0:** Use `sort("value", descending=True).head(1)` instead of `max_by`. Less elegant but equivalent.

#### 2.4 Add `rolling_roc_auc()` and `rolling_average_precision()`

**What:** Windowed versions of ROC AUC and average precision that compute the metric over a sliding window of observations.

**Why:** In production ML monitoring, you want to track metric drift over time. The current `roc_auc()` computes a single global value. A rolling version would compute AUC over the last N observations at each point, enabling time-series metric monitoring.

**How it would work:** This can't use `rolling_rank()` directly for AUC (the Mann-Whitney statistic requires a custom rolling computation), but it can use the new `list.agg()` pattern:

```python
def rolling_roc_auc(
    target: str, score: str, window_size: int
) -> pl.Expr:
    """Compute ROC AUC over a rolling window.

    Returns a column with one AUC value per row, computed over the
    preceding `window_size` rows.
    """
    # Collect target and score into rolling lists
    target_list = pl.col(target).rolling(index_column="index", period=f"{window_size}i")
    score_list = pl.col(score).rolling(index_column="index", period=f"{window_size}i")
    # Compute AUC within each list using list.agg()
    # ... (implementation would use the Mann-Whitney U statistic on list elements)
```

**Minimum Polars:** 1.35.0 (for `list.agg()`). This is a complex feature — it may require the `arr.eval()` pattern or a UDF fallback for the Mann-Whitney computation within each window.

**Constraint:** This is the most architecturally complex improvement. The Mann-Whitney statistic doesn't decompose neatly into a rolling operation. Two implementation strategies:

1. **List-based** (v1.35+): Collect window elements into lists, then use `list.agg()` to compute AUC per list. Clean but potentially slow for large windows.
2. **Approximate** (v1.35+): Use `rolling_rank()` as a building block for an approximate rolling AUC. Faster but not exact.

#### 2.5 Improve `log_loss()` with expression-based log

**What:** Use the new `Expr.log(base_expr)` capability (v1.33) to make log loss more flexible for multi-class extensions.

**Why:** The current `log_loss()` implementation uses natural log (`prob_clipped.log()`) and manual `1 - prob` computation. While correct for binary cross-entropy, extending to multi-class or custom-base entropy would benefit from the expression-based log.

**Current code (`log_loss.py:23-29`):**
```python
log_prob = prob_clipped.log()
log_1_minus_prob = (1 - prob_clipped).log()
per_sample = -(target_float * log_prob + (1 - target_float) * log_1_minus_prob)
```

**Change:** No change to the binary version (it already works). But the expression-based log enables a cleaner multi-class `categorical_cross_entropy()`:

```python
def categorical_cross_entropy(
    target: str, prob_columns: list[str], weight: str | None = None
) -> pl.Expr:
    """Multi-class cross-entropy loss.

    Args:
        target: Column with integer class labels (0 to K-1).
        prob_columns: List of K columns, each containing P(class=k).
        weight: Optional sample weight column.
    """
    # Use list column + list.agg() to select the correct class probability
    # per row, then take -log
    ...
```

**Minimum Polars:** 1.35.0 (for `list.agg()` to select the target-class probability from a list of class probabilities).

### Tier 3 — Aggressive version bump (polars >= 1.37.0+)

These changes require raising the minimum to 1.37.0 or later. This drops support for installations more than ~6 months old.

#### 3.1 `confusion_matrix()` as a streaming lazy expression

**What:** A new function that returns a confusion matrix as a DataFrame, using lazy `pivot()` (v1.36) for streaming support.

**Why:** Confusion matrices are fundamental to classification evaluation. Currently polarbear only exposes the scalar metrics derived from the confusion matrix (precision, recall, etc.), not the matrix itself. Users who want the raw TP/FP/FN/TN counts must compute them manually.

**Proposed implementation:**
```python
def confusion_matrix(
    target: str, prob: str, threshold: float = 0.5
) -> pl.LazyFrame:
    """Compute a confusion matrix as a LazyFrame.

    Returns a 2×2 DataFrame with actual classes as rows, predicted as columns.
    Can be collected with engine='streaming' for large datasets.
    """
    return (
        pl.LazyFrame()  # would need the source data
        .with_columns(
            (pl.col(prob) >= threshold).cast(pl.Int8).alias("predicted")
        )
        .group_by(pl.col(target).alias("actual"), "predicted")
        .agg(pl.len().alias("count"))
        .pivot(on="predicted", index="actual", values="count")
        .fill_null(0)
    )
```

**Minimum Polars:** 1.36.0 (for `LazyFrame.pivot()`).

**Design constraint:** Unlike other polarbear functions that return `pl.Expr`, this would return a `pl.LazyFrame` — a different API shape. It needs the source data as input (a LazyFrame, not just column names), which breaks the current pattern of returning expressions that users plug into `.select()`. Two options:
1. Accept a `LazyFrame` argument and return a `LazyFrame` (different API pattern).
2. Return a struct expression with fields `tp`, `fp`, `fn`, `tn` (consistent with existing pattern, but not a traditional matrix layout).

#### 3.2 Per-group metrics with `.over()` and `PartitionBy`

**What:** Enable computing metrics partitioned by a grouping column (e.g., AUC per model version, precision per demographic group).

**Why:** Fairness auditing, A/B testing, and model comparison all require computing the same metric across subgroups. Today users must filter/group manually and call polarbear on each subset.

**Proposed API:**
```python
# AUC per model version
df.select(
    pb.roc_auc("target", "score").over("model_version")
)

# Classification metrics per demographic group
df.select(
    pb.precision("target", "prob").over("group")
)
```

**Challenge:** Polarbear's metrics are scalar aggregations (they reduce a column to a single value via `.sum()`, `.mean()`, etc.). The `.over()` modifier turns an aggregation into a window function that repeats the aggregated value for each row in the partition. This works naturally for simple aggregations like `sum().over("group")`, but polarbear metrics use complex multi-expression compositions (e.g., ROC AUC uses `rank()`, `sort_by()`, `cum_sum()` in sequence).

**Implementation strategy:**
- For simple metrics (MAE, MSE, RMSE, Brier score) that are just `mean()` or `sum()` of per-row values: `.over()` works directly. These can be supported immediately.
- For threshold metrics (precision, recall, F1) that use `_confusion_components()`: The TP/FP/FN/TN sums need to be computed per group. This requires refactoring `_confusion_components()` to accept an optional `partition_by` parameter.
- For ranking metrics (ROC AUC, average precision) that use `sort_by()` and `cum_sum()`: These are fundamentally harder because the sort and cumulative operations must be scoped to each group. This may require a `group_by().map_groups()` approach rather than pure expressions.

**Minimum Polars:** 1.37.0 (for `PartitionBy` API) or 1.30.0 (for `.over()` without partition_by).

**Constraint:** This is the highest-impact but also highest-effort improvement. It would likely require refactoring the internal architecture of most metrics to accept a group-by context.

#### 3.3 Float16 input handling

**What:** Ensure all metrics work correctly when input columns are Float16.

**Why:** ML models increasingly output predictions in fp16 (especially GPU-trained models). The current implementation casts all inputs to Float64 (`pl.col(target).cast(pl.Float64)`), which is correct — Float16 inputs will be upcast automatically. But this should be explicitly tested and documented.

**Current code (in every metric):**
```python
target_float = pl.col(target).cast(pl.Float64)
```

**Change:** Add Float16 test cases to the property-based test suite. No code change needed (the cast already handles it), but explicit coverage prevents regressions.

**Minimum Polars:** 1.36.0 (for Float16 dtype availability).

### Tier 4 — New metric categories enabled by new Polars features

These are entirely new metrics that become practical only because of recently added Polars capabilities.

#### 4.1 Calibration metrics (reliability diagram data)

**What:** `calibration_curve()` that bins predicted probabilities and computes observed frequency per bin — the data behind a reliability diagram.

**Why:** Brier score measures calibration as a single number, but practitioners need the curve to diagnose *where* the model is miscalibrated. This requires binning, group-by, and aggregation — all of which are now streamable.

**Implementation sketch:**
```python
def calibration_curve(
    target: str, prob: str, n_bins: int = 10, strategy: str = "uniform"
) -> pl.LazyFrame:
    """Compute calibration curve data.

    Returns a LazyFrame with columns: bin_center, observed_frequency,
    mean_predicted, count.
    """
    # Use cut() or qcut() for binning, then group_by + agg
    ...
```

**Minimum Polars:** 1.35.0 (for streaming group-by with all needed operations).

**Constraint:** Like `confusion_matrix()`, this returns a LazyFrame, not an Expr.

#### 4.2 Ranking metrics (NDCG, MRR) using `rolling_rank()`

**What:** Normalized Discounted Cumulative Gain (NDCG) and Mean Reciprocal Rank (MRR) — standard metrics for search and recommendation systems.

**Why:** `rolling_rank()` (v1.35) provides the ranking primitive needed to compute position-based discounts efficiently. Previously, implementing NDCG required manual sort-and-index operations.

**Implementation sketch:**
```python
def ndcg(target: str, score: str, k: int | None = None) -> pl.Expr:
    """Normalized Discounted Cumulative Gain.

    Args:
        target: Column with relevance scores (0, 1, or graded).
        score: Column with predicted scores.
        k: Truncation rank (compute NDCG@k). None for full ranking.
    """
    # Sort by score descending, compute DCG using rank-based discounts
    # DCG = Σ (2^rel - 1) / log2(rank + 1)
    # NDCG = DCG / ideal_DCG
    ...
```

**Minimum Polars:** 1.35.0 (for `rolling_rank()`).

#### 4.3 Distribution metrics (KL divergence, JS divergence)

**What:** Kullback-Leibler and Jensen-Shannon divergence between predicted and observed distributions.

**Why:** With `Expr.log(base_expr)` (v1.33) and streaming support for all the underlying operations, these can be implemented efficiently as pure expressions.

```python
def kl_divergence(p: str, q: str, weight: str | None = None) -> pl.Expr:
    """KL(P || Q) = Σ P(x) * log(P(x) / Q(x))"""
    p_col = pl.col(p).cast(pl.Float64)
    q_col = pl.col(q).cast(pl.Float64).clip(1e-15, 1.0)
    per_sample = p_col * (p_col / q_col).log()
    ...
```

**Minimum Polars:** 1.33.0.

### Decision matrix: recommended version strategy

Given the improvements above, here is the tradeoff:

| Strategy | Min Polars | What you gain | What you lose |
|----------|-----------|---------------|---------------|
| **Conservative** — keep 1.0.0 | 1.0.0 | Maximum compatibility. 19 existing metrics unchanged. | No new features. Can only add docs/examples for streaming. |
| **Moderate** — bump to 1.35.0 | 1.35.0 | `item()`, `list.agg()`, `rolling_rank()`, Decimal, lazy percentile thresholds, rolling metrics, NDCG/MRR, calibration curve. ~8 improvements. | Drops users on Polars < 1.35 (released ~9 months ago). |
| **Aggressive** — bump to 1.37.0 | 1.37.0 | Everything in moderate, plus `max_by`, `PartitionBy`, per-group metrics, `optimal_threshold()`. ~12 improvements. | Drops users on Polars < 1.37 (released ~5 months ago). |
| **Split** — keep 1.0.0 core, 1.35.0+ extras | 1.0.0 / 1.35.0 | Best of both. Core metrics work everywhere. New functions require newer Polars and raise `ImportError` with a helpful message on older versions. | More complex codebase. Version-gated imports. Two test matrices. |

**Recommendation:** The **moderate** bump to `polars >= 1.35.0` offers the best balance. It unlocks the most impactful improvements (lazy percentile thresholds, safe `.item()` extraction, rolling/ranking metrics, calibration curves) while only requiring a version that is nearly a year old. The `max_by`-based `optimal_threshold()` can use the sort-based fallback until a future bump to 1.37.0.

---

## 11. Update: Releases 1.40–1.41 (April–May 2026)

Extending the analysis past the original 1.39 cutoff to the latest release at
the time of writing, **polars 1.41.2** (May 29, 2026). Both releases are
dominated by streaming-engine work, so most gains are **transparent** — the same
polarbear expression runs faster on a newer polars with no code change (measured
in this repo's benchmarks: e.g. Log Loss @100k went 4.7x → 7.9x vs scikit-learn
from polars 1.0.0 → 1.41).

### Polars 1.40.0 (April 18, 2026)

**Theme: more operations lowered to streaming primitives.**

Metrics-relevant changes:
- **Streaming `cov` and `corr`** (#27008) — a future correlation metric would
  scale out-of-core for free.
- **`entropy` lowered to streaming reductions** (#27174), **`skew` / `kurtosis`
  streaming** (#27176) — information- and distribution-shape metrics stream.
- **`cut` now outputs `Enum` and is marked elementwise** (#27173) — directly
  helps the Tier-4 calibration-curve sketch: probability binning becomes a cheap
  elementwise op that fuses into streaming pipelines.
- **`over()` lowered to streaming primitives** (#27303) — windowed/grouped metric
  expressions stream.
- **`group_by()` without key expressions** (#27141) — whole-frame aggregation
  without a dummy key; minor ergonomics for scalar metrics.
- `is_unique` on list/array dtypes (#27290); `pl.merge_sorted` over multiple
  frames (#27014).

Deprecation (does **not** affect polarbear): the DataFrame interchange protocol.

### Polars 1.41.0 (May 22, 2026)

**Theme: the streaming engine is now stabilized.**

Metrics-relevant changes:
- **Streaming engine stabilized** (#27497) — no longer just the default path but
  a stable, supported API. This is the strongest form yet of §9's recommendation
  to document the `.collect(engine="streaming")` path: it is now the blessed way
  to compute metrics on larger-than-memory data.
- **`float16` stabilized** (#27607) — fp16 prediction columns are first-class.
  polarbear casts inputs to Float64 internally, so fp16 inputs are accepted
  without friction (relevant to GPU / mixed-precision pipelines).
- **Nested common subplan elimination (CSE)** (#27340) — repeated sub-expressions
  in a plan are computed once. Relevant to expression-heavy metrics that
  reference the same confusion-matrix components many times; the engine now
  dedups that work automatically — a general version of the manual flattening we
  applied to `cohens_kappa`.
- `LazyFrame.gather` (#27501), `Expr.is_empty` (#27583) — minor building blocks.

Deprecation (does **not** affect polarbear): the global `StringCache`.

### Net effect on the version strategy

Nothing in 1.40–1.41 changes the earlier recommendation. Both deprecations are in
areas polarbear doesn't touch — confirmed: the suite passes against 1.41.2 with
`filterwarnings = ["error::DeprecationWarning"]`. The headline is that the
streaming engine — already the most impactful "free" win — is now **stable** as
of 1.41, further strengthening the case to document the streaming path rather
than chase version-gated micro-optimizations. The "moderate bump to 1.35" path
for new metric categories (NDCG/MRR, calibration, KL/JS) still stands; 1.40–1.41
additionally make streaming `cov`/`corr`/`entropy` available, so a future
correlation- or entropy-based metric would scale out-of-core out of the box.

---

## Further Reading

- [Polars User Guide](https://docs.pola.rs/user-guide/) — comprehensive documentation
- [Polars API Reference](https://docs.pola.rs/api/python/stable/reference/) — full Python API
- [Polars GitHub Releases](https://github.com/pola-rs/polars/releases) — detailed per-version changelogs
- [Polars Blog](https://pola.rs/posts/) — deep dives on architecture and features
- [Polars Cloud Docs](https://docs.pola.rs/polars-cloud/) — distributed execution
- [GPU Support Guide](https://docs.pola.rs/user-guide/gpu-support/) — NVIDIA RAPIDS integration
- [Categorical Refactor Blog](https://pola.rs/posts/categoricals-refactor/) — migration guide for the new Categories system
- [UVM Blog Post](https://pola.rs/posts/uvm-larger-than-ram-gpu/) — larger-than-VRAM GPU processing
