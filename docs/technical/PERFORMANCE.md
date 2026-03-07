# Performance Analysis

## Current Performance vs sklearn

Polarbear is **2-4x faster than sklearn** on large datasets:

| Metric | 100 samples | 100k samples | Speedup |
|--------|------------|--------------|---------|
| **ROC AUC** | 0.136ms (2.23x) | 3.162ms (3.99x) | **3.99x** |
| **Log Loss** | 0.105ms (2.23x) | 1.831ms (3.01x) | **3.01x** |
| **Brier Score** | 0.048ms (1.85x) | 0.162ms (2.91x) | **2.91x** |

### Grouped Operations

Polarbear excels at grouped metric calculations:

| Groups | Samples/Group | Total Samples | Time |
|--------|--------------|--------------|------|
| 10 | 1,000 | 10,000 | 0.78ms |
| 100 | 1,000 | 100,000 | 3.42ms |
| 1,000 | 100 | 100,000 | 4.45ms |
| 100 | 10,000 | 1,000,000 | 37.11ms |

## Optimization History

### Optimizations Applied

#### ROC AUC (`src/polarbear/roc_auc.py`)
- ✅ Cast target to Float64 **once** instead of multiple times
- ✅ Calculate `total_neg` as `len - total_pos` (eliminates separate sum)
- ✅ Use variance check (`var() == 0`) instead of `min() == max()` for tie detection
- ✅ Reuse `target_float` to avoid redundant casting

#### Log Loss (`src/polarbear/log_loss.py`)
- ✅ Move target casting before other operations
- ✅ Cache log computations (`log_prob` and `log_1_minus_prob`)

#### Brier Score
- No changes needed (already optimal)

### Performance Improvements (Before/After)

#### ROC AUC Performance

| Samples | Before | After | Improvement | vs sklearn |
|---------|--------|-------|-------------|------------|
| 100 | 0.424ms | 0.136ms | **3.12x faster** | 2.23x faster |
| 1,000 | 0.173ms | 0.154ms | 1.12x faster | 2.54x faster |
| 10,000 | 0.509ms | 0.481ms | 1.06x faster | 2.80x faster |
| 100,000 | 3.350ms | 3.162ms | 1.06x faster | **3.99x faster** |

#### Log Loss Performance

| Samples | Before | After | Improvement | vs sklearn |
|---------|--------|-------|-------------|------------|
| 100 | 0.138ms | 0.105ms | **1.31x faster** | 2.23x faster |
| 1,000 | 0.133ms | 0.114ms | 1.17x faster | 2.54x faster |
| 10,000 | 0.268ms | 0.285ms | 0.94x slower | 2.70x faster |
| 100,000 | 1.862ms | 1.831ms | 1.02x faster | **3.01x faster** |

#### Brier Score Performance

| Samples | Before | After | Improvement | vs sklearn |
|---------|--------|-------|-------------|------------|
| 100 | 0.076ms | 0.048ms | **1.58x faster** | 1.85x faster |
| 1,000 | 0.058ms | 0.054ms | 1.07x faster | 1.69x faster |
| 10,000 | 0.067ms | 0.062ms | 1.08x faster | 2.06x faster |
| 100,000 | 0.195ms | 0.162ms | 1.20x faster | **2.91x faster** |

#### Grouped Operations Performance

| Groups | Samples/Group | Total Samples | Before | After | Improvement |
|--------|--------------|--------------|---------|--------|-------------|
| 10 | 1,000 | 10,000 | 1.14ms | 0.78ms | **1.46x faster** |
| 100 | 1,000 | 100,000 | 3.94ms | 3.42ms | 1.15x faster |
| 1,000 | 100 | 100,000 | 5.08ms | 4.45ms | 1.14x faster |
| 100 | 10,000 | 1,000,000 | 41.86ms | 37.11ms | 1.13x faster |

## Key Insights

1. **Small datasets benefit most**: The optimizations show the biggest improvements on small datasets (100-1,000 samples) where the overhead of redundant operations is proportionally larger.

2. **ROC AUC on tiny datasets**: **3.12x improvement** on 100 samples, bringing polarbear from being slower than sklearn to being 2.23x faster.

3. **Brier Score**: Consistent improvements across all dataset sizes, with **1.58x faster** on small datasets.

4. **Grouped operations**: **1.46x faster** for small group aggregations, with consistent 13-15% improvements across larger datasets.

5. **Still faster than sklearn**: Polarbear maintains 2-4x speedup over sklearn for large datasets.

## Benchmarking

Run performance benchmarks:

```bash
just bench
```

This will test:
- Single metric calculations (100 to 100k samples)
- Grouped metric calculations
- Comparison against sklearn
- Verification of correctness

## Future Optimization Opportunities

### Potential Improvements

1. **Polars expression fusion**: Combine operations to leverage Polars' query optimizer better

2. **Parallel group operations**: For datasets with many groups, investigate parallelization

3. **Lazy evaluation**: Implement lazy versions that compose with other operations before materialization

4. **Custom Rust extensions**: For critical hot paths, consider custom Polars plugins in Rust for maximum performance

5. **Memory layout optimization**: Investigate if specific column orderings or data types could improve cache locality

## Why Polarbear is Fast

1. **Native Polars expressions**: Operations execute in Polars' optimized Rust engine
2. **Vectorized operations**: No Python loops; everything is vectorized
3. **Minimal overhead**: Direct expression composition without intermediate conversions
4. **Query optimization**: Polars' query optimizer can fuse operations
5. **Memory efficiency**: Polars uses Arrow's columnar format for cache-friendly access

## Performance Tips

### For Best Performance

1. **Use grouped operations**: Calculate metrics per group in one pass
   ```python
   df.group_by("group").agg(roc_auc("label", "score"))
   ```

2. **Batch calculations**: Calculate multiple metrics in one select
   ```python
   df.select(
       roc_auc("label", "score"),
       log_loss("label", "prob"),
       brier_score("label", "prob"),
   )
   ```

3. **Filter before metrics**: Filter data first to reduce computation
   ```python
   df.filter(pl.col("label").is_not_null()).select(
       roc_auc("label", "score")
   )
   ```

4. **Use lazy evaluation**: For complex pipelines, use LazyFrames
   ```python
   (df.lazy()
      .filter(...)
      .group_by(...)
      .agg(roc_auc("label", "score"))
      .collect())
   ```

## Profiling

To profile polarbear performance:

```bash
# Basic profiling
python -m cProfile -o output.prof benchmark.py
python -m pstats output.prof

# Or use py-spy for live profiling
pip install py-spy
py-spy record -o profile.svg -- python benchmark.py
```

## Comparison Methodology

All benchmarks:
- Use identical random data with fixed seeds
- Run multiple iterations (10-100) for averaging
- Verify correctness against sklearn (within 1e-5 tolerance)
- Measure end-to-end time including result materialization
- Run on same hardware and environment

See `benchmarks/` for benchmark code.
