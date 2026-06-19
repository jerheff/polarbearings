# Polars Version Compatibility

## Current Requirements

**Minimum Polars Version**: `1.0.0`
**CI Tested Versions**: `1.0.0`, `1.24.0`, `1.41.2`

## Syntax Compatibility Analysis

### Methods Used by Polarbearings

Our implementation uses the following Polars expression methods:

1. **`Expr.cast(dtype)`** - Standard casting, stable since early versions
2. **`Expr.sum()`** - Aggregation, stable since early versions
3. **`Expr.len()`** - Length calculation, stable since early versions
4. **`Expr.var()`** - Variance calculation, stable since early versions
5. **`Expr.rank(method='average')`** - Ranking with ties handling
6. **`Expr.clip(lower, upper)`** - Clipping values to range
7. **`Expr.log()`** - Natural logarithm
8. **`Expr.mean()`** - Mean aggregation
9. **`Expr.median()`** - Median aggregation
10. **`Expr.alias(name)`** - Expression aliasing
11. **`Expr.sort()`** / **`Expr.sort_by()`** - Sorting
12. **`Expr.cum_sum()`** - Cumulative sum
13. **`Expr.shift()`** - Shifting values
14. **`Expr.fill_null()`** / **`Expr.backward_fill()`** - Null handling
15. **`Expr.sqrt()`** - Square root
16. **`Expr.abs()`** - Absolute value

All of these are stable since Polars 1.0.

### Polars 1.0+ Breaking Changes Review

After reviewing Polars 1.0 breaking changes, **none affect our implementation**:

- No use of deprecated `replace()` methods
- No use of LazyFrame schema properties
- No use of EWM methods
- No decimal handling
- No JSON serialization
- No Series construction with mixed types
- No DataFrame constructor orientation issues
- No `cut`/`qcut` usage

## Version Support Strategy

We support `polars>=1.0.0` because:

1. All our methods are stable since Polars 1.0
2. No breaking changes affect us
3. This widens the user base without adding complexity
4. CI tests verify compatibility against 1.0.0, 1.24.0, and 1.41.2

## Testing Compatibility

```bash
# Test against all CI versions
just test-compat

# Test a specific version
just test-polars 1.0.0
```

## Current Assessment

- Our code uses modern, stable Polars syntax
- No deprecated methods
- No breaking changes from Polars 1.0+ affect us
- Safe to support Polars `>=1.0.0`
