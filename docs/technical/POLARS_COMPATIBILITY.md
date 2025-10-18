# Polars Version Compatibility

## Current Requirements

**Minimum Polars Version**: `1.0.0`
**Latest Tested Version**: `1.34.0`

## Syntax Compatibility Analysis

### Methods Used by Polarbear

Our implementation uses the following Polars expression methods:

1. **`Expr.cast(dtype)`** - Standard casting, stable since early versions
2. **`Expr.sum()`** - Aggregation, stable since early versions
3. **`Expr.len()`** - Length calculation, stable since early versions
4. **`Expr.var()`** - Variance calculation, stable since early versions
5. **`Expr.rank(method='average')`** - Ranking with ties handling
   - Current signature: `rank(method: RankMethod = 'average', *, descending: bool = False, seed: int | None = None)`
   - Using default `method='average'` parameter
   - **Status**: ✅ Stable, no deprecations
6. **`Expr.clip(lower, upper)`** - Clipping values to range
7. **`Expr.log()`** - Natural logarithm
8. **`Expr.mean()`** - Mean aggregation
9. **`Expr.alias(name)`** - Expression aliasing

### Polars 1.0+ Breaking Changes Review

After reviewing Polars 1.0 breaking changes, **none affect our implementation**:

- ✅ No use of deprecated `replace()` methods
- ✅ No use of LazyFrame schema properties
- ✅ No use of EWM methods
- ✅ No decimal handling
- ✅ No JSON serialization
- ✅ No Series construction with mixed types
- ✅ No DataFrame constructor orientation issues
- ✅ No `cut`/`qcut` usage
- ✅ No `clip` null propagation issues (we clip on non-null bounds)

## Version Support Strategy

### Current Approach: Minimum Version `>=1.27.1`

**Pros**:
- Ensures stability and consistency
- Users get latest performance improvements
- Simpler testing matrix
- No need for version-specific workarounds

**Cons**:
- May exclude some users on older Polars versions
- Tighter coupling to Polars release cycle

### Alternative Approach: Broader Version Support

To support a wider range of Polars versions (e.g., `>=1.0.0`), we would need to:

1. **Test against multiple versions**:
   ```toml
   # In CI/CD matrix
   polars-versions: ["1.0.0", "1.10.0", "1.20.0", "1.27.1", "latest"]
   ```

2. **Handle deprecations gracefully**:
   ```python
   # Example pattern (not currently needed)
   try:
       # New API
       result = expr.new_method()
   except AttributeError:
       # Fallback to old API
       result = expr.old_method()
   ```

3. **Check version at import time** (if needed):
   ```python
   import polars as pl

   POLARS_VERSION = tuple(map(int, pl.__version__.split('.')[:2]))
   if POLARS_VERSION < (1, 0):
       raise ImportError("polarbear requires polars >= 1.0.0")
   ```

## Recommendations

### Option 1: Keep Current Approach (Recommended)
**Stay with `polars>=1.27.1`**

**Rationale**:
- Our syntax is modern and stable
- Polars moves quickly; supporting old versions creates maintenance burden
- Users benefit from Polars' performance improvements
- No deprecated methods to worry about
- Clear, simple dependency

**Implementation**: No changes needed

### Option 2: Support Polars 1.0+
**Change to `polars>=1.0.0`**

**Rationale**:
- Broader user base
- Version 1.0 was a stability milestone
- Our methods are stable since 1.0

**Implementation**:
```toml
# pyproject.toml
dependencies = [
    "polars>=1.0.0",
]
```

**Required testing**: Add CI matrix to test versions 1.0.0, 1.10.0, 1.20.0, 1.27.1

### Option 3: Support Latest Only
**Change to `polars~=1.27` (compatible with 1.27.x)**

**Rationale**:
- Most conservative approach
- Guarantees exact behavior
- Least maintenance burden

**Not recommended** because it's too restrictive for users

## Current Assessment

✅ **Our code uses modern, stable Polars syntax**
✅ **No deprecated methods**
✅ **No breaking changes from Polars 1.0+ affect us**
✅ **Safe to support Polars `>=1.0.0` if desired**

## Recommendation: Support Polars >=1.0.0

Given that:
1. All our methods are stable since Polars 1.0
2. No breaking changes affect us
3. This widens the user base without adding complexity

**Suggested change**:
```toml
dependencies = [
    "polars>=1.0.0",
]
```

This provides a good balance between compatibility and maintainability.
