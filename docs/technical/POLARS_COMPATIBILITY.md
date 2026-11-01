---
description: Polars version support for polarbearings — minimum 1.0.0, the stable expression APIs it relies on, and the CI-tested floor, mid, last-1.x, and recent releases.
---
# Polars Version Compatibility

## Current Requirements

**Minimum Polars Version**: `1.0.0`
**CI Tested Versions**: `1.0.0` (floor), `1.24.0` (mid-1.x), `1.44.2` (last 1.x), and the recent release (`2.1.0`); a weekly run also tests the newest Polars release, which is exempt from the dependency cooldown

## Syntax Compatibility Analysis

### Expression APIs

Polarbearings is built entirely from Polars expression methods that are **stable
since Polars 1.0.0** — arithmetic and casts, aggregations (`sum`, `mean`, `median`,
`var`, `quantile`), `rank`/`sort`/`cum_sum`/`shift`, `clip`/`log`/`sqrt`/`abs`, null
handling, and the reshape/asof/window operators the curve helpers use (`explode`,
`concat_list`, `join_asof`, `over`, `set_sorted`, `reinterpret`).

This inventory is **non-exhaustive and not hand-maintained** — the **floor CI leg**
(the full suite run against polars 1.0.0 on every push) is the living guarantee that
no newer API slips in.

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

### Polars 2.0

None of 2.0's API removals (e.g. `collect(no_optimization=...)`) or deprecations
(`cut`/`qcut`) touch `src/`, and results are unchanged. Two *behavior* changes
matter:

- **`LazyFrame.collect()` defaults to the streaming engine** (`engine="auto"` fell
  back to in-memory on 1.x). Streaming runs many hashed `int_range(0, pl.len())`
  expressions in one `select` in quadratic time, which made `bootstrap_ci` ~400x
  slower at 40 replicates — minutes at the default `n_resamples=200`.
  `bootstrap_ci` now hashes a materialized `with_row_index` column instead:
  identical replicate weights, linear time on every version. When composing
  `bootstrap()` / `bootstrap_weight()` into your own *lazy* query, pass `row_index=`
  / `id_col=` such a column for the same reason. An eager `DataFrame.select` is
  unaffected.
- **Float aggregations can differ in the last bit between runs.** The streaming
  engine sums in parallel chunks, so e.g. a `calibration_curve` per-bin mean can
  vary by ~1 ulp from one `collect()` to the next. Compare float results with a
  tolerance (`polars.testing.assert_frame_equal`), not `.equals`.

### Version-gated fast paths (feature-detected)

Everything works on `polars>=1.0.0`. One hot path additionally detects a newer
Polars capability at runtime and switches to a faster implementation when present,
falling back to the floor-compatible form otherwise. Detection is by probing the
operation (not by parsing the version), so dev builds behave correctly. Both
branches are exercised by the CI matrix (the floor/mid legs hit the fallback, the
latest leg hits the fast path).

| Capability | Added in | Fallback (< version) | Fast path (>= version) |
|---|---|---|---|
| `_supports_over_in_agg` — elementwise `Expr.over` inside `group_by().agg()` (pola-rs/polars#25402) | 1.36.0 | ECE/MCE per-bin `filter` aggregations, `O(n · n_bins)` | ECE/MCE windowed bin means, single `O(n)` pass (≈4.5x at `n_bins=10`, ≈40x at `n_bins=50`) |

ECE/MCE remain composable inside `group_by().agg()` on every supported version —
only the internal implementation differs.

## Version Support Strategy

We support `polars>=1.0.0` because:

1. All our methods are stable since Polars 1.0
2. No breaking changes affect us
3. This widens the user base without adding complexity
4. CI tests verify compatibility against the floor (1.0.0), a mid-1.x release, the last 1.x release, and the recent 2.x release

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
- No breaking changes from Polars 1.0+ affect us; 2.0's streaming default needed
  one `bootstrap_ci` performance fix (see Polars 2.0 above)
- Safe to support Polars `>=1.0.0`, including 2.x
