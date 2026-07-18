# Polars Version Compatibility

## Current Requirements

**Minimum Polars Version**: `1.0.0`
**CI Tested Versions**: `1.0.0`, `1.24.0`, `1.42.0`

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
4. CI tests verify compatibility against 1.0.0, 1.24.0, and 1.42.0

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
