# Findings Log

Performance and approach experiments: what we tried, what we measured, and what we
decided. The point is to stop re-running the same experiment from scratch and to
know *when* a rejected or deferred idea deserves another look (a new Polars
release, an API leaving experimental, a user report).

This is not the feature backlog; new metrics and utilities live in
[FUTURE_IDEAS.md](FUTURE_IDEAS.md).

## How to use this file

- **One entry per idea**, newest first within each status section.
- **Status** is one of:
  - `adopted`: shipped (link the PR).
  - `pending`: we want it, nobody has done it yet.
  - `deferred`: plausible, but waiting on a decision or an external change.
  - `rejected`: measured and not worth it (keep the evidence so we don't redo it).
- **Revisit when** is the trigger to look again. When doing a periodic review,
  scan that line first: if the trigger has fired, re-measure and update the entry
  (bump the date, keep the old numbers in the evidence for comparison).
- **Benchmark through the real public function**, not an isolated primitive.
  Entry F-003 is the cautionary tale.
- Note the machine. Unless stated, numbers below are from a 4-core cloud container.

### Entry template

```markdown
### F-NNN: Short title

| Status | Date | Polars tested | Revisit when |
|---|---|---|---|
| pending | YYYY-MM-DD | 1.0.0, 1.44.2, 2.1.0 | trigger |

**Idea.** One or two sentences.

**Evidence.** Numbers, sizes, versions, and how measured.

**Decision / next step.** What we did or will do, with PR links.
```

## Summary

| ID | Idea | Status | Revisit when |
|---|---|---|---|
| [F-001](#f-001-pin-internal-collects-to-the-in-memory-engine-on-polars-2x) | `engine="in-memory"` on internal collects (2.x) | pending | Polars streaming engine gains CSE / closes the gap |
| [F-002](#f-002-update-the-readme-go-lazy-advice-for-polars-2x) | Fix README "go lazy" advice for 2.x | pending | Ships with F-001 |
| [F-003](#f-003-search_sorted-for-calibration-bin-ids) | `search_sorted` for calibration bin ids | rejected | Polars makes `search_sorted` elementwise, or the 1.0 floor is raised |
| [F-004](#f-004-bin_intervals-fast-path-for-calibration-bin-ids) | `bin_intervals` fast path for many bins (2.x) | deferred | Jeremy decides; or `bin_intervals` leaves experimental |
| [F-005](#f-005-multiclass-averaging-as-separate-functions) | Multiclass averaging as separate functions | pending | PR merged |
| [F-006](#f-006-smaller-ideas-that-showed-no-gain) | Smaller ideas with no gain | rejected | A Polars release notes a relevant optimizer change |

## Pending

### F-001: Pin internal collects to the in-memory engine on Polars 2.x

| Status | Date | Polars tested | Revisit when |
|---|---|---|---|
| pending | 2026-11-15 | 1.0.0, 1.24.0, 1.44.2, 2.1.0 | The streaming engine closes the gap (re-time `bootstrap_ci` on each 2.x minor) |

**Idea.** Polars 2.x makes `collect()` default to the streaming engine. For our
workloads (many small aggregations over one frame) that is slower. Pass
`engine="in-memory"` on the library's internal `collect()` calls.

**Evidence.**

- `bootstrap_ci`, whole frame, B=200, n=100k: **14.5 s** with the 2.1 default vs
  **3.6 s** with `engine="in-memory"` (~4x). Linear in B either way. The `by=` path
  showed no difference.
- 10-metric suite, 2M rows on 2.1: eager `select` 917 ms; `lazy().collect()`
  (streaming default) 1124 ms; `collect(engine="in-memory")` 426 ms. Streaming
  does not deliver the CSE win that lazy gets in-memory.
- Version gotcha: `collect(engine=...)` does not exist on 1.0.0, and 1.24.0 accepts
  the kwarg but rejects `"in-memory"`. Must be feature-detected, not version-sniffed
  by guesswork.

**Decision / next step.** Recommended; not yet PR'd. Add a small internal
`_collect(lf)` helper that probes once and passes `engine="in-memory"` where
supported, then route every internal collect through it.

### F-002: Update the README "go lazy" advice for Polars 2.x

| Status | Date | Polars tested | Revisit when |
|---|---|---|---|
| pending | 2026-11-15 | 2.1.0 | Ships alongside F-001 |

**Idea.** The README says to run metric suites through
`df.lazy().select([...]).collect()` to get common-subexpression elimination. On
2.x that is now *slower* than eager unless the user passes
`collect(engine="in-memory")` (see F-001 numbers: 1124 ms lazy-default vs 917 ms
eager vs 426 ms lazy in-memory).

**Decision / next step.** Update the README performance section (and
`docs/technical/PERFORMANCE.md`) to recommend `collect(engine="in-memory")` on 2.x,
with a note that it isn't needed on 1.x. Keep the README Python blocks passing
under `just doctest` on the 1.0 floor.

### F-005: Multiclass averaging as separate functions

| Status | Date | Polars tested | Revisit when |
|---|---|---|---|
| pending | 2026-11-15 | 1.0.0, 2.1.0 | PR merged (then flip to adopted) |

**Idea.** Multiclass precision/recall/F1-style averaging built as five separate
functions taking explicit `labels`, rather than one function with an `average=`
switch.

**Evidence.** Eager `select` does no CSE, so the first version was 3-5x *slower*
than scikit-learn at 50 classes. Restructuring so each per-class aggregate is
referenced once or twice (NaN-as-undefined via `fill_nan(None)`,
`pl.mean_horizontal` to skip nulls) made it 3-10x faster than scikit-learn.

**Decision / next step.** PR open, awaiting Jeremy's go.

## Deferred

### F-004: `bin_intervals` fast path for calibration bin ids

| Status | Date | Polars tested | Revisit when |
|---|---|---|---|
| deferred | 2026-11-15 | 2.0, 2.1.0 | Jeremy decides; or `Expr.bin_intervals` leaves experimental |

**Idea.** On Polars 2.x, compute calibration `_bin_id` with
`Expr.bin_intervals(edges, labels=False, right_closed=True)` instead of the current
expression.

**Evidence.**

- Bit-identical bin ids, including NaN, null, inf, and values exactly on edges.
- Works inside `group_by().agg()` (it is elementwise with explicit edges), unlike
  `search_sorted` (F-003).
- Speed for `calibration_curve` and ECE/MCE (both `select` and grouped): ~10%
  **slower** at the default 10 bins, break-even ~15, 1.3-1.6x faster at 20-30,
  ~2x at 50.
- Quantile-strategy ECE edges are expressions, so that path is unaffected.

**Decision / next step.** Proposed: a gated fork used only on 2.x with >= 15 edges,
behind a feature probe (the API is experimental, so probe rather than check the
version). Awaiting Jeremy's call; low priority since the default 10 bins doesn't
benefit.

## Rejected

### F-003: `search_sorted` for calibration bin ids

| Status | Date | Polars tested | Revisit when |
|---|---|---|---|
| rejected | 2026-11-15 | 1.0.0, 1.36+, 1.44.2, 2.1.0 | Polars makes `search_sorted` elementwise (usable in `group_by`), or the 1.0 floor is raised |

**Idea.** Replace the calibration `_bin_id` expression with `search_sorted`
against the bin edges.

**Evidence.**

- Isolated binning step: ~4x faster. This is what made it look promising.
- End to end: default `calibration_curve` on 2.1 went from **14 ms to 62 ms**
  (2-4x slower, largely due to the streaming default). ECE at the default 10 bins
  was ~0.9x on 1.44+.
- Correctness: `search_sorted` is not elementwise, so ECE/MCE's `.over(binid)`
  fails inside `group_by().agg()` on Polars >= 1.36.
- Only clear win: Polars 1.0 floor with ~50 bins.
- Also tried an arithmetic `floor(p * n)` uniform binning: faster lazy, 3x slower
  eager at 10 bins (no eager CSE), and it had edge-rounding bugs.

**Decision.** Not shipped. **Lesson:** benchmark through the real public
functions, not an isolated step. A 4x primitive win turned into an end-to-end loss
plus a correctness bug.

### F-006: Smaller ideas that showed no gain

| Status | Date | Polars tested | Revisit when |
|---|---|---|---|
| rejected | 2026-11-15 | 1.44.2, 2.1.0 | A Polars release notes a relevant optimizer change |

- `arg_sort` + `gather` for weighted ROC AUC inside `bootstrap_ci`: no gain.
- Relying on `group_by` to CSE shared inputs across metrics: no gain.
- Checked 2.0/2.1 release notes: no eager CSE and no bitwise shift ops were added,
  so ideas depending on either stay blocked.

## Research tooling notes

- The **Polars docs MCP server** answers behavior questions well ("does X run in
  streaming?", "is this elementwise?").
- **Context7** is better for exact API signatures and kwargs across versions.
- Still verify against the actual installed versions (`just test-compat`); docs
  lag releases.
