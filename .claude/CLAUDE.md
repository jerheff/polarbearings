# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polarbear is a high-performance machine learning metrics library implemented as native Polars expressions. It provides 2-4x faster metric computation compared to scikit-learn for large datasets.

**Key Features:**
- 19 metrics: ranking, probabilistic, classification, and regression
- Thoroughly tested against scikit-learn with property-based testing
- Full type hints and pyright strict mode
- Supports Polars 1.0.0+

## Quick Commands

```bash
just                 # List all available commands
just setup           # Install dependencies + prek hooks
just test            # Run full test suite
just test-fast       # Quick test run
just quality         # Run linting and type checking
just ci              # Run all CI checks locally
just bench           # Run performance benchmarks
just pre-commit      # Quick pre-commit check (auto-fix + format + test)
```

## Development Workflow

### Initial Setup
```bash
uv sync  # Install all dependencies including dev tools
```

### Running Tests
```bash
just test                    # Full test suite with verbose output
just test-fast              # Quick test run
just test-cov               # With coverage report
just test-compat            # Test against min, mid, and latest Polars versions
```

### Code Quality
```bash
just lint                   # Check code style (ruff)
just lint-fix              # Auto-fix style issues
just format                # Format code
just type-check            # Type checking (pyright strict mode)
just quality               # Run all quality checks (lint + type-check)
```

### Performance
```bash
just bench                  # Run benchmarks against scikit-learn
```

## Architecture

**Source Structure:**
- `src/polarbear/__init__.py` - Public API exports
- `src/polarbear/roc_auc.py` - ROC AUC (Mann-Whitney U statistic)
- `src/polarbear/average_precision.py` - Average precision score
- `src/polarbear/log_loss.py` - Log loss / binary cross-entropy
- `src/polarbear/brier_score.py` - Brier score
- `src/polarbear/classification.py` - Threshold-based classification metrics (precision, recall, F1, fbeta, specificity, accuracy, balanced accuracy, MCC, Cohen's kappa, threshold sweep, percentile thresholds)
- `src/polarbear/regression.py` - Regression metrics (MAE, MSE, RMSE, R², MAPE)

**Metric Implementation Pattern:**
All metrics are implemented as Polars expressions that:
1. Take column name strings as input (e.g., `target`, `score`/`prob`/`pred`)
2. Return a named `pl.Expr`
3. Support optional sample weights via a `weight` parameter
4. Return null for undefined cases (e.g., single-class ROC AUC)

**Testing Strategy:**
- Unit tests for basic functionality
- Property-based tests with Hypothesis for random data
- Compatibility tests against scikit-learn
- Multi-version Polars compatibility tests (1.0.0, 1.24.0, 1.38.1)

## Important Files

- `justfile` - Task runner definitions
- `pyproject.toml` - Project configuration, dependencies, tool settings
- `docs/guides/TESTING.md` - Testing documentation
- `docs/technical/PERFORMANCE.md` - Performance analysis
- `docs/technical/POLARS_COMPATIBILITY.md` - Version compatibility details

## Code Style

- **Python Version:** 3.11+ (strict requirement)
- **Line Length:** 100 characters
- **Type Checking:** Pyright strict mode enabled
- **Linting:** Ruff with pycodestyle, pyflakes, isort, flake8-bugbear, comprehensions, pyupgrade
- **Formatting:** Double quotes, space indentation

## Adding New Metrics

When implementing a new metric:

1. Add the implementation to the appropriate module in `src/polarbear/` (classification.py, regression.py, or a new file for a new category)
2. Export it in `src/polarbear/__init__.py`
3. Write unit tests in `tests/test_<metric_name>.py`
4. Add property-based tests with Hypothesis
5. Add compatibility tests against scikit-learn
6. Update README.md with examples and documentation
7. Run `just ci` to verify all checks pass

## Dependencies

**Core:**
- polars >= 1.0.0

**Development:**
- pytest (testing framework)
- hypothesis[numpy] (property-based testing)
- pytest-benchmark (performance benchmarks)
- pytest-cov (coverage reporting)
- scikit-learn (compatibility testing)
- numpy (test utilities)

**Linting:**
- pyright (type checking)
- ruff (linting and formatting)

## Common Tasks

**Before committing:**
```bash
just pre-commit  # Auto-fixes lint issues, formats code, runs fast tests
```

**Full CI check locally:**
```bash
just ci  # Runs quality checks + full test suite
```

**Test compatibility:**
```bash
just test-compat  # Test against min, mid, and latest Polars versions
```

**Clean up artifacts:**
```bash
just clean  # Remove __pycache__, .pytest_cache, etc.
```
