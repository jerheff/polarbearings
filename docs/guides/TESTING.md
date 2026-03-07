# Testing Guide

## Quick Start (Using Just)

If you have [just](https://github.com/casey/just) installed, use these convenient shortcuts:

```bash
just test              # Run all tests
just test-fast         # Run tests (quiet mode)
just test-compat       # Test min, mid & latest Polars versions
just test-cov          # Run tests with coverage
just bench             # Run benchmarks
just quality           # Run linting and type checking
just ci                # Run all CI checks locally
just pre-commit        # Quick pre-commit check
```

See `just --list` for all available commands.

**Install just**: `brew install just` (macOS) or see [installation guide](https://github.com/casey/just#installation)

## Quick Testing (Manual)

### Run all tests
```bash
uv run pytest tests/ -v
```

### Run with coverage
```bash
uv run pytest --cov=src/polarbear --cov-report=term-missing tests/
```

## Testing Multiple Polars Versions Locally

Use the `just test-polars` and `just test-compat` commands:

```bash
# Test against min, ~1 year old, and latest Polars versions
just test-compat

# Test a specific version
just test-polars 1.0.0
```

### Supported Polars Versions

- **Minimum**: 1.0.0
- **CI tested**: 1.0.0, 1.24.0, 1.38.1

## Continuous Integration

CI automatically tests against multiple Polars versions on every push and PR:

- **Python versions**: 3.11, 3.12, 3.13, 3.14
- **Polars versions**: 1.0.0, 1.24.0, 1.38.1

See `.github/workflows/ci.yml` for the full matrix.

## Test Organization

```
tests/
├── test_aoc.py                    # ROC AUC tests with property-based testing
├── test_additional_metrics.py     # Log loss and Brier score tests
├── test_average_precision.py      # Average precision tests
├── test_classification.py         # Precision, recall, F1, accuracy, balanced accuracy
├── test_new_classification.py     # Specificity, fbeta, MCC, Cohen's kappa
├── test_regression.py             # MAE, MSE, RMSE tests
├── test_new_regression.py         # R², MAPE tests
├── test_weights.py                # Weighted metric tests
├── test_edge_cases.py             # Edge case tests for all metrics
└── test_degenerate_inputs.py      # Degenerate input tests
```

### Test Categories

1. **Property-based tests** (Hypothesis)
   - Generative testing with random data
   - Verifies mathematical properties
   - Compares against sklearn

2. **Unit tests**
   - Tests specific functionality
   - Tests edge cases (perfect predictions, worst case, etc.)
   - Verifies sklearn compatibility

3. **Edge case tests**
   - Minimal datasets (single positive/negative)
   - Imbalanced data
   - Tied scores
   - Large datasets (10k samples)
   - Grouped aggregations

4. **Weighted metric tests**
   - Uniform weights match unweighted
   - Weighted results match sklearn with sample_weight

## Linting and Type Checking

### Run all quality checks
```bash
# Linting
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/

# Type checking
uv run pyright src/polarbear
```

## Adding New Tests

When adding new tests:

1. **Place in appropriate file** or create a new `tests/test_<metric_name>.py`

2. **Follow naming convention**: `test_<description>`

3. **Verify sklearn compatibility** when applicable:
   ```python
   def test_new_metric():
       # Your test data
       df = pl.DataFrame({"label": [...], "score": [...]})

       # Test polarbear
       result = df.select(metric("label", "score")).to_series()[0]

       # Compare with sklearn
       sklearn_result = sklearn_metric([...], [...])
       assert result == pytest.approx(sklearn_result)
   ```

4. **Test edge cases**:
   - Empty data
   - Single samples
   - All same values
   - Extreme values
   - Null handling

5. **Run tests before committing**:
   ```bash
   just ci
   ```

## Pre-commit Hooks

Pre-commit hooks are configured in `prek.toml` and run automatically on commit.

```bash
# Install hooks
prek install

# Run manually against all files
uv run prek run --all-files
```
