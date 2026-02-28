# Testing Guide

## Quick Start (Using Just)

If you have [just](https://github.com/casey/just) installed, use these convenient shortcuts:

```bash
just test              # Run all tests
just test-fast         # Run tests (quiet mode)
just test-versions     # Test min and latest Polars versions
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

### Run benchmarks
```bash
uv run python benchmark.py
```

## Testing Multiple Polars Versions Locally

We provide two convenient ways to test against multiple Polars versions locally.

### Option 1: Python Script (Recommended)

The `test_versions.py` script provides the most flexible testing options:

#### Test minimum and latest versions (fastest)
```bash
uv run python test_versions.py --min-max
```

#### Test all default versions
```bash
uv run python test_versions.py
```

#### Test specific versions
```bash
uv run python test_versions.py --versions 1.0.0 1.20.0 1.34.0
```

#### Skip benchmarks for faster testing
```bash
uv run python test_versions.py --min-max --no-benchmark
```

#### Verbose output
```bash
uv run python test_versions.py --min-max --verbose
```

#### Get help
```bash
uv run python test_versions.py --help
```

### Option 2: Shell Script (Simpler)

The `test_versions.sh` script provides a simpler alternative:

#### Test minimum and latest (default)
```bash
./test_versions.sh
```

#### Test all default versions
```bash
./test_versions.sh --all
```

#### Test specific versions
```bash
./test_versions.sh 1.0.0 1.34.0
```

### Supported Polars Versions

- **Minimum**: 1.0.0
- **Latest tested**: 1.34.0
- **Default test versions**: 1.0.0, 1.10.0, 1.20.0, 1.34.0

## Manual Version Testing

If you prefer to test versions manually:

```bash
# Install specific version
uv pip install polars==1.0.0

# Run tests
uv run pytest tests/ -v

# Restore latest version
uv pip install polars==1.34.0
```

## Continuous Integration

Our CI automatically tests against multiple Polars versions on every push and PR:

- **Python versions**: 3.11, 3.12, 3.13
- **Polars versions**: 1.0.0, 1.10.0, 1.20.0, 1.34.0

See `.github/workflows/ci.yml` for the full matrix.

## Test Organization

```
tests/
├── test_aoc.py                 # ROC AUC tests with property-based testing
├── test_additional_metrics.py  # Log loss and Brier score tests
└── test_edge_cases.py          # Edge case tests for all metrics
```

### Test Categories

1. **Property-based tests** (`test_aoc.py`)
   - Uses Hypothesis for generative testing
   - Verifies mathematical properties
   - Compares against sklearn

2. **Unit tests** (`test_additional_metrics.py`)
   - Tests specific functionality
   - Tests edge cases (perfect predictions, worst case, etc.)
   - Verifies sklearn compatibility

3. **Edge case tests** (`test_edge_cases.py`)
   - Minimal datasets (single positive/negative)
   - Imbalanced data
   - Tied scores
   - Large datasets (10k samples)
   - Grouped aggregations

## Performance Testing

### Run benchmarks
```bash
uv run python benchmark.py
```

This tests performance across different dataset sizes:
- 100, 1,000, 10,000, 100,000 samples
- Compares against sklearn
- Tests grouped operations

### Expected Performance

All metrics should be **2-4x faster** than sklearn on large datasets (100k+ samples).

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

1. **Place in appropriate file**:
   - ROC AUC → `test_aoc.py`
   - Log loss/Brier score → `test_additional_metrics.py`
   - Edge cases → `test_edge_cases.py`

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
   uv run pytest tests/ -v
   uv run ruff check src/ tests/
   uv run pyright src/polarbear
   ```

## Troubleshooting

### Tests fail with specific Polars version

1. Check if the version is supported (>= 1.0.0)
2. Verify the exact error message
3. Check if Polars API changed in that version
4. Update implementation or minimum version as needed

### Benchmark performance degraded

1. Run benchmarks multiple times to rule out system noise
2. Compare with previous results in `PERFORMANCE_COMPARISON.md`
3. Use profiling tools to identify bottlenecks:
   ```bash
   uv run python -m cProfile -o output.prof benchmark.py
   uv run python -m pstats output.prof
   ```

### Type checking fails

1. Ensure all imports have type stubs
2. Check `pyproject.toml` for pyright configuration
3. Add type ignores only as last resort:
   ```python
   result = some_call()  # type: ignore[some-issue]
   ```

## Pre-commit Hooks

Pre-commit hooks are configured in `prek.toml` and run automatically on commit.

```bash
# Install hooks (included in `just install`)
prek install

# Run manually against all files
uv run prek run --all-files
```

## Getting Help

- **Documentation**: Check `README.md` and other `*.md` files
- **Issues**: Report bugs at the project's issue tracker
- **CI Logs**: Check GitHub Actions for detailed CI output
