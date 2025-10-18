# Quick Start Guide

## For Users

### Installation

```bash
# With pip
pip install polarbear

# With uv
uv add polarbear
```

### Requirements

- Python >= 3.11
- Polars >= 1.0.0 (tested up to 1.34.0)

### Basic Usage

```python
import polars as pl
from polarbear import roc_auc, log_loss, brier_score

# Your data
df = pl.DataFrame({
    "actual": [0, 0, 1, 1, 1],
    "predicted": [0.1, 0.4, 0.35, 0.8, 0.9]
})

# Calculate metrics
result = df.select(
    roc_auc("actual", "predicted"),
    log_loss("actual", "predicted"),
    brier_score("actual", "predicted"),
)

print(result)
```

### With Group By

```python
# Calculate metrics per group
result = df.group_by("user_id").agg(
    roc_auc("label", "score"),
    log_loss("label", "prob"),
)
```

## For Contributors

### Setup Development Environment

```bash
# Clone repo
git clone <repo-url>
cd polarbear

# Install dependencies
uv sync --all-groups

# Run tests
uv run pytest tests/ -v
```

### Quick Commands (with just)

Install [just](https://github.com/casey/just): `brew install just`

```bash
just                    # List all commands
just test               # Run tests
just test-fast          # Quick test run
just test-versions      # Test min & latest Polars
just quality            # Lint + type check
just ci                 # Full CI checks
just pre-commit         # Quick pre-commit check
```

### Testing Polars Compatibility

```bash
# Quick test (min 1.0.0 + latest 1.34.0)
just test-versions

# Or manually
uv run python test_versions.py --min-max

# Test all versions
uv run python test_versions.py

# Test specific versions
uv run python test_versions.py --versions 1.0.0 1.20.0
```

### Before Committing

```bash
# Quick check
just pre-commit

# Or full CI check
just ci
```

## Project Structure

```
polarbear/
├── src/polarbear/
│   ├── __init__.py           # Public API
│   └── metrics.py            # Metric implementations
├── tests/
│   ├── test_aoc.py          # ROC AUC tests
│   ├── test_additional_metrics.py  # Log loss & Brier tests
│   └── test_edge_cases.py   # Edge case tests
├── benchmark.py              # Performance benchmarks
├── test_versions.py          # Version compatibility testing
├── justfile                  # Task runner commands
└── docs/
    ├── guides/
    │   └── TESTING.md       # Comprehensive testing guide
    └── technical/
        ├── POLARS_COMPATIBILITY.md  # Version compatibility info
        └── PERFORMANCE.md   # Performance analysis & benchmarks
```

## Key Files

- **`justfile`**: Development commands (run `just --list`)
- **`docs/guides/TESTING.md`**: Full testing documentation
- **`docs/technical/PERFORMANCE.md`**: Performance analysis
- **`docs/technical/POLARS_COMPATIBILITY.md`**: Version compatibility
- **`test_versions.py`**: Test multiple Polars versions locally
- **`benchmark.py`**: Performance benchmarking

## Common Tasks

### Run Tests
```bash
just test          # or: uv run pytest tests/ -v
```

### Check Code Quality
```bash
just quality       # or: uv run ruff check src/ && uv run mypy src/
```

### Run Benchmarks
```bash
just bench         # or: uv run python benchmark.py
```

### Test Polars Versions
```bash
just test-versions  # or: uv run python test_versions.py --min-max
```

### Format Code
```bash
just format        # or: uv run ruff format src/ tests/
```

## Performance

Polarbear is **2-4x faster than sklearn** on large datasets:

| Metric | 100k samples | Speedup |
|--------|--------------|---------|
| ROC AUC | 3.2ms | 3.99x |
| Log Loss | 1.8ms | 3.01x |
| Brier Score | 0.16ms | 2.91x |

See `benchmark.py` for detailed benchmarks.

## Polars Version Support

- **Minimum**: 1.0.0 (July 2024)
- **Latest tested**: 1.34.0 (October 2025)
- **Tested versions**: 1.0.0, 1.10.0, 1.20.0, 1.34.0

All tests pass on all tested versions. See `docs/technical/POLARS_COMPATIBILITY.md` for details.

## Getting Help

- **Documentation**: See `docs/` directory for guides and technical docs
- **Testing guide**: `docs/guides/TESTING.md`
- **Performance**: `docs/technical/PERFORMANCE.md`
- **Issues**: Report on GitHub
- **Questions**: Open a discussion

## Links

- **Repository**: [GitHub](https://github.com/your-username/polarbear)
- **Just**: [Installation](https://github.com/casey/just#installation)
- **Polars**: [Documentation](https://docs.pola.rs/)
- **uv**: [Documentation](https://docs.astral.sh/uv/)
