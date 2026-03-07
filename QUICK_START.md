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
- Polars >= 1.0.0

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

### Classification Metrics

```python
from polarbear import precision, recall, f1_score, specificity, matthews_corrcoef

df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.4, 0.6, 0.9]})

result = df.select(
    precision("label", "prob"),
    recall("label", "prob"),
    f1_score("label", "prob"),
    specificity("label", "prob"),
    matthews_corrcoef("label", "prob"),
)
```

### Regression Metrics

```python
from polarbear import mae, mse, rmse, r2_score, mape

df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "pred": [1.1, 2.2, 2.8]})

result = df.select(
    mae("y", "pred"),
    mse("y", "pred"),
    rmse("y", "pred"),
    r2_score("y", "pred"),
    mape("y", "pred"),
)
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
uv sync

# Run tests
uv run pytest tests/ -v
```

### Quick Commands (with just)

Install [just](https://github.com/casey/just): `brew install just`

```bash
just                    # List all commands
just test               # Run tests
just test-fast          # Quick test run
just test-compat        # Test min, mid & latest Polars
just quality            # Lint + type check
just ci                 # Full CI checks
just pre-commit         # Quick pre-commit check
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
│   ├── __init__.py              # Public API exports
│   ├── roc_auc.py               # ROC AUC metric
│   ├── average_precision.py     # Average precision metric
│   ├── log_loss.py              # Log loss metric
│   ├── brier_score.py           # Brier score metric
│   ├── classification.py        # Threshold-based classification metrics
│   └── regression.py            # Regression metrics
├── tests/
│   ├── test_aoc.py              # ROC AUC tests
│   ├── test_additional_metrics.py   # Log loss & Brier tests
│   ├── test_average_precision.py    # Average precision tests
│   ├── test_classification.py       # Classification metric tests
│   ├── test_regression.py           # Regression metric tests
│   ├── test_new_classification.py   # MCC, kappa, specificity, fbeta tests
│   ├── test_new_regression.py       # R², MAPE tests
│   ├── test_weights.py              # Weighted metric tests
│   ├── test_edge_cases.py           # Edge case tests
│   └── test_degenerate_inputs.py    # Degenerate input tests
├── benchmarks/                  # Performance benchmarks
├── justfile                     # Task runner commands
└── docs/
    ├── guides/
    │   └── TESTING.md           # Testing guide
    └── technical/
        ├── POLARS_COMPATIBILITY.md  # Version compatibility info
        └── PERFORMANCE.md      # Performance analysis & benchmarks
```

## Key Files

- **`justfile`**: Development commands (run `just --list`)
- **`docs/guides/TESTING.md`**: Testing documentation
- **`docs/technical/PERFORMANCE.md`**: Performance analysis
- **`docs/technical/POLARS_COMPATIBILITY.md`**: Version compatibility

## Common Tasks

### Run Tests
```bash
just test          # or: uv run pytest tests/ -v
```

### Check Code Quality
```bash
just quality       # or: uv run ruff check src/ && uv run pyright src/polarbear
```

### Run Benchmarks
```bash
just bench
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

## Polars Version Support

- **Minimum**: 1.0.0 (July 2024)
- **CI tested**: 1.0.0, 1.24.0, 1.38.1

## Getting Help

- **Documentation**: See `docs/` directory for guides and technical docs
- **Testing guide**: `docs/guides/TESTING.md`
- **Performance**: `docs/technical/PERFORMANCE.md`
- **Issues**: Report on GitHub

## Links

- **Just**: [Installation](https://github.com/casey/just#installation)
- **Polars**: [Documentation](https://docs.pola.rs/)
- **uv**: [Documentation](https://docs.astral.sh/uv/)
