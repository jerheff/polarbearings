# Polarbear =;D

High-performance machine learning metrics implemented as native Polars expressions.

## Features

- **Fast**: Metrics implemented as Polars expressions for maximum performance
- **Correct**: Thoroughly tested against scikit-learn with property-based testing
- **Simple**: Clean, intuitive API that follows Polars conventions
- **Type-safe**: Full type hints with pyright strict mode

## Installation

```bash
pip install polarbear
```

Or with uv:

```bash
uv add polarbear
```

## Quick Start

```python
import polars as pl
from polarbear import roc_auc

# Create your data
df = pl.DataFrame({
    "actual": [0, 0, 1, 1, 1],
    "predicted_score": [0.1, 0.4, 0.35, 0.8, 0.9]
})

# Calculate ROC AUC as a Polars expression
result = df.select(roc_auc("actual", "predicted_score"))
print(result)
# shape: (1, 1)
# ┌──────────────────────────────────┐
# │ roc_auc_actual_predicted_score   │
# │ ---                              │
# │ f64                              │
# ╞══════════════════════════════════╡
# │ 0.833333                         │
# └──────────────────────────────────┘
```

## Available Metrics

### Ranking Metrics

#### ROC AUC

Receiver Operating Characteristic Area Under the Curve for binary classification.

```python
from polarbear import roc_auc

df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9]})
df.select(roc_auc("label", "score"))  # Returns: 1.0
```

- Uses the Mann-Whitney U statistic for correct tie handling
- Matches scikit-learn's `roc_auc_score` exactly

#### Average Precision

Non-interpolated average precision score for binary classification.

```python
from polarbear import average_precision

df = pl.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.4, 0.35, 0.8]})
df.select(average_precision("label", "score"))
```

- Matches scikit-learn's `average_precision_score`
- Handles tied scores correctly

### Probabilistic Metrics

#### Log Loss (Binary Cross-Entropy)

```python
from polarbear import log_loss

df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
df.select(log_loss("label", "prob"))
```

#### Brier Score

```python
from polarbear import brier_score

df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]})
df.select(brier_score("label", "prob"))
```

### Classification Metrics (Threshold-Based)

All classification metrics accept an optional `threshold` parameter (default 0.5).

```python
from polarbear import precision, recall, f1_score, fbeta_score, specificity
from polarbear import accuracy, balanced_accuracy, matthews_corrcoef, cohens_kappa

df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.4, 0.6, 0.9]})

df.select(
    precision("label", "prob"),
    recall("label", "prob"),
    f1_score("label", "prob"),
    specificity("label", "prob"),
    accuracy("label", "prob"),
    balanced_accuracy("label", "prob"),
    matthews_corrcoef("label", "prob"),
    cohens_kappa("label", "prob"),
)

# F-beta with custom beta (0.5 weights precision higher, 2.0 weights recall higher)
df.select(fbeta_score("label", "prob", beta=2.0))

# Custom threshold
df.select(precision("label", "prob", threshold=0.7))
```

#### Threshold Sweep

Compute any classification metric across multiple thresholds in a single pass:

```python
from polarbear import f1_score, threshold_sweep

df = pl.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.4, 0.6, 0.9]})
df.select(*threshold_sweep(f1_score, "label", "prob", [0.3, 0.5, 0.7]))
```

### Regression Metrics

```python
from polarbear import mae, mse, rmse, r2_score, mape

df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "pred": [1.1, 2.2, 2.8, 4.5]})

df.select(
    mae("y", "pred"),
    mse("y", "pred"),
    rmse("y", "pred"),
    r2_score("y", "pred"),
    mape("y", "pred"),
)
```

### Sample Weights

All metrics support optional sample weights:

```python
df.select(roc_auc("label", "score", weight="sample_weight"))
df.select(mae("y", "pred", weight="w"))
```

## Use Cases

Polarbear is perfect for:

1. **Large-scale model evaluation**: Evaluate millions of predictions efficiently
2. **Real-time metrics**: Calculate metrics in streaming pipelines
3. **Group-wise metrics**: Leverage Polars' `group_by` for segment analysis
4. **Memory efficiency**: Process data that doesn't fit in memory with Polars LazyFrames

### Example: Group-wise ROC AUC

```python
import polars as pl
from polarbear import roc_auc

# Calculate ROC AUC per customer segment
df = pl.DataFrame({
    "segment": ["A", "A", "A", "B", "B", "B"],
    "label": [0, 1, 1, 0, 0, 1],
    "score": [0.2, 0.8, 0.9, 0.1, 0.3, 0.85]
})

result = df.group_by("segment").agg(
    roc_auc("label", "score")
)
print(result)
# shape: (2, 2)
# ┌─────────┬─────────────────────┐
# │ segment │ roc_auc_label_score │
# │ ---     │ ---                 │
# │ str     │ f64                 │
# ╞═════════╪═════════════════════╡
# │ A       │ 1.0                 │
# │ B       │ 0.5                 │
# └─────────┴─────────────────────┘
```

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and [just](https://github.com/casey/just) for task running.

### Quick Commands (with just)

```bash
just              # List all commands
just test         # Run tests
just test-fast    # Quick test run
just quality      # Run linting and type checking
just ci           # Run all CI checks locally
just pre-commit   # Quick pre-commit check
```

### Manual Commands

```bash
# Install dependencies
uv sync --all-groups

# Run tests
uv run pytest

# Run type checking
uv run pyright src/polarbear

# Run linting
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/
```

## Testing

Polarbear uses a comprehensive testing strategy:

- **Unit tests**: Basic functionality and edge cases
- **Property-based tests**: Random data generation with Hypothesis
- **Compatibility tests**: Verified against scikit-learn on multiple Polars versions

```bash
# Run all tests
just test  # or: uv run pytest

# Test multiple Polars versions
just test-compat

# Run benchmarks
just bench
```

## Performance

Polarbear is **2-4x faster than sklearn** on large datasets:

| Metric | 100k samples | Speedup vs sklearn |
|--------|--------------|-------------------|
| ROC AUC | 3.2ms | **3.99x faster** |
| Log Loss | 1.8ms | **3.01x faster** |
| Brier Score | 0.16ms | **2.91x faster** |

## Requirements

- **Python**: 3.11+
- **Polars**: 1.0.0+

## Roadmap

- [x] ROC AUC for binary classification
- [x] Log Loss / Binary Cross-Entropy
- [x] Brier Score
- [x] Average Precision Score
- [x] Precision, Recall, F1 Score, Accuracy, Balanced Accuracy
- [x] Specificity, F-beta Score, Matthews Correlation Coefficient, Cohen's Kappa
- [x] R-squared, MAPE
- [x] Weighted variants for all metrics
- [x] Multi-version Polars support (1.0.0+)
- [ ] Multi-class ROC AUC (one-vs-rest, one-vs-one)
- [ ] Precision-Recall AUC
- [ ] Calibration metrics (ECE, MCE)

## Contributing

Contributions are welcome! Please:

1. Run `just ci` to verify your changes locally
2. Submit a Pull Request with a clear description

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with [Polars](https://www.pola.rs/)
- Tested against [scikit-learn](https://scikit-learn.org/)
- Property-based testing with [Hypothesis](https://hypothesis.readthedocs.io/)
