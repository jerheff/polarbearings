"""Public type aliases used throughout the metric signatures.

Every metric parameter that names a column, a sample weight, a positive-class label,
a grouping, or a threshold is annotated with one of these aliases. They are exported
here so you can reuse them in your own annotations:

```python
from polarbearings.typing import IntoExpr


def my_metric(target: IntoExpr, score: IntoExpr) -> pl.Expr: ...
```

They follow Polars' own ``IntoExpr`` naming convention: anywhere a metric names a
column you may pass either a column name or a Polars expression.
"""

from collections.abc import Sequence
from typing import TypeAlias

import polars as pl

IntoExpr: TypeAlias = str | pl.Expr
"""A column reference: a column name (``str``) or a Polars expression.

Every parameter that names a column accepts this, so a computed column such as
``pl.col("raw").rank()`` can be passed without a prior ``with_columns``.
"""

WeightInput: TypeAlias = IntoExpr | None
"""A sample-weight input: a column reference, or ``None`` for the unweighted case."""

PosLabel: TypeAlias = int | float | str | bool
"""A positive-class label: any scalar comparable to the target column.

For example ``1``, ``100``, ``"cancer"``, or ``True``. Defaults to ``1``.
"""

ByInput: TypeAlias = IntoExpr | Sequence[IntoExpr] | None
"""A ``by=`` grouping argument: one column reference, a sequence of them, or ``None``."""

ThresholdValue: TypeAlias = float | pl.Expr
"""A classification threshold: a fixed ``float`` or an expression.

An expression (e.g. a data-derived quantile evaluated in-engine) lets each group
threshold at its own value.
"""

__all__ = [
    "IntoExpr",
    "WeightInput",
    "PosLabel",
    "ByInput",
    "ThresholdValue",
]
