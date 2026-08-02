---
description: Public type aliases in the polarbearings.typing module (IntoExpr, WeightInput, …) used across the metric signatures and importable for your own annotations.
---
# Type aliases

The metric signatures use a handful of short type aliases for the values they
accept. They live in the public `polarbearings.typing` module, so you can import them
for your own annotations:

```python
from polarbearings.typing import IntoExpr
```

Every `IntoExpr`, `WeightInput`, … in an API signature links back to this page.

These follow Polars' own `IntoExpr` naming convention: anywhere a metric names a
column you may pass either a **column name** (`str`) or a **Polars expression**, so a
computed column such as `pl.col("raw").rank()` works without a prior `with_columns`.

::: polarbearings.typing.IntoExpr
::: polarbearings.typing.WeightInput
::: polarbearings.typing.PosLabel
::: polarbearings.typing.ByInput
::: polarbearings.typing.ThresholdValue
