"""The ``weight`` parameter accepts a Polars expression, not only a column name.

This is the foundation for composed weights (e.g. base weight x bootstrap
weight). Passing ``pl.col("w")`` must match ``weight="w"``, and passing a
composed expression must match the same expression pre-materialized as a column.
"""

import numpy as np
import polars as pl
import pytest

from polarbearings import f1_score, mae, roc_auc


def _val(df: pl.DataFrame, expr: pl.Expr) -> float:
    return df.select(expr.alias("m")).to_series()[0]


def _frame() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    n = 500
    return pl.DataFrame(
        {
            "label": rng.integers(0, 2, n),
            "score": rng.random(n),
            "y": rng.normal(size=n),
            "pred": rng.normal(size=n),
            "w": rng.uniform(0.1, 3.0, n),
        }
    )


@pytest.mark.parametrize(
    ("metric", "cols"),
    [
        (mae, ("y", "pred")),
        (roc_auc, ("label", "score")),
        (f1_score, ("label", "score")),
    ],
    ids=["mae", "roc_auc", "f1_score"],
)
def test_col_expr_matches_column_name(metric, cols):
    df = _frame()
    by_name = _val(df, metric(*cols, weight="w"))
    by_expr = _val(df, metric(*cols, weight=pl.col("w")))
    assert by_expr == pytest.approx(by_name)


@pytest.mark.parametrize(
    ("metric", "cols"),
    [
        (mae, ("y", "pred")),
        (roc_auc, ("label", "score")),
        (f1_score, ("label", "score")),
    ],
    ids=["mae", "roc_auc", "f1_score"],
)
def test_composed_expr_matches_materialized_column(metric, cols):
    df = _frame()
    # weight = w * 2 as an inline expression ...
    by_expr = _val(df, metric(*cols, weight=pl.col("w") * 2.0))
    # ... must equal pre-materializing that doubled column and passing its name.
    df_mat = df.with_columns(w2=pl.col("w") * 2.0)
    by_col = _val(df_mat, metric(*cols, weight="w2"))
    assert by_expr == pytest.approx(by_col)


def test_expression_weight_alias_suffix():
    # A named-column weight keeps its name in the alias; an expression gets "_w".
    df = _frame()
    assert df.select(mae("y", "pred", weight="w")).columns == ["mae_y_pred_w"]
    assert df.select(mae("y", "pred", weight=pl.col("w") * 2.0)).columns == ["mae_y_pred_w"]
    assert df.select(mae("y", "pred")).columns == ["mae_y_pred"]
