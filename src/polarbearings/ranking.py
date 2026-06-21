"""Ranking-quality metrics (DCG / NDCG) implemented as Polars expressions.

These operate on **long-format** data: one row per (query, document), with a
``relevance`` (gain) column and a ``score`` column. Evaluate a single ranking by
selecting over the whole frame, or many rankings at once with
``group_by(query).agg(...)`` — each group is one query.

Discounted Cumulative Gain at position ``i`` (0-indexed, ranked by ``score``
descending) is ``gain_i / log_base(i + 2)``; NDCG normalizes DCG by the ideal
ordering (sorting gains by themselves).

**Tie handling.** For *distinct* scores this matches scikit-learn's ``dcg_score`` /
``ndcg_score`` with ``ignore_ties=True``. Under *tied* scores it does not: ties are
broken by physical row order (``sort_by`` is stable on equal keys), not
gain-averaged, so the result is order-dependent and diverges from scikit-learn's
``ignore_ties=True`` (which breaks ties differently). Gain-averaging is not
expressible as a single pure Polars expression. If you need a reproducible result
under tied scores, sort or break ties upstream before evaluating.
"""

import math

import polars as pl

from polarbearings._common import IntoExpr, col_expr, col_name, guarded


def _dcg_expr(gain: pl.Expr, order_by: pl.Expr, k: int | None, log_base: float) -> pl.Expr:
    """DCG aggregation: gains taken in ``order_by``-descending order, log-discounted.

    Works in both whole-frame ``select`` and per-group ``agg`` contexts because
    ``pl.int_range(0, pl.len())`` yields the positional index within the current
    context (the group, under ``agg``).
    """
    pos = pl.int_range(0, pl.len()).cast(pl.Float64)
    discount = math.log(log_base) / (pos + 2.0).log()
    gains_sorted = gain.sort_by(order_by, descending=True)
    if k is not None:
        gains_sorted = gains_sorted.head(k)
        discount = discount.head(k)
    return (gains_sorted * discount).sum()


def _ranking_alias(name: str, relevance: IntoExpr, score: IntoExpr, k: int | None) -> str:
    """Build a ``name_relevance_score[_k<k>]`` alias."""
    alias = f"{name}_{col_name(relevance)}_{col_name(score)}"
    if k is not None:
        alias += f"_k{k}"
    return alias


def dcg_score(
    relevance: IntoExpr, score: IntoExpr, *, k: int | None = None, log_base: float = 2.0
) -> pl.Expr:
    """Compute Discounted Cumulative Gain (DCG) as a Polars expression.

    Documents are ranked by ``score`` (descending); each contributes
    ``relevance / log_base(rank + 2)`` (rank 0-indexed). Mirrors scikit-learn's
    ``dcg_score`` with ``ignore_ties=True`` **for distinct scores**; under tied
    scores ties break by row order (see the module docstring) rather than matching
    scikit-learn.

    Args:
        relevance: Column name or expression with graded relevance / gain values
            (non-negative).
        score: Column name or expression with the predicted scores used to rank
            documents.
        k: Only the top-``k`` ranked documents contribute. ``None`` uses all.
        log_base: Base of the rank discount logarithm. Defaults to 2.

    Returns:
        A Polars expression evaluating to the DCG of the ranking.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import dcg_score
        >>> df = pl.DataFrame({"rel": [3, 2, 3, 0, 1], "score": [3.0, 2.2, 3.5, 0.1, 1.0]})
        >>> df.select(dcg_score("rel", "score")).to_series()[0]  # doctest: +SKIP
        7.14...
    """
    dcg = _dcg_expr(col_expr(relevance).cast(pl.Float64), col_expr(score), k, log_base)
    return guarded(dcg, values=[relevance, score]).alias(_ranking_alias("dcg", relevance, score, k))


def ndcg_score(
    relevance: IntoExpr, score: IntoExpr, *, k: int | None = None, log_base: float = 2.0
) -> pl.Expr:
    """Compute Normalized Discounted Cumulative Gain (NDCG) as a Polars expression.

    NDCG = DCG / IDCG, where IDCG is the DCG of the ideal ranking (gains sorted by
    themselves). Lies in ``[0, 1]`` for non-negative relevance. Mirrors
    scikit-learn's ``ndcg_score`` with ``ignore_ties=True`` **for distinct scores**;
    under tied scores ties break by row order (see the module docstring) rather than
    matching scikit-learn.

    Returns null when the ideal DCG is 0 (every document is irrelevant, so the
    ratio is undefined). Note scikit-learn returns ``0.0`` in that degenerate
    case; this follows the library convention of null for undefined results.

    Args:
        relevance: Column name or expression with graded relevance / gain values
            (non-negative).
        score: Column name or expression with the predicted scores used to rank
            documents.
        k: Only the top-``k`` ranked documents contribute. ``None`` uses all.
        log_base: Base of the rank discount logarithm. Defaults to 2.

    Returns:
        A Polars expression evaluating to the NDCG of the ranking.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import ndcg_score
        >>> df = pl.DataFrame({"rel": [3, 2, 3, 0, 1], "score": [3.0, 2.2, 3.5, 0.1, 1.0]})
        >>> df.select(ndcg_score("rel", "score")).to_series()[0]  # doctest: +SKIP
        0.94...
    """
    rel = col_expr(relevance).cast(pl.Float64)
    dcg = _dcg_expr(rel, col_expr(score), k, log_base)
    idcg = _dcg_expr(rel, rel, k, log_base)
    result = pl.when(idcg == 0).then(None).otherwise(dcg / idcg)
    return guarded(result, values=[relevance, score]).alias(
        _ranking_alias("ndcg", relevance, score, k)
    )
