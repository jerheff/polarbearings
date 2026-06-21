"""Deterministic, id-keyed data splitting via hashing.

Hashing a stable record id to a uniform in ``(0, 1]`` gives a split assignment that
is **reproducible** across runs and row orderings, **leaks nothing** (the same id
always lands in the same split, even across datasets), and is a plain Polars
expression — so it drops into ``select``, ``with_columns``, ``group_by``, and lazy
pipelines. The same primitive powers the bootstrap weights
(:func:`~polarbearings.bootstrap.bootstrap_weight`).

``seed`` is the **first, required** argument of every helper: it *is* the split's
identity (the "seed-42 split"), and independent splits must use different seeds — a
shared default would silently correlate them (same id + same seed = same uniform),
so there is no default to fall into.

:func:`hash_split` is *consistent*: growing its ``fraction`` only **adds** records to
the split and shrinking only removes the margin, so a holdout stays stable as the
dataset or the fraction changes. :func:`hash_splits` extends that to a multi-way
partition by giving each split its **own seed** and a **residual-conditional
fraction**, so resizing one split leaves the upstream splits' membership unchanged —
unlike a cumulative-threshold scheme, which couples adjacent boundaries and
reshuffles neighbours when any split is resized.

**Stratification.** Because the hash is independent of the labels, a plain
``hash_split`` already samples each class at the same rate *in expectation* (balanced
up to binomial noise). For an *exact* per-class split, rank within the stratum on the
shared uniform and take the bottom fraction by rank::

    u = hash_uniform(1, "id")
    holdout = u.rank("ordinal").over("class") <= (fraction * pl.len()).over("class")

The same ``.over(stratum)`` trick stratifies :func:`hash_fold` (rank within class,
then bucket).
"""

from collections.abc import Sequence

import polars as pl

from polarbearings._common import IntoExpr, col_expr

# Divisor mapping a UInt64 hash into a (0, 1] uniform.
_U64_SCALE = 2.0**64


def hash_uniform(seed: int, id_col: IntoExpr) -> pl.Expr:
    """Deterministic uniform in ``(0, 1]`` from hashing a record id.

    The shared primitive behind :func:`hash_split`, :func:`hash_fold`, and the
    bootstrap weights: ``(hash(id, seed) + 1) / 2**64``. Keyed on the id, so it is
    reproducible across runs and independent of row order.

    Args:
        seed: Hash seed; the split's identity. Different seeds give independent
            uniforms for the same id.
        id_col: Stable per-row identifier (column name or expression) to hash.

    Returns:
        A ``Float64`` Polars expression with values in ``(0, 1]``.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import hash_uniform
        >>> df = pl.DataFrame({"id": [1, 2, 3]})
        >>> df.select(u=hash_uniform(0, "id"))  # doctest: +SKIP
    """
    return (col_expr(id_col).hash(seed=seed).cast(pl.Float64) + 1.0) / _U64_SCALE


def hash_split(seed: int, id_col: IntoExpr, *, fraction: float) -> pl.Expr:
    """Boolean holdout membership, keyed on a stable id.

    ``True`` for the ``~fraction`` of rows whose hashed id falls below the threshold.
    *Consistent*: increasing ``fraction`` keeps every current member and adds more;
    decreasing it only drops the margin — so the split is stable as data or the
    fraction changes. Compose with different ``seed`` values on the residual for a
    multi-way split (or use :func:`hash_splits`).

    Args:
        seed: Hash seed; the split's identity. Use distinct seeds for independent
            splits — sharing a seed correlates them.
        id_col: Stable per-row identifier (column name or expression) to hash.
        fraction: Target share of rows in the holdout, in ``[0, 1]``.

    Returns:
        A ``Boolean`` Polars expression — ``True`` for rows in the holdout.

    Raises:
        ValueError: If ``fraction`` is outside ``[0, 1]``.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import hash_split
        >>> df = pl.DataFrame({"id": range(1000)})
        >>> df.with_columns(holdout=hash_split(1, "id", fraction=0.2))  # doctest: +SKIP
    """
    if not 0.0 <= fraction <= 1.0:
        msg = f"fraction must be in [0, 1], got {fraction!r}."
        raise ValueError(msg)
    return hash_uniform(seed, id_col) < fraction


def hash_fold(seed: int, id_col: IntoExpr, *, k: int) -> pl.Expr:
    """Assign each row a cross-validation fold id in ``0..k-1``, keyed on its id.

    ``floor(hash_uniform(id) * k)`` — a deterministic, reproducible fold membership
    (mirrors ``KFold(shuffle=True)`` semantics; fold *membership*, not index pairs).

    Args:
        seed: Hash seed; the fold assignment's identity.
        id_col: Stable per-row identifier (column name or expression) to hash.
        k: Number of folds (``>= 1``).

    Returns:
        An ``Int64`` Polars expression with values in ``0..k-1``.

    Raises:
        ValueError: If ``k < 1``.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import hash_fold
        >>> df = pl.DataFrame({"id": range(1000)})
        >>> df.with_columns(fold=hash_fold(0, "id", k=5))  # doctest: +SKIP
    """
    if k < 1:
        msg = f"k must be >= 1, got {k!r}."
        raise ValueError(msg)
    fold = (hash_uniform(seed, id_col) * k).floor().cast(pl.Int64)
    return fold.clip(0, k - 1)  # guard the measure-zero u == 1 endpoint


def hash_splits(
    seed: int,
    id_col: IntoExpr,
    splits: Sequence[tuple[str, float]],
    *,
    remainder: str = "rest",
) -> pl.Expr:
    """Partition rows into named splits, keyed on a stable id.

    ``splits`` is an **ordered** sequence of ``(name, fraction-of-whole)`` pairs;
    whatever is left over is labelled ``remainder``. Each split draws from the rows
    not taken by earlier splits, using its **own seed** and a residual-conditional
    threshold (``fraction / (1 - upstream fractions)``). Consequences:

    - Splits are disjoint and cover every row.
    - The proportions hold (in expectation) at the requested ``fraction``-of-whole.
    - Order is **priority**: resizing a split leaves every *upstream* split's
      membership unchanged (only its own and downstream residuals shift). Put the
      long-lived holdout first so it never churns.

    Args:
        seed: Base hash seed; split ``i`` uses ``seed + i`` (so the first split
            matches ``hash_split(seed, id_col, fraction=splits[0][1])``).
        id_col: Stable per-row identifier (column name or expression) to hash.
        splits: Ordered ``(name, fraction)`` pairs; fractions are shares of the whole
            and must be non-negative and sum to ``<= 1``.
        remainder: Label for rows not assigned to any listed split.

    Returns:
        A ``String`` Polars expression of split labels.

    Raises:
        ValueError: If any fraction is negative or the fractions sum to ``> 1``.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import hash_splits
        >>> df = pl.DataFrame({"id": range(1000)})
        >>> df.with_columns(
        ...     split=hash_splits(1, "id", [("test", 0.15), ("val", 0.15)], remainder="train")
        ... )  # doctest: +SKIP
    """
    fractions = [f for _, f in splits]
    if any(f < 0.0 for f in fractions):
        msg = "split fractions must be non-negative."
        raise ValueError(msg)
    total = sum(fractions)
    if total > 1.0 + 1e-9:
        msg = f"split fractions sum to {total!r} > 1."
        raise ValueError(msg)

    label: pl.Expr = pl.lit(remainder, dtype=pl.String)
    assigned: pl.Expr = pl.lit(value=False)
    cumulative = 0.0
    for i, (name, fraction) in enumerate(splits):
        residual = 1.0 - cumulative
        # Conditional share of the residual pool; <= 1 whenever the totals are valid.
        cond = 0.0 if residual <= 0.0 else min(fraction / residual, 1.0)
        in_split = (~assigned) & (hash_uniform(seed + i, id_col) < cond)
        label = pl.when(in_split).then(pl.lit(name, dtype=pl.String)).otherwise(label)
        assigned = assigned | in_split
        cumulative += fraction
    return label
