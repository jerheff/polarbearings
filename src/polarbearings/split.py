"""Deterministic, id-keyed data splitting via hashing.

Hashing a stable record id to a uniform in ``(0, 1]`` gives a split assignment that
is **reproducible** across runs and row orderings, **leaks nothing** (the same id
always lands in the same split, even across datasets), and is a plain Polars
expression — so it drops into ``select``, ``with_columns``, ``group_by``, and lazy
pipelines. The same primitive powers the bootstrap weights
(:func:`~polarbearings.bootstrap.bootstrap_weight`).

**Stable across Polars versions.** The mixing is a fixed `SplitMix64
<https://prng.di.unimi.it/splitmix64.c>`_ finalizer built only from *defined*
wrapping ``UInt64`` arithmetic (add, multiply, XOR, right-shift), **not** Polars'
:meth:`~polars.Expr.hash`, whose docstring reserves the right to change the hash
between releases (*"stability is only guaranteed within a single version"* — and it
did change between 1.24 and 1.42). Because the split key is computed by us, the same
``(seed, id)`` maps to the same uniform on every supported Polars version, so a
holdout pinned today stays pinned after a Polars upgrade.

**Id types.** ``id_col`` must be an **integer** column (any width, signed or unsigned;
values are read as their two's-complement ``UInt64`` bit pattern). For **string /
UUID** ids, materialize a stable ``Int64`` key once and split on that — see
:func:`hash_uniform` for the recipe. Keep the package's sole dependency ``polars`` by
using a fixed hash you control for that key (a :mod:`hashlib` digest, the
``polars-hash`` plugin, or a surrogate key); never Polars' :meth:`~polars.Expr.hash`,
which is the version-unstable function this module exists to avoid.

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
from typing import Final

import polars as pl

from polarbearings._common import IntoExpr, col_expr

# SplitMix64 (Vigna) constants — a fixed, version-independent integer mix. We use
# it instead of ``Expr.hash`` because it is expressible entirely in *defined*
# wrapping-``UInt64`` arithmetic (add/multiply/XOR/right-shift), whose results are
# identical on every Polars version; ``Expr.hash`` explicitly does not promise that.
_GOLDEN_GAMMA: Final = 0x9E3779B97F4A7C15  # odd increment ≈ 2**64 / golden ratio
_MIX_1: Final = 0xBF58476D1CE4E5B9
_MIX_2: Final = 0x94D049BB133111EB
_U64_MASK: Final = (1 << 64) - 1
# Divisor mapping a UInt64 mix into a (0, 1] uniform.
_U64_SCALE: Final = 2.0**64


def _u64(value: int) -> pl.Expr:
    """A ``UInt64`` literal, reduced mod ``2**64`` (so Python ints can't overflow)."""
    return pl.lit(value & _U64_MASK, dtype=pl.UInt64)


def _shr(z: pl.Expr, bits: int) -> pl.Expr:
    """Logical right shift of a ``UInt64`` expression (``z >> bits``).

    Polars has no expression shift operator on the 1.0.0 floor, but an unsigned
    right shift by ``bits`` is exactly floor-division by ``2**bits``. The cast pins
    the result back to ``UInt64`` so no version can promote it to a wider dtype.
    """
    return (z // _u64(1 << bits)).cast(pl.UInt64)


def _splitmix64(state: pl.Expr) -> pl.Expr:
    """One SplitMix64 step: advance the state by ``GOLDEN_GAMMA``, then finalize.

    Every operation wraps at 64 bits and is dtype-pinned to ``UInt64``, so the mixed
    integer is bit-identical across Polars versions. The leading increment means a
    zero ``state`` (e.g. ``id == 0`` at ``seed == 0``) does not map to zero.
    """
    z = (state + _u64(_GOLDEN_GAMMA)).cast(pl.UInt64)
    z = ((z ^ _shr(z, 30)).cast(pl.UInt64) * _u64(_MIX_1)).cast(pl.UInt64)
    z = ((z ^ _shr(z, 27)).cast(pl.UInt64) * _u64(_MIX_2)).cast(pl.UInt64)
    return (z ^ _shr(z, 31)).cast(pl.UInt64)


def hash_uniform(seed: int, id_col: IntoExpr) -> pl.Expr:
    """Deterministic uniform in ``(0, 1]`` from mixing a record id.

    The shared primitive behind :func:`hash_split`, :func:`hash_fold`, and the
    bootstrap weights: ``(splitmix64(id, seed) + 1) / 2**64``. Keyed on the id, so it
    is reproducible across runs, independent of row order, and — unlike Polars'
    :meth:`~polars.Expr.hash` — **stable across Polars versions** (see the module
    docstring). The id's bits are offset by ``seed`` before the mix, so distinct seeds
    give independent uniforms.

    ``id_col`` must be an **integer** column (any width, signed or unsigned; must fit
    in ``Int64``). For **string / UUID** ids, materialize a stable ``Int64`` key once
    and split on that column. Use a *fixed* hash for the key so it is reproducible;
    do **not** use Polars' :meth:`~polars.Expr.hash`, the version-unstable function
    this module exists to avoid. For example, with the ``polars-hash`` plugin::

        keyed = df.with_columns(id_key=pl.col("uuid").nchash.wyhash().reinterpret(signed=True))
        keyed.with_columns(holdout=hash_split(1, "id_key", fraction=0.2))

    or, dependency-free, a one-time :mod:`hashlib` pass materialized to a column::

        import hashlib

        def _key(s: str) -> int:
            digest = hashlib.blake2b(s.encode(), digest_size=8).digest()
            return int.from_bytes(digest, "little") - 2**63  # center into Int64

        keyed = df.with_columns(id_key=pl.col("uuid").map_elements(_key, pl.Int64))

    Args:
        seed: Hash seed; the split's identity. Different seeds give independent
            uniforms for the same id.
        id_col: Stable per-row **integer** identifier (column name or expression) to
            hash. Any integer width, signed or unsigned; must fit in ``Int64``.

    Returns:
        A ``Float64`` Polars expression with values in ``(0, 1]``.

    Examples:
        >>> import polars as pl
        >>> from polarbearings import hash_uniform
        >>> df = pl.DataFrame({"id": [1, 2, 3]})
        >>> df.select(u=hash_uniform(0, "id"))  # doctest: +SKIP
    """
    # Widen to Int64 then reinterpret the bits as UInt64: works for every integer
    # width and maps negative ids by two's complement (a plain cast would reject them).
    key = col_expr(id_col).cast(pl.Int64).reinterpret(signed=False)
    state = (key + _u64(seed * _GOLDEN_GAMMA)).cast(pl.UInt64)
    return (_splitmix64(state).cast(pl.Float64) + 1.0) / _U64_SCALE


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
        id_col: Stable per-row integer identifier (column name or expression) to hash
            (see :func:`hash_uniform` for string-id handling).
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
        id_col: Stable per-row integer identifier (column name or expression) to hash
            (see :func:`hash_uniform` for string-id handling).
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
