"""Tests for the deterministic, id-keyed data-splitting helpers."""

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polarbearings import hash_fold, hash_split, hash_splits, hash_uniform


def _ids(n=20000):
    return pl.DataFrame({"id": np.arange(n)})


class TestHashUniform:
    def test_range_0_1(self):
        u = _ids().select(hash_uniform(0, "id").alias("u"))["u"].to_numpy()
        assert u.min() > 0.0
        assert u.max() <= 1.0
        assert u.mean() == pytest.approx(0.5, abs=0.01)

    def test_deterministic_and_seed_independent(self):
        df = _ids(2000)
        a = df.select(hash_uniform(1, "id").alias("u"))["u"]
        b = df.select(hash_uniform(1, "id").alias("u"))["u"]
        c = df.select(hash_uniform(2, "id").alias("u"))["u"]
        assert (a == b).all()
        assert not (a == c).all()

    def test_reproducible_across_row_order(self):
        df = _ids(3000)
        base = df.select("id", hash_uniform(4, "id").alias("u"))
        shuffled = (
            df.sample(fraction=1.0, shuffle=True, seed=9)
            .select("id", hash_uniform(4, "id").alias("u"))
            .sort("id")
        )
        assert (base.sort("id")["u"] == shuffled["u"]).all()

    def test_canary_pinned_values_are_stable_across_polars_versions(self):
        # The whole point of the SplitMix64 mix (vs Polars' Expr.hash) is that a
        # holdout pinned today survives a Polars upgrade. These values are the exact
        # `(id, seed) -> uniform` outputs; they are byte-identical across every Polars
        # in the compat matrix (floor through latest). If this fails on a *new* Polars
        # leg, the mix drifted — investigate before shipping: every pinned split
        # silently reshuffled.
        df = pl.DataFrame({"id": [-1, 0, 1, 2, 3]}, schema={"id": pl.Int64})
        got = df.select(u0=hash_uniform(0, "id"), u1=hash_uniform(1, "id"))
        expected_seed0 = [
            0.8939429202831846,
            0.8833108082136427,
            0.566561575172281,
            0.5911897341980795,
            0.11345034205715465,
        ]
        expected_seed1 = [
            0.9125972035944532,
            0.43152799704851,
            0.7457817572627012,
            0.7491496838738247,
            0.7002935135929024,
        ]
        assert got["u0"].to_list() == expected_seed0
        assert got["u1"].to_list() == expected_seed1

    def test_canary_pinned_public_split_assignments(self):
        # Pin the exact output of every user-facing split fn (not just the underlying
        # uniform), so a reshuffle is caught at each entry point a user actually calls.
        # Same 5-record df; values are stable across every Polars in the compat matrix.
        df = pl.DataFrame({"id": [-1, 0, 1, 2, 3]}, schema={"id": pl.Int64})
        assert df.select(hash_split(0, "id", fraction=0.6))[:, 0].to_list() == [
            False,
            False,
            True,
            True,
            True,
        ]
        assert df.select(hash_fold(0, "id", k=5))[:, 0].to_list() == [4, 4, 2, 2, 0]
        splits = hash_splits(0, "id", [("test", 0.3), ("val", 0.3)], remainder="train")
        assert df.select(splits)[:, 0].to_list() == ["train", "train", "train", "train", "test"]

    def test_integer_width_and_sign_invariant(self):
        # The id is read as its two's-complement UInt64 bit pattern, so the same
        # integer value hashes identically regardless of the column's integer width.
        vals = [0, 1, 2, 3, 4]
        ref = pl.DataFrame({"id": vals}, schema={"id": pl.Int64}).select(
            hash_uniform(0, "id").alias("u")
        )["u"]
        for dt in (pl.Int8, pl.Int32, pl.UInt16, pl.UInt64):
            u = pl.DataFrame({"id": vals}, schema={"id": dt}).select(
                hash_uniform(0, "id").alias("u")
            )["u"]
            assert (u == ref).all()

    def test_non_numeric_string_id_is_rejected(self):
        # id_col must be an integer key: a non-numeric string id fails loudly at
        # collect rather than silently falling back to a version-unstable hash. String
        # ids are handled by materializing a stable integer key first (see the
        # hash_uniform docstring), not by this module.
        df = pl.DataFrame({"id": ["a", "b", "c", "d", "e"]})
        with pytest.raises(pl.exceptions.InvalidOperationError):
            df.select(hash_uniform(0, "id"))

    def test_string_id_via_materialized_integer_key(self):
        # The documented string-id path: hash the string to a stable Int64 key column
        # yourself (here, a fixed hashlib digest), then split on that integer column.
        import hashlib

        def _key(s: str) -> int:
            digest = hashlib.blake2b(s.encode(), digest_size=8).digest()
            return int.from_bytes(digest, "little") - 2**63  # center into Int64

        df = pl.DataFrame({"id": ["alice", "bob", "carol", "dave", "eve"]}).with_columns(
            id_key=pl.col("id").map_elements(_key, return_dtype=pl.Int64)
        )
        u = df.select(hash_uniform(0, "id_key").alias("u"))["u"].to_numpy()
        assert u.min() > 0.0
        assert u.max() <= 1.0
        # Deterministic and usable by the public helpers.
        h = df.select(hash_split(0, "id_key", fraction=0.5).alias("h"))["h"]
        assert h.to_list() == df.select(hash_split(0, "id_key", fraction=0.5))[:, 0].to_list()


class TestSeedMixing:
    """The seed enters as ``splitmix64(id_bits + seed*GAMMA)`` — literally the
    SplitMix64 generator — so distinct seeds must behave like independent draws for
    every id. These checks need statistical sample sizes (unlike the exact 5-row
    canaries above); the inputs are fixed ids and fixed seeds, so the numbers are
    deterministic and the margins are wide enough not to be flaky.
    """

    @staticmethod
    def _u(ids, seed):
        return ids.select(hash_uniform(seed, "id").alias("u"))["u"].to_numpy()

    def test_cross_seed_pearson_near_zero(self):
        # No linear correlation between the uniforms of two different seeds.
        ids = _ids()
        for a, b in [(0, 1), (1, 2), (7, 8), (0, 100), (0, 999)]:
            r = np.corrcoef(self._u(ids, a), self._u(ids, b))[0, 1]
            assert abs(r) < 0.05, f"seeds {a},{b} correlated: r={r:.4f}"

    def test_cross_seed_joint_independence(self):
        # Stronger than Pearson: bin (u_seed0, u_seed1) on a 5x5 grid. Independence
        # spreads the points evenly, so every cell holds ~n/25 — this catches
        # nonlinear structure that a correlation near 0 would miss.
        ids = _ids()
        counts, _, _ = np.histogram2d(
            self._u(ids, 0), self._u(ids, 1), bins=5, range=[[0, 1], [0, 1]]
        )
        expected = ids.height / 25
        assert counts.min() > 0.8 * expected
        assert counts.max() < 1.2 * expected

    def test_one_seed_step_rerandomizes_the_split(self):
        # A +1 seed change makes a 50% holdout agree with the previous one only ~50%
        # of the time — it re-randomizes membership rather than merely nudging it.
        ids = _ids()
        for s in (0, 1, 50):
            a = ids.select(hash_split(s, "id", fraction=0.5).alias("h"))["h"].to_numpy()
            b = ids.select(hash_split(s + 1, "id", fraction=0.5).alias("h"))["h"].to_numpy()
            assert abs((a == b).mean() - 0.5) < 0.03

    def test_folds_reassign_independently_across_seeds(self):
        # k-fold membership under adjacent seeds coincides on only ~1/k of rows.
        ids = _ids()
        k = 10
        a = ids.select(hash_fold(0, "id", k=k).alias("f"))["f"].to_numpy()
        b = ids.select(hash_fold(1, "id", k=k).alias("f"))["f"].to_numpy()
        assert abs((a == b).mean() - 1 / k) < 0.02

    def test_split_rate_holds_for_every_seed(self):
        # Each seed independently hits the requested fraction (no degenerate seed).
        ids = _ids()
        for s in range(25):
            rate = ids.select(hash_split(s, "id", fraction=0.2).alias("h"))["h"].mean()
            assert abs(rate - 0.2) < 0.02, f"seed {s} rate {rate:.4f}"

    def test_each_seed_is_uniform(self):
        # Every seed's uniforms cover (0, 1] with the U(0,1) mean and spread.
        ids = _ids()
        for s in (0, 1, 42, 999):
            u = self._u(ids, s)
            assert abs(u.mean() - 0.5) < 0.01
            assert abs(u.std() - (1 / 12) ** 0.5) < 0.01  # U(0,1) std = 0.2887

    def test_seed_axis_stream_is_uniform_and_serially_uncorrelated(self):
        # Dual of the checks above: fix an id and sweep the SEED. That per-id sequence
        # is the SplitMix64 stream for the id, so it must be ~U(0,1) with ~zero lag-1
        # autocorrelation — consecutive seeds are not linked. Deterministic (fixed id
        # + fixed seed range), so the bounds only need to clear the observed values.
        for the_id in (1, 12345):
            df = pl.DataFrame({"id": [the_id]}, schema={"id": pl.Int64})
            u = np.array([df.select(hash_uniform(s, "id"))[0, 0] for s in range(1200)])
            assert abs(u.mean() - 0.5) < 0.04
            assert abs(u.std() - (1 / 12) ** 0.5) < 0.025
            assert abs(np.corrcoef(u[:-1], u[1:])[0, 1]) < 0.13


class TestHashSplit:
    def test_fraction_is_approximately_right(self):
        for f in (0.1, 0.33, 0.8):
            rate = _ids().select(hash_split(1, "id", fraction=f).alias("h"))["h"].mean()
            assert rate == pytest.approx(f, abs=0.02)

    def test_consistent_growing_is_superset(self):
        # The key property: enlarging the fraction keeps every current member.
        df = _ids()
        small = df.select(hash_split(3, "id", fraction=0.2).alias("h"))["h"].to_numpy()
        big = df.select(hash_split(3, "id", fraction=0.35).alias("h"))["h"].to_numpy()
        assert (small <= big).all()  # small holdout is a subset of the bigger one

    def test_reproducible_across_row_order(self):
        df = _ids(3000)
        base = df.select("id", hash_split(2, "id", fraction=0.25).alias("h"))
        shuffled = (
            df.sample(fraction=1.0, shuffle=True, seed=7)
            .select("id", hash_split(2, "id", fraction=0.25).alias("h"))
            .sort("id")
        )
        assert (base.sort("id")["h"] == shuffled["h"]).all()

    def test_approx_stratified_for_free(self):
        # Hash is label-independent, so each class is sampled at ~fraction.
        df = _ids().with_columns(
            cls=pl.when(pl.col("id") % 4 == 0).then(pl.lit("a")).otherwise(pl.lit("b"))
        )
        rates = (
            df.with_columns(h=hash_split(1, "id", fraction=0.3))
            .group_by("cls")
            .agg(pl.col("h").mean())
        )
        for r in rates["h"]:
            assert r == pytest.approx(0.3, abs=0.02)

    def test_rejects_bad_fraction(self):
        with pytest.raises(ValueError, match="fraction must be in"):
            _ids(10).select(hash_split(0, "id", fraction=1.5))


class TestHashFold:
    def test_range_and_balance(self):
        f = _ids().select(hash_fold(0, "id", k=5).alias("f"))["f"]
        assert f.min() == 0
        assert f.max() == 4
        counts = f.value_counts()["count"].to_numpy()
        assert counts.min() > 0.18 * 20000  # each fold within ~10% of n/k

    def test_deterministic(self):
        df = _ids(2000)
        a = df.select(hash_fold(1, "id", k=7).alias("f"))["f"]
        b = df.select(hash_fold(1, "id", k=7).alias("f"))["f"]
        assert (a == b).all()

    def test_rejects_bad_k(self):
        with pytest.raises(ValueError, match="k must be >= 1"):
            _ids(10).select(hash_fold(0, "id", k=0))


class TestHashSplits:
    def test_disjoint_and_covers(self):
        df = _ids()
        out = df.select(
            hash_splits(0, "id", [("test", 0.15), ("val", 0.15)], remainder="train").alias("s")
        )
        assert set(out["s"].unique()) == {"test", "val", "train"}
        assert out.height == df.height  # every row labelled exactly once

    def test_proportions(self):
        out = _ids().select(
            hash_splits(0, "id", [("test", 0.2), ("val", 0.1)], remainder="train").alias("s")
        )["s"]
        vc = out.value_counts()
        counts = dict(zip(vc["s"], vc["count"], strict=True))
        assert counts["test"] / 20000 == pytest.approx(0.2, abs=0.02)
        assert counts["val"] / 20000 == pytest.approx(0.1, abs=0.02)
        assert counts["train"] / 20000 == pytest.approx(0.7, abs=0.02)

    def test_first_split_matches_hash_split(self):
        df = _ids()
        first = df.select(
            (hash_splits(0, "id", [("test", 0.15), ("val", 0.15)]) == "test").alias("a")
        )["a"]
        plain = df.select(hash_split(0, "id", fraction=0.15).alias("a"))["a"]
        assert (first == plain).all()

    def test_resizing_a_split_leaves_upstream_unchanged(self):
        # Upstream-stability: resize val, and test (declared before it) is identical.
        df = _ids()
        a = df.select(
            hash_splits(0, "id", [("test", 0.15), ("val", 0.15)], remainder="train").alias("s")
        )["s"]
        b = df.select(
            hash_splits(0, "id", [("test", 0.15), ("val", 0.30)], remainder="train").alias("s")
        )["s"]
        assert ((a == "test").to_numpy() == (b == "test").to_numpy()).all()

    def test_reproducible_across_row_order(self):
        df = _ids(3000)
        spec = [("test", 0.2), ("val", 0.2)]
        base = df.select("id", hash_splits(1, "id", spec).alias("s"))
        shuffled = (
            df.sample(fraction=1.0, shuffle=True, seed=5)
            .select("id", hash_splits(1, "id", spec).alias("s"))
            .sort("id")
        )
        assert (base.sort("id")["s"] == shuffled["s"]).all()

    def test_rejects_oversized_and_negative(self):
        with pytest.raises(ValueError, match="sum to"):
            _ids(10).select(hash_splits(0, "id", [("a", 0.7), ("b", 0.5)]))
        with pytest.raises(ValueError, match="non-negative"):
            _ids(10).select(hash_splits(0, "id", [("a", -0.1)]))


_PROP = settings(
    deadline=None, database=None, suppress_health_check=[HealthCheck.differing_executors]
)


@given(
    seed=st.integers(min_value=0, max_value=10000),
    offset=st.integers(min_value=1, max_value=10000),
)
@_PROP
def test_cross_seed_uncorrelated_property(seed, offset):
    # For ARBITRARY distinct seed pairs (not just hand-picked ones), the per-id
    # uniforms stay ~uncorrelated. At n=4000 the sampling noise on r is ~0.016, so the
    # 0.09 bound is a comfortable ~5.7 sigma — sensitive to real correlation, not flaky.
    df = _ids(4000)
    ua = df.select(hash_uniform(seed, "id").alias("u"))["u"].to_numpy()
    ub = df.select(hash_uniform(seed + offset, "id").alias("u"))["u"].to_numpy()
    assert abs(np.corrcoef(ua, ub)[0, 1]) < 0.09


@given(
    seed=st.integers(min_value=0, max_value=10000),
    f1=st.floats(min_value=0.0, max_value=1.0),
    f2=st.floats(min_value=0.0, max_value=1.0),
)
@_PROP
def test_hash_split_consistent_property(seed, f1, f2):
    # Consistency: growing the fraction only adds members (the smaller holdout is a
    # subset of the larger), for any seed.
    lo, hi = sorted((f1, f2))
    df = _ids(5000)
    small = df.select(hash_split(seed, "id", fraction=lo).alias("h"))["h"].to_numpy()
    big = df.select(hash_split(seed, "id", fraction=hi).alias("h"))["h"].to_numpy()
    assert (small <= big).all()


@given(
    seed=st.integers(min_value=0, max_value=10000),
    shuf=st.integers(min_value=0, max_value=10000),
    f=st.floats(min_value=0.0, max_value=1.0),
)
@_PROP
def test_hash_split_order_invariant_property(seed, shuf, f):
    # Keyed on the id -> reproducible regardless of row order (no leakage).
    df = _ids(3000)
    base = df.select("id", hash_split(seed, "id", fraction=f).alias("h")).sort("id")
    shuffled = (
        df.sample(fraction=1.0, shuffle=True, seed=shuf)
        .select("id", hash_split(seed, "id", fraction=f).alias("h"))
        .sort("id")
    )
    assert (base["h"] == shuffled["h"]).all()


@given(seed=st.integers(min_value=0, max_value=10000), k=st.integers(min_value=1, max_value=50))
@_PROP
def test_hash_fold_in_range_property(seed, k):
    f = _ids(5000).select(hash_fold(seed, "id", k=k).alias("f"))["f"]
    assert f.min() >= 0
    assert f.max() <= k - 1


@given(
    seed=st.integers(min_value=0, max_value=10000),
    fracs=st.lists(st.floats(min_value=0.0, max_value=0.9), min_size=1, max_size=4),
)
@_PROP
def test_hash_splits_partition_and_upstream_stable_property(seed, fracs):
    # Keep room for the remainder, then check the two invariants: disjoint+cover, and
    # resizing the last split leaves every upstream split's membership unchanged.
    total = sum(fracs)
    if total > 0.9:
        fracs = [f * 0.9 / total for f in fracs]
    spec = [(f"s{i}", f) for i, f in enumerate(fracs)]
    df = _ids(4000)
    a = df.select(hash_splits(seed, "id", spec, remainder="rest").alias("s"))["s"]
    assert a.null_count() == 0  # every row labelled
    assert set(a.unique()) <= {name for name, _ in spec} | {"rest"}
    if len(spec) >= 2:
        spec2 = [*spec[:-1], (spec[-1][0], spec[-1][1] + 0.05)]
        b = df.select(hash_splits(seed, "id", spec2, remainder="rest").alias("s"))["s"]
        for name, _ in spec[:-1]:
            assert ((a == name).to_numpy() == (b == name).to_numpy()).all()
