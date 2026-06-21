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
