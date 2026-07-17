"""Tests for DCG / NDCG ranking metrics."""

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sklearn.metrics import dcg_score as sk_dcg
from sklearn.metrics import ndcg_score as sk_ndcg

from polarbearings.ranking import dcg_score, ndcg_score


class TestDCG:
    def test_known_value(self):
        # rel sorted by score desc = [2, 3, 1]; discounts 1/log2(2,3,4).
        df = pl.DataFrame({"rel": [2.0, 3.0, 1.0], "score": [0.9, 0.5, 0.1]})
        expected = 2 / np.log2(2) + 3 / np.log2(3) + 1 / np.log2(4)
        result = df.select(dcg_score("rel", "score")).to_series()[0]
        assert result == pytest.approx(expected)

    def test_matches_sklearn(self):
        rng = np.random.default_rng(0)
        rel = rng.integers(0, 5, 15).astype(float)
        score = rng.permutation(15).astype(float)
        df = pl.DataFrame({"rel": rel, "score": score})
        for k in (None, 1, 5, 15):
            got = df.select(dcg_score("rel", "score", k=k)).to_series()[0]
            exp = sk_dcg([rel], [score], k=k, ignore_ties=True)
            assert got == pytest.approx(exp)

    def test_top_k_truncates(self):
        df = pl.DataFrame({"rel": [3.0, 2.0, 1.0], "score": [3.0, 2.0, 1.0]})
        full = df.select(dcg_score("rel", "score")).to_series()[0]
        top1 = df.select(dcg_score("rel", "score", k=1)).to_series()[0]
        assert top1 == pytest.approx(3.0)
        assert full > top1

    def test_log_base(self):
        df = pl.DataFrame({"rel": [1.0, 1.0], "score": [2.0, 1.0]})
        # base e: 1/ln(2) + 1/ln(3)
        result = df.select(dcg_score("rel", "score", log_base=np.e)).to_series()[0]
        assert result == pytest.approx(1 / np.log(2) + 1 / np.log(3))


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        df = pl.DataFrame({"rel": [3.0, 2.0, 1.0, 0.0], "score": [4.0, 3.0, 2.0, 1.0]})
        result = df.select(ndcg_score("rel", "score")).to_series()[0]
        assert result == pytest.approx(1.0)

    def test_matches_sklearn(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            rel = rng.integers(0, 4, 10).astype(float)
            if rel.sum() == 0:
                continue
            score = rng.permutation(10).astype(float)
            df = pl.DataFrame({"rel": rel, "score": score})
            for k in (None, 3):
                got = df.select(ndcg_score("rel", "score", k=k)).to_series()[0]
                exp = sk_ndcg([rel], [score], k=k, ignore_ties=True)
                assert got == pytest.approx(exp)

    def test_all_irrelevant_is_null(self):
        df = pl.DataFrame({"rel": [0.0, 0.0, 0.0], "score": [1.0, 2.0, 3.0]})
        assert df.select(ndcg_score("rel", "score")).to_series()[0] is None

    def test_in_unit_range(self):
        rng = np.random.default_rng(3)
        rel = rng.integers(0, 5, 30).astype(float)
        score = rng.random(30)
        df = pl.DataFrame({"rel": rel, "score": score})
        result = df.select(ndcg_score("rel", "score")).to_series()[0]
        assert 0.0 <= result <= 1.0

    def test_expression_columns_match_string_columns(self):
        df = pl.DataFrame({"rel": [3.0, 2.0, 1.0, 0.0], "score": [0.5, 0.9, 0.2, 0.1]})
        from_str = df.select(ndcg_score("rel", "score")).to_series()[0]
        from_expr = df.select(ndcg_score(pl.col("rel"), pl.col("score"))).to_series()[0]
        assert from_expr == pytest.approx(from_str)


class TestRankingGrouped:
    def test_group_by_matches_sklearn(self):
        rng = np.random.default_rng(4)
        rows = []
        for q in range(4):
            n = int(rng.integers(4, 9))
            for rel, sc in zip(
                rng.integers(0, 4, n), rng.permutation(n).astype(float), strict=True
            ):
                rows.append({"q": q, "rel": float(rel), "score": float(sc)})
        df = pl.DataFrame(rows)
        out = df.group_by("q").agg(ndcg_score("rel", "score")).sort("q")
        for row in out.to_dicts():
            sub = df.filter(pl.col("q") == row["q"])
            exp = sk_ndcg([sub["rel"].to_numpy()], [sub["score"].to_numpy()], ignore_ties=True)
            assert row["ndcg_rel_score"] == pytest.approx(exp)


class TestRankingEdgeCases:
    def test_single_row(self):
        df = pl.DataFrame({"rel": [2.0], "score": [0.5]})
        # one document: DCG = gain / log2(2) = gain; NDCG = 1.0 (already ideal).
        assert df.select(dcg_score("rel", "score")).to_series()[0] == pytest.approx(2.0)
        assert df.select(ndcg_score("rel", "score")).to_series()[0] == pytest.approx(1.0)

    @pytest.mark.parametrize("metric", [dcg_score, ndcg_score])
    @pytest.mark.parametrize("k", [0, -1, -5])
    def test_nonpositive_k_raises_eagerly(self, metric, k):
        # k<1 must fail at build time with a named error, not a collect-time Polars
        # ComputeError (k=-1) or a silent head(-n) miscount for other negatives.
        with pytest.raises(ValueError, match="k must be a positive integer"):
            metric("rel", "score", k=k)

    @pytest.mark.parametrize("metric", [dcg_score, ndcg_score])
    def test_valid_k_and_none_accepted(self, metric):
        df = pl.DataFrame({"rel": [3.0, 2.0, 1.0], "score": [0.9, 0.5, 0.1]})
        for k in (1, 3, 100, None):  # k past the row count is fine (head clamps)
            df.select(metric("rel", "score", k=k))


class TestTieHandling:
    """DCG/NDCG break score ties by physical row order.

    ``sort_by`` is stable on equal keys, so when scores tie the input row order
    decides each document's rank. This is *not* sklearn's gain-averaging
    (``ignore_ties=False``) and it also diverges from ``ignore_ties=True`` under
    ties — a single-expression design cannot gain-average. These tests lock the
    documented, order-dependent value for a fixed input order so the behavior
    can't drift silently; callers needing reproducibility under ties must
    sort/break ties upstream.
    """

    def test_dcg_ties_follow_row_order(self):
        # Both documents tie at score=1.0; the gain in the first row takes rank 0
        # (discount 1.0), the second takes rank 1 (discount 1/log2(3)).
        forward = pl.DataFrame({"rel": [3.0, 0.0], "score": [1.0, 1.0]})
        reverse = pl.DataFrame({"rel": [0.0, 3.0], "score": [1.0, 1.0]})
        fwd = forward.select(dcg_score("rel", "score")).to_series()[0]
        rev = reverse.select(dcg_score("rel", "score")).to_series()[0]
        assert fwd == pytest.approx(3.0)
        assert rev == pytest.approx(3.0 / np.log2(3))
        assert fwd != pytest.approx(rev)  # order-dependent, as documented

    def test_ndcg_ties_follow_row_order(self):
        # IDCG (gains sorted by themselves) = 3.0, so NDCG = DCG / 3.0.
        forward = pl.DataFrame({"rel": [3.0, 0.0], "score": [1.0, 1.0]})
        reverse = pl.DataFrame({"rel": [0.0, 3.0], "score": [1.0, 1.0]})
        fwd = forward.select(ndcg_score("rel", "score")).to_series()[0]
        rev = reverse.select(ndcg_score("rel", "score")).to_series()[0]
        assert fwd == pytest.approx(1.0)
        assert rev == pytest.approx(1.0 / np.log2(3))


@given(
    # size >= 2: sklearn's ndcg_score rejects a single document (our impl handles
    # it, covered in TestRankingEdgeCases).
    size=st.integers(min_value=2, max_value=40),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(deadline=None, database=None, suppress_health_check=[HealthCheck.differing_executors])
def test_ndcg_against_sklearn(size, seed):
    rng = np.random.default_rng(seed)
    rel = rng.integers(0, 5, size).astype(float)
    # Distinct scores avoid tie-ordering ambiguity (we match ignore_ties=True).
    score = rng.permutation(size).astype(float)
    df = pl.DataFrame({"rel": rel, "score": score})
    got = df.select(ndcg_score("rel", "score")).to_series()[0]
    if rel.sum() == 0:
        assert got is None
    else:
        exp = sk_ndcg([rel], [score], ignore_ties=True)
        assert got == pytest.approx(exp)
