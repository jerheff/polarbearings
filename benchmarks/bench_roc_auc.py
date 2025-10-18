"""Benchmarks for ROC AUC implementation."""

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from polarbear import roc_auc


class TestROCAUCPerformance:
    """Performance benchmarks for ROC AUC."""

    @pytest.fixture(params=[100, 1000, 10000, 100000])
    def data(self, request):
        """Generate test data of various sizes."""
        n = request.param
        np.random.seed(42)
        labels = np.random.randint(0, 2, n)
        scores = labels * 0.6 + np.random.randn(n) * 0.3
        return labels, scores, n

    def test_polarbear_roc_auc(self, benchmark, data):
        """Benchmark polarbear ROC AUC implementation."""
        labels, scores, n = data
        df = pl.DataFrame({"label": labels, "score": scores})

        def compute_auc():
            return df.select(roc_auc("label", "score")).to_series()[0]

        result = benchmark(compute_auc)
        assert 0.0 <= result <= 1.0

    def test_sklearn_roc_auc(self, benchmark, data):
        """Benchmark sklearn ROC AUC for comparison."""
        labels, scores, n = data

        def compute_auc():
            return roc_auc_score(labels, scores)

        result = benchmark(compute_auc)
        assert 0.0 <= result <= 1.0


class TestGroupByPerformance:
    """Benchmarks for group-by aggregations."""

    @pytest.fixture(params=[10, 100])
    def grouped_data(self, request):
        """Generate grouped data."""
        n_groups = request.param
        samples_per_group = 1000
        np.random.seed(42)

        groups = np.repeat(np.arange(n_groups), samples_per_group)
        labels = np.random.randint(0, 2, n_groups * samples_per_group)
        scores = labels * 0.6 + np.random.randn(n_groups * samples_per_group) * 0.3

        return groups, labels, scores, n_groups

    def test_polarbear_grouped_auc(self, benchmark, grouped_data):
        """Benchmark polarbear group-by ROC AUC."""
        groups, labels, scores, n_groups = grouped_data
        df = pl.DataFrame(
            {"group": groups, "label": labels, "score": scores}
        )

        def compute_grouped_auc():
            return df.group_by("group").agg(roc_auc("label", "score"))

        result = benchmark(compute_grouped_auc)
        assert len(result) == n_groups

    def test_sklearn_grouped_auc(self, benchmark, grouped_data):
        """Benchmark sklearn group-by ROC AUC for comparison."""
        groups, labels, scores, n_groups = grouped_data

        def compute_grouped_auc():
            results = []
            for group in range(n_groups):
                mask = groups == group
                group_labels = labels[mask]
                group_scores = scores[mask]
                if len(np.unique(group_labels)) > 1:
                    auc = roc_auc_score(group_labels, group_scores)
                else:
                    auc = 0.5
                results.append(auc)
            return results

        result = benchmark(compute_grouped_auc)
        assert len(result) == n_groups


class TestTiedScoresPerformance:
    """Benchmarks for datasets with many tied scores."""

    @pytest.fixture(params=[1000, 10000])
    def tied_data(self, request):
        """Generate data with many ties."""
        n = request.param
        np.random.seed(42)
        labels = np.random.randint(0, 2, n)
        # Create scores with only 10 unique values (lots of ties)
        scores = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], n)
        return labels, scores

    def test_polarbear_tied_scores(self, benchmark, tied_data):
        """Benchmark polarbear with tied scores."""
        labels, scores = tied_data
        df = pl.DataFrame({"label": labels, "score": scores})

        def compute_auc():
            return df.select(roc_auc("label", "score")).to_series()[0]

        result = benchmark(compute_auc)
        assert 0.0 <= result <= 1.0

    def test_sklearn_tied_scores(self, benchmark, tied_data):
        """Benchmark sklearn with tied scores."""
        labels, scores = tied_data

        def compute_auc():
            return roc_auc_score(labels, scores)

        result = benchmark(compute_auc)
        assert 0.0 <= result <= 1.0
