"""Tests for weighted variants of all metrics against sklearn."""

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.metrics import (
    f1_score as sklearn_f1,
)
from sklearn.metrics import (
    fbeta_score as sklearn_fbeta,
)
from sklearn.metrics import (
    log_loss as sklearn_log_loss,
)
from sklearn.metrics import (
    precision_score as sklearn_precision,
)
from sklearn.metrics import (
    recall_score as sklearn_recall,
)

from polarbear import (
    accuracy,
    average_precision,
    balanced_accuracy,
    brier_score,
    f1_score,
    fbeta_score,
    log_loss,
    precision,
    recall,
    roc_auc,
)


@pytest.fixture()
def weighted_binary_data():
    """Common weighted binary classification data."""
    np.random.seed(42)
    n = 200
    labels = np.random.randint(0, 2, n)
    probs = labels * 0.6 + np.random.rand(n) * 0.4
    probs = np.clip(probs, 0.01, 0.99)
    weights = np.random.rand(n) + 0.5
    df = pl.DataFrame({"label": labels, "prob": probs, "w": weights})
    return df, labels, probs, weights


class TestWeightedROCAUC:
    def test_matches_sklearn(self, weighted_binary_data):
        df, labels, probs, weights = weighted_binary_data
        result = df.select(roc_auc("label", "prob", weight="w")).to_series()[0]
        expected = roc_auc_score(labels, probs, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-3)

    def test_uniform_weights_match_unweighted(self):
        np.random.seed(42)
        labels = np.random.randint(0, 2, 100)
        scores = np.random.rand(100)
        df = pl.DataFrame({"label": labels, "score": scores, "w": [1.0] * 100})
        weighted = df.select(roc_auc("label", "score", weight="w")).to_series()[0]
        unweighted = df.select(roc_auc("label", "score")).to_series()[0]
        assert weighted == pytest.approx(unweighted, rel=1e-3)


class TestWeightedLogLoss:
    def test_matches_sklearn(self, weighted_binary_data):
        df, labels, probs, weights = weighted_binary_data
        result = df.select(log_loss("label", "prob", weight="w")).to_series()[0]
        expected = sklearn_log_loss(labels, probs, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_uniform_weights_match_unweighted(self):
        np.random.seed(42)
        labels = np.random.randint(0, 2, 50)
        probs = np.clip(np.random.rand(50), 0.01, 0.99)
        df = pl.DataFrame({"label": labels, "prob": probs, "w": [1.0] * 50})
        weighted = df.select(log_loss("label", "prob", weight="w")).to_series()[0]
        unweighted = df.select(log_loss("label", "prob")).to_series()[0]
        assert weighted == pytest.approx(unweighted, rel=1e-5)


class TestWeightedBrierScore:
    def test_matches_sklearn(self, weighted_binary_data):
        df, labels, probs, weights = weighted_binary_data
        result = df.select(brier_score("label", "prob", weight="w")).to_series()[0]
        expected = brier_score_loss(labels, probs, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_uniform_weights_match_unweighted(self):
        np.random.seed(42)
        labels = np.random.randint(0, 2, 50)
        probs = np.random.rand(50)
        df = pl.DataFrame({"label": labels, "prob": probs, "w": [1.0] * 50})
        weighted = df.select(brier_score("label", "prob", weight="w")).to_series()[0]
        unweighted = df.select(brier_score("label", "prob")).to_series()[0]
        assert weighted == pytest.approx(unweighted, rel=1e-5)


class TestWeightedAveragePrecision:
    def test_matches_sklearn(self, weighted_binary_data):
        df, labels, probs, weights = weighted_binary_data
        result = df.select(average_precision("label", "prob", weight="w")).to_series()[0]
        expected = average_precision_score(labels, probs, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-3)

    def test_uniform_weights_match_unweighted(self):
        np.random.seed(42)
        labels = np.random.randint(0, 2, 100)
        scores = np.random.rand(100)
        df = pl.DataFrame({"label": labels, "score": scores, "w": [1.0] * 100})
        weighted = df.select(average_precision("label", "score", weight="w")).to_series()[0]
        unweighted = df.select(average_precision("label", "score")).to_series()[0]
        assert weighted == pytest.approx(unweighted, rel=1e-5)


class TestWeightedClassification:
    def test_weighted_precision_matches_sklearn(self, weighted_binary_data):
        df, labels, probs, weights = weighted_binary_data
        threshold = 0.5
        result = df.select(precision("label", "prob", threshold=threshold, weight="w")).to_series()[
            0
        ]
        preds = (probs >= threshold).astype(int)
        expected = sklearn_precision(labels, preds, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_weighted_recall_matches_sklearn(self, weighted_binary_data):
        df, labels, probs, weights = weighted_binary_data
        threshold = 0.5
        result = df.select(recall("label", "prob", threshold=threshold, weight="w")).to_series()[0]
        preds = (probs >= threshold).astype(int)
        expected = sklearn_recall(labels, preds, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_weighted_f1_matches_sklearn(self, weighted_binary_data):
        df, labels, probs, weights = weighted_binary_data
        threshold = 0.5
        result = df.select(f1_score("label", "prob", threshold=threshold, weight="w")).to_series()[
            0
        ]
        preds = (probs >= threshold).astype(int)
        expected = sklearn_f1(labels, preds, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_weighted_fbeta_matches_sklearn(self, weighted_binary_data):
        df, labels, probs, weights = weighted_binary_data
        threshold = 0.5
        for beta in (0.5, 2.0):
            result = df.select(
                fbeta_score("label", "prob", beta=beta, threshold=threshold, weight="w")
            ).to_series()[0]
            preds = (probs >= threshold).astype(int)
            expected = sklearn_fbeta(labels, preds, beta=beta, sample_weight=weights)
            assert result == pytest.approx(expected, rel=1e-4)

    def test_weighted_accuracy_matches_sklearn(self, weighted_binary_data):
        df, labels, probs, weights = weighted_binary_data
        threshold = 0.5
        result = df.select(accuracy("label", "prob", threshold=threshold, weight="w")).to_series()[
            0
        ]
        preds = (probs >= threshold).astype(int)
        expected = accuracy_score(labels, preds, sample_weight=weights)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_weighted_balanced_accuracy_matches_manual(self, weighted_binary_data):
        """Weighted balanced_accuracy: manual computation since sklearn doesn't support weights."""
        df, labels, probs, weights = weighted_binary_data
        threshold = 0.5
        result = df.select(
            balanced_accuracy("label", "prob", threshold=threshold, weight="w")
        ).to_series()[0]
        preds = (probs >= threshold).astype(int)
        # Manual: weighted TPR and TNR
        pos_mask = labels == 1
        neg_mask = labels == 0
        w_tp = (weights[pos_mask & (preds == 1)]).sum()
        w_fn = (weights[pos_mask & (preds == 0)]).sum()
        w_tn = (weights[neg_mask & (preds == 0)]).sum()
        w_fp = (weights[neg_mask & (preds == 1)]).sum()
        tpr = w_tp / (w_tp + w_fn)
        tnr = w_tn / (w_tn + w_fp)
        expected = (tpr + tnr) / 2
        assert result == pytest.approx(expected, rel=1e-4)

    def test_weighted_classification_varied_thresholds(self, weighted_binary_data):
        """Weighted precision/recall/f1 at multiple thresholds."""
        df, labels, probs, weights = weighted_binary_data
        for threshold in [0.1, 0.3, 0.7, 0.9]:
            preds = (probs >= threshold).astype(int)
            if preds.sum() == 0 or preds.sum() == len(preds):
                continue

            result_p = df.select(
                precision("label", "prob", threshold=threshold, weight="w")
            ).to_series()[0]
            expected_p = sklearn_precision(labels, preds, sample_weight=weights)
            assert result_p == pytest.approx(expected_p, rel=1e-4), f"precision at t={threshold}"

            result_r = df.select(
                recall("label", "prob", threshold=threshold, weight="w")
            ).to_series()[0]
            expected_r = sklearn_recall(labels, preds, sample_weight=weights)
            assert result_r == pytest.approx(expected_r, rel=1e-4), f"recall at t={threshold}"

            result_f = df.select(
                f1_score("label", "prob", threshold=threshold, weight="w")
            ).to_series()[0]
            expected_f = sklearn_f1(labels, preds, sample_weight=weights)
            assert result_f == pytest.approx(expected_f, rel=1e-4), f"f1 at t={threshold}"


class TestWeightedGroupBy:
    """Weighted metrics in group_by context — the most dangerous path."""

    def test_weighted_roc_auc_group_by(self):
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "group": ["A"] * 50 + ["B"] * 50,
                "label": np.random.randint(0, 2, 100).tolist(),
                "score": np.random.rand(100).tolist(),
                "w": (np.random.rand(100) + 0.5).tolist(),
            }
        )
        result = df.group_by("group").agg(roc_auc("label", "score", weight="w")).sort("group")

        for i, grp in enumerate(["A", "B"]):
            grp_df = df.filter(pl.col("group") == grp)
            expected = roc_auc_score(
                grp_df["label"].to_numpy(),
                grp_df["score"].to_numpy(),
                sample_weight=grp_df["w"].to_numpy(),
            )
            assert result["roc_auc_label_score_w"][i] == pytest.approx(expected, rel=1e-3)

    def test_weighted_average_precision_group_by(self):
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "group": ["A"] * 50 + ["B"] * 50,
                "label": np.random.randint(0, 2, 100).tolist(),
                "score": np.random.rand(100).tolist(),
                "w": (np.random.rand(100) + 0.5).tolist(),
            }
        )
        result = (
            df.group_by("group").agg(average_precision("label", "score", weight="w")).sort("group")
        )

        for i, grp in enumerate(["A", "B"]):
            grp_df = df.filter(pl.col("group") == grp)
            expected = average_precision_score(
                grp_df["label"].to_numpy(),
                grp_df["score"].to_numpy(),
                sample_weight=grp_df["w"].to_numpy(),
            )
            assert result["average_precision_label_score_w"][i] == pytest.approx(expected, rel=1e-3)

    def test_weighted_log_loss_group_by(self):
        np.random.seed(42)
        labels = np.random.randint(0, 2, 100)
        probs = np.clip(labels * 0.6 + np.random.rand(100) * 0.4, 0.01, 0.99)
        df = pl.DataFrame(
            {
                "group": ["A"] * 50 + ["B"] * 50,
                "label": labels.tolist(),
                "prob": probs.tolist(),
                "w": (np.random.rand(100) + 0.5).tolist(),
            }
        )
        result = df.group_by("group").agg(log_loss("label", "prob", weight="w")).sort("group")

        for i, grp in enumerate(["A", "B"]):
            grp_df = df.filter(pl.col("group") == grp)
            expected = sklearn_log_loss(
                grp_df["label"].to_numpy(),
                grp_df["prob"].to_numpy(),
                sample_weight=grp_df["w"].to_numpy(),
            )
            assert result["log_loss_label_prob_w"][i] == pytest.approx(expected, rel=1e-4)

    def test_weighted_brier_score_group_by(self):
        np.random.seed(42)
        labels = np.random.randint(0, 2, 100)
        probs = np.random.rand(100)
        df = pl.DataFrame(
            {
                "group": ["A"] * 50 + ["B"] * 50,
                "label": labels.tolist(),
                "prob": probs.tolist(),
                "w": (np.random.rand(100) + 0.5).tolist(),
            }
        )
        result = df.group_by("group").agg(brier_score("label", "prob", weight="w")).sort("group")

        for i, grp in enumerate(["A", "B"]):
            grp_df = df.filter(pl.col("group") == grp)
            expected = brier_score_loss(
                grp_df["label"].to_numpy(),
                grp_df["prob"].to_numpy(),
                sample_weight=grp_df["w"].to_numpy(),
            )
            assert result["brier_score_label_prob_w"][i] == pytest.approx(expected, rel=1e-4)
