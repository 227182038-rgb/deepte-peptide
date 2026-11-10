"""Unit tests for evaluation metrics and evaluator."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepte.evaluation.metrics import (
    auc_pr,
    auc_roc,
    bootstrap_ci,
    compute_all_metrics,
    f1_max,
    kendall_corr,
    mcc,
    precision_at_k,
    spearman_corr,
)


class TestMetrics:
    def test_auc_roc(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.6, 0.9])
        assert auc_roc(y_true, y_score) == 1.0

    def test_auc_roc_single_class(self):
        y_true = np.array([1, 1, 1, 1])
        y_score = np.array([0.1, 0.4, 0.6, 0.9])
        assert np.isnan(auc_roc(y_true, y_score))

    def test_auc_pr(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, 0.8, 0.2, 0.9])
        assert auc_pr(y_true, y_score) > 0.8

    def test_f1_max(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        best_f1, best_t = f1_max(y_true, y_score)
        assert best_f1 == 1.0
        assert 0.2 < best_t <= 0.8

    def test_mcc(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert mcc(y_true, y_pred) == 1.0

    def test_correlations(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_score = np.array([1.1, 1.9, 3.2, 4.1, 4.9])
        assert spearman_corr(y_true, y_score) > 0.9
        assert kendall_corr(y_true, y_score) > 0.9

    def test_precision_at_k(self):
        y_true = np.array([0, 0, 1, 1, 0])
        y_score = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
        # Top-2 are the two positives
        assert precision_at_k(y_true, y_score, k=2) == 1.0

    def test_bootstrap_ci(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_score = np.array([0.1, 0.9, 0.2, 0.85, 0.15, 0.75, 0.3, 0.95])
        point, low, high = bootstrap_ci(
            y_true, y_score, lambda yt, ys: float(np.mean(ys[yt == 1])), n_bootstrap=100
        )
        assert low <= point <= high

    def test_compute_all_metrics(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_score = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])
        res = compute_all_metrics(y_true, y_score)
        assert "auc_roc" in res
        assert "auc_pr" in res
        assert "f1_max" in res
        assert "mcc" in res
        assert res["auc_roc"] == 1.0
