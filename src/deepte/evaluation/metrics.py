"""Evaluation metrics for DeepTE-Peptide."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the Precision-Recall curve (average precision)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def f1_max(y_true: np.ndarray, y_score: np.ndarray, n_thresh: int = 100) -> tuple[float, float]:
    """F1 score maximised over thresholds.

    Returns
    -------
    (best_f1, best_threshold)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresh)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_f1), float(best_t)


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews correlation coefficient."""
    return float(matthews_corrcoef(y_true, y_pred))


def spearman_corr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Spearman rank correlation."""
    rho, _ = spearmanr(y_true, y_score)
    return float(rho) if not np.isnan(rho) else 0.0


def kendall_corr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kendall's tau."""
    tau, _ = kendalltau(y_true, y_score)
    return float(tau) if not np.isnan(tau) else 0.0


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> float:
    """Precision in the top-k highest-scoring predictions."""
    if k > len(y_true):
        k = len(y_true)
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[top_k_idx]))


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for any metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            boot_vals.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    boot_vals = np.array(boot_vals)
    boot_vals = boot_vals[~np.isnan(boot_vals)]
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(boot_vals, 100 * alpha))
    upper = float(np.percentile(boot_vals, 100 * (1 - alpha)))
    point = float(metric_fn(y_true, y_score))
    return point, lower, upper


def compute_all_metrics(
    y_true: np.ndarray, y_score: np.ndarray, k_values: Iterable[int] = (100, 1000)
) -> dict:
    """Compute all standard metrics."""
    out = {
        "auc_roc": auc_roc(y_true, y_score),
        "auc_pr": auc_pr(y_true, y_score),
        "spearman": spearman_corr(y_true, y_score),
        "kendall": kendall_corr(y_true, y_score),
    }
    best_f1, best_t = f1_max(y_true, y_score)
    out["f1_max"] = best_f1
    out["f1_threshold"] = best_t
    y_pred = (y_score >= best_t).astype(int)
    out["mcc"] = mcc(y_true, y_pred)
    for k in k_values:
        out[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k=k)
    return out
