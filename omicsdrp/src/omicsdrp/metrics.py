"""Regression metrics used across training, validation and test evaluation.

Every evaluation call returns the *full* dict so nothing is silently dropped;
the recorder persists all of it.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
from scipy.stats import pearsonr, spearmanr


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, R^2, Pearson r and Spearman rho.

    Correlations are NaN-safe: with <2 points or zero variance they return 0.0
    rather than crashing, which matters for tiny smoke-test batches.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def _safe_corr(fn):
        if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0
        try:
            return float(fn(y_true, y_pred)[0])
        except Exception:
            return 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": _safe_corr(pearsonr),
        "spearman": _safe_corr(spearmanr),
        "n": int(y_true.size),
    }
