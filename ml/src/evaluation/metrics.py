import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec


def compute_all(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    sens, spec = sensitivity_specificity(y_true, y_pred)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1_score(y_true, y_pred)),
    }


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n: int = 1000,
                 seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx]))
        except ValueError:
            continue  # happens if resample has only one class
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
