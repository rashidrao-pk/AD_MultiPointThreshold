import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def _has_two_classes(y_true):
    """Return whether binary labels contain both normal and anomalous classes."""
    return len(np.unique(np.asarray(y_true).astype(int))) >= 2


def ranking_metrics(scores, y_true):
    """Compute ranking metrics from anomaly scores and binary labels."""
    scores = np.asarray(scores)
    y_true = np.asarray(y_true).astype(int)
    if not _has_two_classes(y_true):
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
        }

    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


def compute_auc(scores, y_true):
    """Compute AUROC from anomaly scores and binary labels."""
    scores = np.asarray(scores)
    y_true = np.asarray(y_true).astype(int)
    if not _has_two_classes(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def compute_roc(scores, y_true):
    """Compute ROC curve points from anomaly scores and binary labels."""
    scores = np.asarray(scores)
    y_true = np.asarray(y_true).astype(int)
    if not _has_two_classes(y_true):
        return {
            "fpr": np.asarray([]),
            "tpr": np.asarray([]),
            "thresholds": np.asarray([]),
        }
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def compute_confusion_matrix(y_pred, y_true):
    """Compute a 2x2 confusion matrix for binary predictions and labels."""
    y_pred = np.asarray(y_pred).astype(int)
    y_true = np.asarray(y_true).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def threshold_metrics(y_pred, y_true):
    """Compute binary classification metrics from threshold predictions."""

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fn": int(fn),
        "fp": int(fp),
    }
