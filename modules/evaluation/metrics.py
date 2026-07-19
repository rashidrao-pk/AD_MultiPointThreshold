import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def ranking_metrics(scores, y_true):
    """Compute ranking metrics from anomaly scores and binary labels."""
    scores = np.asarray(scores)
    y_true = np.asarray(y_true).astype(int)

    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


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
