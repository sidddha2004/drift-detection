"""Predictive-quality metrics. Macro-F1 is primary (class imbalance); accuracy is secondary."""
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict:
    """Macro-F1 is taken over the classes present in y_true or y_pred (sklearn default), so a client
    that never performed an activity is not penalised for it."""
    labels = np.arange(n_classes)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(),
                      "support": sup.tolist()},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def aggregate_clients(per_client: Dict[int, Dict]) -> Dict:
    """Mean / worst / std of client macro-F1 and accuracy (fairness group of the metric suite)."""
    f1 = np.array([m["macro_f1"] for m in per_client.values()])
    acc = np.array([m["accuracy"] for m in per_client.values()])
    return {
        "mean_macro_f1": float(f1.mean()), "worst_macro_f1": float(f1.min()),
        "std_macro_f1": float(f1.std()), "mean_accuracy": float(acc.mean()),
        "worst_accuracy": float(acc.min()),
    }
