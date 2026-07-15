"""Metricas de clasificacion para validacion simple."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def classification_report_dict(y_true, y_pred) -> dict[str, object]:
    """Devuelve métricas principales de clasificación sin matriz de confusión."""

    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    true_labels = np.unique(truth)
    balanced_accuracy = float(
        np.mean([np.mean(prediction[truth == label] == label) for label in true_labels])
    )
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
