"""Modelos simples para validar separabilidad, no para detector operacional."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_dummy_classifier() -> Pipeline:
    """Baseline nulo que predice según la clase mayoritaria de entrenamiento."""

    return Pipeline(steps=[("dummy", DummyClassifier(strategy="prior"))])


def make_linear_classifier(random_state: int = 42) -> Pipeline:
    """Clasificador lineal simple para validar separabilidad de features."""

    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, random_state=random_state, class_weight="balanced"
                ),
            ),
        ]
    )


def make_baseline_classifier(random_state: int = 42) -> Pipeline:
    """Compatibilidad con flujos previos: devuelve el modelo lineal simple."""

    return make_linear_classifier(random_state=random_state)
