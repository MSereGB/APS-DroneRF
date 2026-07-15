"""Calibración jerárquica por grupos para la demostración DroneRF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from aps_drone_rf.metrics import classification_report_dict
from aps_drone_rf.models import make_dummy_classifier, make_linear_classifier

SCORE_PREFIX = "puntaje__"


@dataclass(frozen=True)
class StageSpec:
    """Define una etapa y qué filas puede usar."""

    name: str
    label_column: str
    model_filter: str | None = None
    drone_only: bool = False


STAGES = {
    "actividad": StageSpec("actividad", "activity"),
    "modelo": StageSpec("modelo", "model", drone_only=True),
    "modo_bebop": StageSpec("modo_bebop", "mode", model_filter="bebop"),
    "modo_ar": StageSpec("modo_ar", "mode", model_filter="ar"),
}


def filter_stage(features: pd.DataFrame, spec: StageSpec) -> pd.DataFrame:
    """Selecciona filas de una etapa sin cambiar sus grupos."""

    selected = features
    if spec.drone_only:
        selected = selected[selected["activity"] == "dron"]
    if spec.model_filter is not None:
        selected = selected[selected["model"] == spec.model_filter]
    selected = selected.copy().reset_index(drop=True)
    if selected.empty or selected[spec.label_column].nunique() < 2:
        raise ValueError(f"La etapa {spec.name} no tiene clases suficientes")
    return selected


def _make_model(kind: str, random_state: int):
    if kind == "lineal":
        return make_linear_classifier(random_state=random_state)
    if kind == "dummy":
        return make_dummy_classifier()
    raise ValueError(f"Modelo no soportado: {kind}")


def _check_group_consistency(frame: pd.DataFrame, label_column: str) -> None:
    counts = frame.groupby("group_id")[label_column].nunique()
    if (counts != 1).any():
        raise ValueError("Un grupo tiene más de una etiqueta")
    if "partition" in frame:
        partitions = frame.groupby("group_id")["partition"].nunique()
        if (partitions != 1).any():
            raise ValueError("Un grupo aparece en más de una partición")


def aggregate_part_scores(
    score_rows: pd.DataFrame,
    *,
    label_column: str,
) -> pd.DataFrame:
    """Usa mediana por parte y luego igual peso para L/H dentro del grupo."""

    score_columns = [column for column in score_rows if column.startswith(SCORE_PREFIX)]
    if not score_columns:
        raise ValueError("No hay columnas de puntaje")
    _check_group_consistency(score_rows, label_column)
    metadata_columns = ["group_id", "part", label_column, "fold"]
    part_scores = (
        score_rows[metadata_columns + score_columns]
        .groupby(["group_id", "part"], as_index=False)
        .agg(
            {
                **{label_column: "first", "fold": "first"},
                **{column: "median" for column in score_columns},
            }
        )
    )
    group_scores = part_scores.groupby("group_id", as_index=False).agg(
        {
            **{label_column: "first", "fold": "first"},
            **{column: "mean" for column in score_columns},
        }
    )
    group_scores["parts_used"] = part_scores.groupby("group_id")["part"].nunique().values
    return group_scores


def grouped_oof_scores(
    features: pd.DataFrame,
    feature_columns: list[str],
    spec: StageSpec,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    model_kind: str = "lineal",
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Genera puntajes out-of-fold sin compartir un group_id entre train y test."""

    frame = filter_stage(features, spec)
    _check_group_consistency(frame, spec.label_column)
    x = frame[feature_columns].to_numpy(dtype=float)
    y = frame[spec.label_column].astype(str).to_numpy()
    groups = frame["group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    output_rows = []
    fold_summaries = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(x, y, groups=groups), start=1
    ):
        train_groups = set(groups[train_indices])
        test_groups = set(groups[test_indices])
        if train_groups.intersection(test_groups):
            raise ValueError("Leakage de grupos durante la calibración")
        model = _make_model(model_kind, random_state + fold)
        model.fit(x[train_indices], y[train_indices])
        probabilities = model.predict_proba(x[test_indices])
        classes = [str(value) for value in model.classes_]
        fold_rows = frame.iloc[test_indices][
            ["group_id", "part", spec.label_column]
        ].reset_index(drop=True)
        fold_rows["fold"] = fold
        for index, class_name in enumerate(classes):
            fold_rows[f"{SCORE_PREFIX}{class_name}"] = probabilities[:, index]
        output_rows.append(fold_rows)
        fold_summaries.append(
            {
                "fold": fold,
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "train_group_ids": sorted(train_groups),
                "test_group_ids": sorted(test_groups),
            }
        )

    score_rows = pd.concat(output_rows, ignore_index=True).fillna(0.0)
    group_scores = aggregate_part_scores(score_rows, label_column=spec.label_column)
    return group_scores, fold_summaries


def top_predictions(group_scores: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Obtiene hipótesis principal y margen respecto de la segunda."""

    score_columns = [column for column in group_scores if column.startswith(SCORE_PREFIX)]
    classes = np.array([column.removeprefix(SCORE_PREFIX) for column in score_columns])
    scores = group_scores[score_columns].to_numpy(dtype=float)
    order = np.argsort(scores, axis=1)
    best = order[:, -1]
    second = order[:, -2] if scores.shape[1] > 1 else order[:, -1]
    predictions = classes[best]
    margins = scores[np.arange(len(scores)), best] - scores[np.arange(len(scores)), second]
    return predictions, margins


def choose_binary_threshold(y_true, drone_scores) -> float:
    """Elige umbral por balanced accuracy usando solo puntajes OOF de desarrollo."""

    labels = np.asarray(y_true).astype(str)
    scores = np.asarray(drone_scores, dtype=float)
    candidates = np.unique(np.concatenate([[0.5], np.linspace(0.05, 0.95, 181), scores]))
    ranked = []
    for threshold in candidates:
        prediction = np.where(scores >= threshold, "dron", "fondo")
        metric = balanced_accuracy_score(labels, prediction)
        ranked.append((float(metric), -abs(float(threshold) - 0.5), -float(threshold)))
    best_index = max(range(len(ranked)), key=ranked.__getitem__)
    return float(candidates[best_index])


def choose_rejection_margin(
    y_true,
    predictions,
    margins,
    *,
    rejection_cost: float = 0.25,
    max_rejection: float = 0.30,
) -> float:
    """Calibra el margen con riesgo balanceado y un límite de rechazos."""

    truth = np.asarray(y_true).astype(str)
    predicted = np.asarray(predictions).astype(str)
    margin_values = np.asarray(margins, dtype=float)
    candidates = np.unique(np.concatenate([[0.0], np.linspace(0.0, 0.5, 101), margin_values]))
    classes = np.unique(truth)
    choices = []
    for threshold in candidates:
        rejected = margin_values < threshold
        rejection_rate = float(np.mean(rejected))
        if rejection_rate > max_rejection:
            continue
        class_risks = []
        for class_name in classes:
            mask = truth == class_name
            errors = (~rejected[mask]) & (predicted[mask] != truth[mask])
            risk = np.mean(errors.astype(float) + rejection_cost * rejected[mask].astype(float))
            class_risks.append(float(risk))
        choices.append((float(np.mean(class_risks)), float(threshold)))
    if not choices:
        return 0.0
    return min(choices, key=lambda value: (value[0], value[1]))[1]


def summarize_group_scores(
    group_scores: pd.DataFrame,
    spec: StageSpec,
    *,
    binary_threshold: float | None = None,
    rejection_margin: float = 0.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Calcula métricas por grupo y conserva los rechazos como no concluyentes."""

    result = group_scores.copy()
    truth = result[spec.label_column].astype(str).to_numpy()
    if spec.name == "actividad":
        if binary_threshold is None:
            raise ValueError("La etapa actividad requiere un umbral")
        drone_scores = result[f"{SCORE_PREFIX}dron"].to_numpy(dtype=float)
        prediction = np.where(drone_scores >= binary_threshold, "dron", "fondo")
        margins = np.abs(drone_scores - binary_threshold)
    else:
        prediction, margins = top_predictions(result)
    rejected = margins < rejection_margin
    visible_prediction = prediction.astype(object)
    visible_prediction[rejected] = "no_concluyente"
    result["prediction"] = visible_prediction
    result["margin"] = margins
    result["rejected"] = rejected

    report = classification_report_dict(truth, visible_prediction)
    report["coverage"] = float(np.mean(~rejected))
    report["rejection_rate"] = float(np.mean(rejected))
    report["n_groups"] = int(len(result))
    if spec.name == "actividad":
        drone_mask = truth == "dron"
        background_mask = truth == "fondo"
        detected = prediction[drone_mask] == "dron"
        false_alarms = prediction[background_mask] == "dron"
        report["detection_rate"] = float(np.mean(detected))
        report["false_alarm_rate"] = float(np.mean(false_alarms))
        report["drone_group_count"] = int(np.sum(drone_mask))
        report["background_group_count"] = int(np.sum(background_mask))
        report["detected_drone_count"] = int(np.sum(detected))
        report["missed_drone_count"] = int(np.sum(~detected))
        report["false_alarm_count"] = int(np.sum(false_alarms))
        report["true_negative_count"] = int(np.sum(~false_alarms))
        report["threshold"] = float(binary_threshold)
    report["rejection_margin"] = float(rejection_margin)
    return report, result


def fit_final_model(
    features: pd.DataFrame,
    feature_columns: list[str],
    spec: StageSpec,
    *,
    random_state: int = 42,
):
    """Ajusta el modelo congelado usando solamente desarrollo."""

    frame = filter_stage(features, spec)
    model = make_linear_classifier(random_state=random_state)
    model.fit(frame[feature_columns].to_numpy(dtype=float), frame[spec.label_column].astype(str))
    return model


def score_with_model(
    model,
    features: pd.DataFrame,
    feature_columns: list[str],
    spec: StageSpec,
) -> pd.DataFrame:
    """Puntúa una partición y combina partes sin usar sus etiquetas para predecir."""

    frame = filter_stage(features, spec)
    probabilities = model.predict_proba(frame[feature_columns].to_numpy(dtype=float))
    score_rows = frame[["group_id", "part", spec.label_column]].copy()
    score_rows["fold"] = 0
    for index, class_name in enumerate(model.classes_):
        score_rows[f"{SCORE_PREFIX}{class_name}"] = probabilities[:, index]
    return aggregate_part_scores(score_rows, label_column=spec.label_column)


def nested_binary_group_scores(
    features: pd.DataFrame,
    feature_columns: list[str],
    *,
    n_outer_splits: int = 5,
    n_inner_splits: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Evalúa actividad con umbral calibrado dentro de cada fold externo.

    Cada grupo de prueba externo queda fuera tanto del ajuste del clasificador como de
    la selección de umbral. Devuelve una fila por grupo y la traza de los folds para
    poder comprobar esa separación en pruebas y auditorías.
    """

    spec = STAGES["actividad"]
    frame = filter_stage(features, spec)
    _check_group_consistency(frame, spec.label_column)
    x = frame[feature_columns].to_numpy(dtype=float)
    y = frame[spec.label_column].astype(str).to_numpy()
    groups = frame["group_id"].astype(str).to_numpy()
    outer = StratifiedGroupKFold(
        n_splits=n_outer_splits,
        shuffle=True,
        random_state=random_state,
    )
    outputs = []
    summaries = []
    for outer_fold, (train_indices, test_indices) in enumerate(
        outer.split(x, y, groups=groups), start=1
    ):
        train_frame = frame.iloc[train_indices].copy().reset_index(drop=True)
        test_frame = frame.iloc[test_indices].copy().reset_index(drop=True)
        train_groups = set(train_frame["group_id"].astype(str))
        test_groups = set(test_frame["group_id"].astype(str))
        if train_groups.intersection(test_groups):
            raise ValueError("Leakage de grupos en el fold externo")

        inner_scores, inner_folds = grouped_oof_scores(
            train_frame,
            feature_columns,
            spec,
            n_splits=n_inner_splits,
            random_state=random_state + outer_fold,
        )
        threshold = choose_binary_threshold(
            inner_scores[spec.label_column],
            inner_scores[f"{SCORE_PREFIX}dron"],
        )
        model = fit_final_model(
            train_frame,
            feature_columns,
            spec,
            random_state=random_state + outer_fold,
        )
        outer_scores = score_with_model(model, test_frame, feature_columns, spec)
        _, predicted = summarize_group_scores(
            outer_scores,
            spec,
            binary_threshold=threshold,
        )
        predicted["outer_fold"] = outer_fold
        predicted["nested_threshold"] = threshold
        outputs.append(predicted)
        summaries.append(
            {
                "outer_fold": outer_fold,
                "train_group_ids": sorted(train_groups),
                "test_group_ids": sorted(test_groups),
                "threshold_group_ids": sorted(
                    inner_scores["group_id"].astype(str).unique()
                ),
                "threshold": threshold,
                "inner_folds": inner_folds,
            }
        )
    return pd.concat(outputs, ignore_index=True), summaries


def clone_model(model):
    """Helper usado en tests para comprobar que el bundle no se modifica al inferir."""

    return clone(model)
