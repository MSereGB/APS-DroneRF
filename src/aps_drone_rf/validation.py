"""Particiones y validacion por grupos para evitar leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold

from aps_drone_rf.metrics import classification_report_dict
from aps_drone_rf.models import make_dummy_classifier, make_linear_classifier


def _make_classifier(kind: str, random_state: int):
    """Construye el modelo pedido para la validación secundaria."""

    normalized = kind.strip().lower()
    if normalized == "dummy":
        return make_dummy_classifier()
    if normalized in {"lineal", "logistica"}:
        return make_linear_classifier(random_state=random_state)
    raise ValueError(f"Modelo no soportado: {kind}")


def assert_no_group_leakage(train_idx, test_idx, groups) -> None:
    """Lanza error si un grupo aparece simultaneamente en train y test."""

    group_array = np.asarray(groups)
    train_groups = set(group_array[np.asarray(train_idx)])
    test_groups = set(group_array[np.asarray(test_idx)])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise ValueError(f"Leakage de grupos detectado: {sorted(overlap)[:5]}")


def assert_group_labels_consistent(
    df: pd.DataFrame, group_col: str = "group_id", label_col: str = "label"
) -> None:
    """Verifica que cada grupo pertenezca a una sola clase."""

    if group_col not in df or label_col not in df:
        return
    label_counts = df.groupby(group_col)[label_col].nunique(dropna=False)
    conflicting = label_counts[label_counts > 1]
    if not conflicting.empty:
        examples = conflicting.index.astype(str).tolist()[:5]
        raise ValueError(f"Grupos asociados a mas de una etiqueta: {examples}")


def limit_windows_per_group(
    df: pd.DataFrame,
    max_windows_per_group: int | None = 200,
    group_col: str = "group_id",
    random_state: int = 42,
) -> pd.DataFrame:
    """Limita ventanas por grupo para validaciones livianas y reproducibles."""

    if max_windows_per_group is None or max_windows_per_group <= 0:
        return df.copy().reset_index(drop=True)
    if group_col not in df:
        raise ValueError(f"No existe la columna de grupos: {group_col}")

    rng = np.random.default_rng(random_state)
    parts = []
    for _, group in df.groupby(group_col, sort=False):
        if len(group) > max_windows_per_group:
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            parts.append(group.sample(n=max_windows_per_group, random_state=seed))
        else:
            parts.append(group)
    return pd.concat(parts, ignore_index=True)


def group_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = "group_id",
    label_col: str = "label",
    max_attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Particion train/test simple respetando grupos e intentando conservar clases."""

    assert_group_labels_consistent(df, group_col=group_col, label_col=label_col)
    splitter = GroupShuffleSplit(
        n_splits=max_attempts, test_size=test_size, random_state=random_state
    )
    y = df[label_col].to_numpy() if label_col in df else np.zeros(len(df))
    groups = df[group_col].to_numpy()
    all_labels = set(np.unique(y))
    for train_idx, test_idx in splitter.split(df, y, groups=groups):
        assert_no_group_leakage(train_idx, test_idx, groups)
        if len(all_labels) <= 1:
            return train_idx, test_idx
        train_labels = set(np.unique(y[train_idx]))
        test_labels = set(np.unique(y[test_idx]))
        if all_labels.issubset(train_labels) and all_labels.issubset(test_labels):
            return train_idx, test_idx
    raise ValueError(
        "No se pudo construir una particion train/test por grupos con todas las clases "
        "en entrenamiento y prueba"
    )


def choose_group_cv(
    y,
    groups,
    preferred_splits: int = 20,
    random_state: int = 42,
) -> tuple[object, int, str]:
    """Elige StratifiedGroupKFold o GroupKFold con cantidad viable de folds."""

    labels = np.asarray(y)
    group_array = np.asarray(groups)
    unique_groups = np.unique(group_array)
    if len(unique_groups) < 2:
        raise ValueError("Se necesitan al menos dos grupos para validacion por grupos")

    group_labels = pd.DataFrame({"group": group_array, "label": labels}).drop_duplicates()
    groups_per_class = group_labels.groupby("label")["group"].nunique()
    max_splits = int(min(len(unique_groups), groups_per_class.min()))
    if max_splits >= 2:
        n_splits = _select_reportable_n_splits(max_splits, preferred_splits)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return splitter, n_splits, "StratifiedGroupKFold"

    n_splits = _select_reportable_n_splits(len(unique_groups), preferred_splits)
    if n_splits < 2:
        raise ValueError("No hay grupos suficientes para GroupKFold")
    splitter = GroupKFold(n_splits=n_splits)
    return splitter, n_splits, "GroupKFold"


def _select_reportable_n_splits(max_splits: int, preferred_splits: int) -> int:
    """Elige 20, 10 o 5 folds cuando sea viable; si no, usa el maximo posible."""

    preferred = min(int(preferred_splits), int(max_splits))
    for candidate in [20, 10, 5]:
        if candidate <= preferred:
            return candidate
    return preferred


def evaluate_train_test(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    group_col: str = "group_id",
    test_size: float = 0.2,
    random_state: int = 42,
    classifier_kind: str = "lineal",
) -> dict[str, object]:
    """Evalua un clasificador simple con train/test por grupos."""

    train_idx, test_idx = group_train_test_split(
        features_df,
        test_size=test_size,
        random_state=random_state,
        group_col=group_col,
        label_col=label_col,
    )
    model = _make_classifier(classifier_kind, random_state=random_state)
    x = features_df[feature_cols].to_numpy()
    y = features_df[label_col].to_numpy()
    model.fit(x[train_idx], y[train_idx])
    y_pred = model.predict(x[test_idx])
    report = classification_report_dict(y[test_idx], y_pred)
    report["n_train"] = int(len(train_idx))
    report["n_test"] = int(len(test_idx))
    report["train_groups"] = int(features_df.iloc[train_idx][group_col].nunique())
    report["test_groups"] = int(features_df.iloc[test_idx][group_col].nunique())
    report["modelo"] = classifier_kind
    return report


def evaluate_group_cv(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    group_col: str = "group_id",
    preferred_splits: int = 20,
    random_state: int = 42,
    classifier_kind: str = "lineal",
) -> dict[str, object]:
    """Evalua con validacion cruzada por grupos y predicciones out-of-fold."""

    x = features_df[feature_cols].to_numpy()
    y = features_df[label_col].to_numpy()
    groups = features_df[group_col].to_numpy()
    assert_group_labels_consistent(features_df, group_col=group_col, label_col=label_col)
    splitter, n_splits, splitter_name = choose_group_cv(
        y, groups, preferred_splits=preferred_splits, random_state=random_state
    )

    y_true_all = []
    y_pred_all = []
    fold_sizes = []
    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups=groups)):
        assert_no_group_leakage(train_idx, test_idx, groups)
        model = _make_classifier(classifier_kind, random_state=random_state + fold_idx)
        model.fit(x[train_idx], y[train_idx])
        y_pred = model.predict(x[test_idx])
        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(y_pred.tolist())
        fold_sizes.append(int(len(test_idx)))
        fold_report = classification_report_dict(y[test_idx], y_pred)
        fold_metrics.append(
            {
                "fold": int(fold_idx + 1),
                "n_test": int(len(test_idx)),
                "accuracy": fold_report["accuracy"],
                "balanced_accuracy": fold_report["balanced_accuracy"],
                "f1_macro": fold_report["f1_macro"],
            }
        )

    report = classification_report_dict(y_true_all, y_pred_all)
    report["splitter"] = splitter_name
    report["n_splits"] = int(n_splits)
    report["fold_sizes"] = fold_sizes
    report["fold_metrics"] = fold_metrics
    report["n_groups"] = int(pd.Series(groups).nunique())
    report["modelo"] = classifier_kind
    return report
