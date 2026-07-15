"""Control de compatibilidad entre una entrada y el dominio de calibración."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vector_por_grupo(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Resume ventanas por parte y combina L/H con el mismo peso."""

    required = {"group_id", "part", *feature_columns}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Faltan columnas para el control de dominio: {sorted(missing)[:3]}")
    by_part = features.groupby(["group_id", "part"])[feature_columns].median()
    return by_part.groupby(level="group_id").mean()


def _robust_scale(values: pd.DataFrame) -> pd.Series:
    """Obtiene una escala robusta y evita divisiones por cero."""

    scale = values.quantile(0.75) - values.quantile(0.25)
    fallback = values.std(ddof=0)
    scale = scale.where(scale > 1e-12, fallback)
    return scale.where(scale > 1e-12, 1e-12)


def _distance(values: pd.DataFrame, center: pd.Series, scale: pd.Series) -> pd.Series:
    """Distancia RMS con features normalizadas respecto del dominio conocido."""

    z_values = (values - center) / scale
    return np.sqrt((z_values**2).mean(axis=1))


def calibrate_domain_guard(
    development: pd.DataFrame,
    feature_columns: list[str],
    *,
    safety_factor: float = 1.05,
) -> dict[str, object]:
    """Calibra el límite solo con grupos de desarrollo ya congelados."""

    if safety_factor < 1.0:
        raise ValueError("El factor de seguridad debe ser mayor o igual que uno")
    vectors = vector_por_grupo(development, feature_columns)
    if vectors.empty:
        raise ValueError("No hay grupos de desarrollo para calibrar el dominio")
    center = vectors.median()
    scale = _robust_scale(vectors)
    distances = _distance(vectors, center, scale)
    maximum = float(distances.max())
    threshold = max(maximum * safety_factor, 1e-12)
    return {
        "schema_version": "1.0",
        "purpose": "Rechazar entradas fuera del dominio de calibración DroneRF.",
        "feature_columns": list(feature_columns),
        "center": {column: float(center[column]) for column in feature_columns},
        "scale": {column: float(scale[column]) for column in feature_columns},
        "distance_metric": "rms_robust_z",
        "threshold": threshold,
        "safety_factor": float(safety_factor),
        "development_groups": int(vectors.index.nunique()),
        "development_distance_max": maximum,
    }


def check_domain(
    features: pd.DataFrame, guard: dict[str, object]
) -> dict[str, float | bool]:
    """Indica si una entrada queda dentro del rango de desarrollo conocido."""

    columns = [str(value) for value in guard["feature_columns"]]
    vectors = vector_por_grupo(features, columns)
    if len(vectors) != 1:
        raise ValueError("La inferencia debe contener un único grupo de entrada")
    center = pd.Series(guard["center"], dtype=float).reindex(columns)
    scale = pd.Series(guard["scale"], dtype=float).reindex(columns)
    if center.isna().any() or scale.isna().any() or (scale <= 0).any():
        raise ValueError("Control de dominio inválido")
    distance = float(_distance(vectors, center, scale).iloc[0])
    threshold = float(guard["threshold"])
    return {
        "compatible": bool(distance <= threshold),
        "distance": distance,
        "threshold": threshold,
    }


def calibrate_signal_quality_guard(
    development: pd.DataFrame,
    *,
    safety_factor: float = 1.05,
) -> dict[str, object]:
    """Fija un límite de transitorios usando solo las partes de desarrollo.

    El factor de cresta resume cuán grande es el pico respecto del RMS de una
    ventana. Un valor fuera del rango de desarrollo no se interpreta como fondo ni
    como dron: la entrada se marca como no concluyente antes de clasificar.
    """

    required = {"group_id", "part", "factor_cresta"}
    missing = required.difference(development.columns)
    if missing:
        raise ValueError(f"Faltan columnas para el control de calidad: {sorted(missing)}")
    if safety_factor < 1.0:
        raise ValueError("El factor de seguridad debe ser mayor o igual que uno")
    values = development.groupby(["group_id", "part"])["factor_cresta"].median()
    if values.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("No hay factores de cresta finitos para calibrar la calidad")
    maximum = float(values.max())
    limit = max(maximum * safety_factor, np.finfo(float).eps)
    return {
        "schema_version": "1.0",
        "purpose": "Rechazar entradas con transitorios fuera del rango de desarrollo.",
        "feature": "factor_cresta",
        "aggregation": "mediana_por_grupo_y_parte",
        "upper_limit": limit,
        "safety_factor": float(safety_factor),
        "development_groups": int(values.index.get_level_values("group_id").nunique()),
        "development_maximum": maximum,
    }


def check_signal_quality(
    features: pd.DataFrame, guard: dict[str, object]
) -> dict[str, float | bool | list[str]]:
    """Informa si una entrada mantiene transitorios compatibles con desarrollo."""

    required = {"group_id", "part", "factor_cresta"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Faltan columnas para el control de calidad: {sorted(missing)}")
    values = features.groupby("part")["factor_cresta"].median()
    if values.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("La entrada no tiene factores de cresta finitos")
    limit = float(guard["upper_limit"])
    if limit <= 0:
        raise ValueError("Control de calidad inválido")
    exceeding = values[values > limit]
    maximum = float(values.max())
    return {
        "compatible": bool(exceeding.empty),
        "maximum": maximum,
        "upper_limit": limit,
        "parts_exceeding": [str(part) for part in exceeding.index],
    }
