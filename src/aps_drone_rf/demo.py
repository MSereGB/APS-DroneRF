"""Carga de muestras e inferencia trazable para la demostración local."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aps_drone_rf.dominio import check_domain, check_signal_quality
from aps_drone_rf.dronerf import DISPLAY_NAMES
from aps_drone_rf.features import build_feature_table, equal_frequency_bands
from aps_drone_rf.io import load_signal_file
from aps_drone_rf.pipeline import classifier_feature_columns
from aps_drone_rf.preprocessing import preprocess_signal, sample_signal_windows, segment_signal

IGNORED_LABEL_KEYS = {
    "label",
    "activity",
    "model",
    "mode",
    "expected",
    "etiqueta",
    "modelo",
    "modo",
}


@dataclass(frozen=True)
class SignalInput:
    """Señales disponibles para inferencia, sin etiqueta conocida."""

    signals: dict[str, np.ndarray]
    fs_hz: float
    source_name: str
    window_sets: dict[str, np.ndarray] | None = None


def _npz_signals(archive) -> dict[str, np.ndarray]:
    preferred = {"L": "senal_l", "H": "senal_h"}
    signals = {
        part: np.asarray(archive[key]).reshape(-1)
        for part, key in preferred.items()
        if key in archive.files
    }
    if signals:
        return signals
    candidates = []
    for key in archive.files:
        if key.lower() in IGNORED_LABEL_KEYS or key.lower() in {"fs", "fs_hz", "sample_id"}:
            continue
        value = np.asarray(archive[key])
        if np.issubdtype(value.dtype, np.number) and value.size > 1:
            candidates.append((key, value.reshape(-1)))
    if not candidates:
        raise ValueError("El NPZ no contiene una señal numérica utilizable")
    return {"UNICA": max(candidates, key=lambda item: item[1].size)[1]}


def _npz_window_sets(archive) -> dict[str, np.ndarray] | None:
    keys = {"L": "ventanas_l", "H": "ventanas_h"}
    available = {
        part: np.asarray(archive[key])
        for part, key in keys.items()
        if key in archive.files
    }
    if not available:
        return None
    if set(available) != {"L", "H"}:
        raise ValueError("El paquete de ventanas debe contener L y H")
    if "preprocessed" not in archive.files or int(np.asarray(archive["preprocessed"]).item()) != 1:
        raise ValueError("El paquete no declara ventanas preprocesadas")
    for part, windows in available.items():
        if windows.ndim != 2 or windows.shape[0] == 0 or windows.shape[1] == 0:
            raise ValueError(f"Forma de ventanas inválida para {part}")
    return available


def load_signal_input(path: str | Path, fs_hz: float | None = None) -> SignalInput:
    """Carga CSV, TXT, NPY, NPZ o MAT y descarta cualquier etiqueta incluida."""

    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    if file_path.suffix.lower() == ".npz":
        with np.load(file_path, allow_pickle=False) as archive:
            window_sets = _npz_window_sets(archive)
            if window_sets is None:
                signals = _npz_signals(archive)
            else:
                signals = {
                    part: windows.reshape(-1) for part, windows in window_sets.items()
                }
            embedded_fs = None
            for key in ("fs_hz", "fs"):
                if key in archive.files:
                    embedded_fs = float(np.asarray(archive[key]).reshape(-1)[0])
                    break
        final_fs = fs_hz if fs_hz is not None else embedded_fs
    else:
        signals = {"UNICA": load_signal_file(file_path)}
        window_sets = None
        final_fs = fs_hz
    if final_fs is None or final_fs <= 0:
        raise ValueError("Hay que indicar una frecuencia de muestreo válida")
    for part, signal in signals.items():
        if len(signal) < 16:
            raise ValueError(f"La parte {part} es demasiado corta")
        if not np.all(np.isfinite(signal)):
            raise ValueError(f"La parte {part} contiene NaN o infinitos")
    return SignalInput(
        signals=signals,
        fs_hz=float(final_fs),
        source_name=file_path.name,
        window_sets=window_sets,
    )


def load_bundle(path: str | Path) -> dict[str, object]:
    """Carga un bundle y rechaza esquemas incompletos o incompatibles."""

    bundle_path = Path(path).expanduser().resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"No se encontró el bundle calibrado: {bundle_path}")
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != "1.1":
        version = bundle.get("schema_version") if isinstance(bundle, dict) else None
        raise ValueError(f"Versión de bundle no soportada: {version}")
    required = {
        "schema_version",
        "band_count",
        "feature_columns",
        "models",
        "thresholds",
        "stage_reliability",
        "domain_guard",
        "signal_quality_guard",
    }
    if not required.issubset(bundle):
        raise ValueError("Bundle incompleto")
    if set(bundle["models"]) != {"actividad", "modelo", "modo_bebop", "modo_ar"}:
        raise ValueError("El bundle no contiene toda la jerarquía esperada")
    expected_features = classifier_feature_columns(int(bundle["band_count"]))
    if list(bundle["feature_columns"]) != expected_features:
        raise ValueError("El esquema de features del bundle no coincide")
    return bundle


def extract_input_features(
    signal_input: SignalInput,
    bundle: dict[str, object],
    *,
    windows_per_part: int = 100,
) -> pd.DataFrame:
    """Extrae la misma representación APS usada durante la calibración."""

    feature_config = bundle.get("feature_config", {}).get("feature_config", {})
    window_size = int(feature_config.get("window_size", 512))
    hop_size = int(feature_config.get("hop_size", window_size // 2))
    nperseg = int(feature_config.get("welch_nperseg", min(256, window_size)))
    analysis_window = str(feature_config.get("analysis_window", "hann"))
    rows = []
    for part, raw_signal in signal_input.signals.items():
        if signal_input.window_sets is not None:
            windows = np.asarray(signal_input.window_sets[part])[:windows_per_part]
            if windows.shape[1] != window_size:
                raise ValueError("El tamaño de ventana del paquete no coincide con el bundle")
            metadata = pd.DataFrame(
                {
                    "window_id": [
                        f"entrada_{part.lower()}:{index}" for index in range(len(windows))
                    ],
                    "group_id": "entrada_demo",
                    "label": "desconocido",
                    "part": part,
                    "start": np.arange(len(windows)) * hop_size,
                    "stop": np.arange(len(windows)) * hop_size + window_size,
                }
            )
        else:
            processed = preprocess_signal(raw_signal)
            windows, metadata = sample_signal_windows(
                processed,
                window_size,
                hop_size,
                windows_per_part,
                group_id="entrada_demo",
                label="desconocido",
                part=part,
            )
        rows.append(
            build_feature_table(
                windows,
                metadata,
                fs_hz=signal_input.fs_hz,
                bands_hz=equal_frequency_bands(signal_input.fs_hz, int(bundle["band_count"])),
                nperseg=nperseg,
                analysis_window=analysis_window,
            )
        )
    return pd.concat(rows, ignore_index=True)


def _model_scores(model, features: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    probabilities = model.predict_proba(features[feature_columns].to_numpy(dtype=float))
    score_frame = pd.DataFrame(probabilities, columns=[str(value) for value in model.classes_])
    score_frame["part"] = features["part"].to_numpy()
    by_part = score_frame.groupby("part").median(numeric_only=True)
    return {column: float(value) for column, value in by_part.mean(axis=0).items()}


def _ordered_hypotheses(scores: dict[str, float]) -> list[dict[str, object]]:
    return [
        {"name": DISPLAY_NAMES.get(name, name), "key": name, "score": float(score)}
        for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def _top_with_margin(scores: dict[str, float]) -> tuple[str, float]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ordered:
        raise ValueError("El modelo no devolvió puntajes")
    margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1]
    return ordered[0][0], float(margin)


def predict_hierarchy(bundle: dict[str, object], features: pd.DataFrame) -> dict[str, object]:
    """Aplica la jerarquía sin recibir ni consultar una etiqueta esperada."""

    feature_columns = list(bundle["feature_columns"])
    missing = [column for column in feature_columns if column not in features]
    if missing:
        raise ValueError(f"Faltan features requeridas: {missing[:3]}")
    models = bundle["models"]
    thresholds = bundle["thresholds"]

    activity_scores = _model_scores(models["actividad"], features, feature_columns)
    drone_score = float(activity_scores.get("dron", 0.0))
    decision_threshold = float(thresholds["actividad"]["decision"])
    activity_margin = abs(drone_score - decision_threshold)
    activity_rejection = float(thresholds["actividad"]["rejection_margin"])
    result: dict[str, object] = {
        "state": "no_concluyente",
        "model": None,
        "mode": None,
        "drone_score": drone_score,
        "threshold": decision_threshold,
        "margin": activity_margin,
        "activity_hypotheses": _ordered_hypotheses(activity_scores),
        "model_hypotheses": [],
        "mode_hypotheses": [],
        "last_reliable_level": "ninguno",
    }
    required_parts = set(
        bundle.get("deployment_policy", {}).get("required_parts_for_activity", ["L", "H"])
    )
    available_parts = set(features["part"].astype(str).str.upper())
    if not required_parts.issubset(available_parts):
        result["stopped_reason"] = (
            "La calibración final combina L (banda baja) y H (banda alta). Con una sola "
            "parte o banda se muestran las curvas, pero la decisión de actividad queda "
            "como no concluyente."
        )
        return result
    domain = check_domain(features, bundle["domain_guard"])
    result.update(
        {
            "domain_compatible": domain["compatible"],
            "domain_distance": domain["distance"],
            "domain_threshold": domain["threshold"],
        }
    )
    if not bool(domain["compatible"]):
        result["stopped_reason"] = (
            "La señal queda fuera del rango de características usado para calibrar con "
            "DroneRF. Se muestran los gráficos, pero la decisión queda como no concluyente."
        )
        return result
    quality = check_signal_quality(features, bundle["signal_quality_guard"])
    result.update(
        {
            "quality_compatible": quality["compatible"],
            "crest_maximum": quality["maximum"],
            "crest_upper_limit": quality["upper_limit"],
            "crest_parts_exceeding": quality["parts_exceeding"],
        }
    )
    if not bool(quality["compatible"]):
        result["stopped_reason"] = (
            "La entrada presenta transitorios con un factor de cresta fuera del rango "
            "calibrado. Se muestran los gráficos, pero la decisión queda como no concluyente."
        )
        return result
    if activity_margin < activity_rejection:
        return result
    if drone_score < decision_threshold:
        result.update({"state": "fondo", "last_reliable_level": "actividad"})
        return result

    result.update({"state": "dron", "last_reliable_level": "actividad"})
    reliability = bundle["stage_reliability"]
    model_scores = _model_scores(models["modelo"], features, feature_columns)
    model_name, model_margin = _top_with_margin(model_scores)
    result["model_hypotheses"] = _ordered_hypotheses(model_scores)
    result["model_margin"] = model_margin
    if not reliability["modelo"]["enabled"]:
        result["stopped_reason"] = (
            "La actividad sí está respaldada, pero la identificación de modelo no pasó "
            "la verificación reservada. La aplicación no fuerza un modelo."
        )
        return result
    if model_margin < float(thresholds["modelo"]["rejection_margin"]):
        result["stopped_reason"] = "Los puntajes de modelo quedaron demasiado próximos entre sí."
        return result
    result.update({"model": model_name, "last_reliable_level": "modelo"})

    if model_name == "phantom":
        result.update({"mode": "conectado", "last_reliable_level": "modo"})
        return result
    mode_stage = f"modo_{model_name}"
    if not reliability[mode_stage]["enabled"]:
        result["stopped_reason"] = (
            "La identificación de modo no tiene respaldo suficiente en la evaluación "
            "reservada y se deja como no concluyente."
        )
        return result
    mode_scores = _model_scores(models[mode_stage], features, feature_columns)
    mode_name, mode_margin = _top_with_margin(mode_scores)
    result["mode_hypotheses"] = _ordered_hypotheses(mode_scores)
    result["mode_margin"] = mode_margin
    if mode_margin < float(thresholds[mode_stage]["rejection_margin"]):
        return result
    result.update({"mode": mode_name, "last_reliable_level": "modo"})
    return result


def progressive_feature_batches(
    signal_input: SignalInput,
    bundle: dict[str, object],
    *,
    batch_size: int = 10,
    max_windows_per_part: int = 200,
) -> Iterator[pd.DataFrame]:
    """Entrega features en orden para simular bloques sucesivos de adquisición."""

    feature_config = bundle.get("feature_config", {}).get("feature_config", {})
    window_size = int(feature_config.get("window_size", 512))
    hop_size = int(feature_config.get("hop_size", window_size // 2))
    nperseg = int(feature_config.get("welch_nperseg", min(256, window_size)))
    analysis_window = str(feature_config.get("analysis_window", "hann"))
    bands = equal_frequency_bands(signal_input.fs_hz, int(bundle["band_count"]))
    part_tables = []
    for part, raw_signal in signal_input.signals.items():
        if signal_input.window_sets is not None:
            windows = np.asarray(signal_input.window_sets[part])[:max_windows_per_part]
            metadata = pd.DataFrame(
                {
                    "window_id": [
                        f"entrada_{part.lower()}:{index}" for index in range(len(windows))
                    ],
                    "group_id": "entrada_demo",
                    "label": "desconocido",
                    "part": part,
                    "start": np.arange(len(windows)) * hop_size,
                    "stop": np.arange(len(windows)) * hop_size + window_size,
                }
            )
        else:
            processed = preprocess_signal(raw_signal)
            required_samples = window_size + max(0, max_windows_per_part - 1) * hop_size
            progressive_signal = processed[:required_samples]
            windows, metadata = segment_signal(
                progressive_signal,
                window_size,
                hop_size,
                group_id="entrada_demo",
                label="desconocido",
                window_id_prefix=f"entrada_{part.lower()}",
            )
            windows = windows[:max_windows_per_part]
            metadata = metadata.iloc[: len(windows)].copy()
            metadata["part"] = part
        part_tables.append(
            build_feature_table(
                windows,
                metadata,
                fs_hz=signal_input.fs_hz,
                bands_hz=bands,
                nperseg=nperseg,
                analysis_window=analysis_window,
            )
        )
    maximum = max(len(table) for table in part_tables)
    for stop in range(batch_size, maximum + batch_size, batch_size):
        current = [table.iloc[:stop] for table in part_tables if len(table)]
        yield pd.concat(current, ignore_index=True)
