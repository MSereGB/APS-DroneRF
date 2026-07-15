"""Pipeline incremental de features APS para manifiestos explícitos de DroneRF."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aps_drone_rf.features import build_feature_table, equal_frequency_bands
from aps_drone_rf.io import load_signal_file
from aps_drone_rf.preprocessing import preprocess_signal, sample_signal_windows

BASE_FEATURE_COLUMNS = [
    "rms",
    "potencia_media",
    "energia",
    "pico",
    "factor_cresta",
    "frecuencia_dominante_fft_hz",
    "frecuencia_dominante_hz",
    "psd_maxima",
    "potencia_total_psd",
    "centroide_espectral_hz",
    "ancho_banda_espectral_hz",
]


@dataclass(frozen=True)
class FeatureConfig:
    """Parámetros congelables de una extracción de características."""

    fs_hz: float = 40_000_000.0
    window_size: int = 512
    hop_size: int = 256
    windows_per_part: int = 100
    band_count: int = 20
    welch_nperseg: int = 256
    analysis_window: str = "hann"
    remove_mean: bool = True
    normalization: str = "maxabs"
    seed: int = 42

    def validate(self) -> None:
        if self.window_size <= 0 or self.hop_size <= 0 or self.windows_per_part <= 0:
            raise ValueError("Los parámetros de ventanas deben ser positivos")
        if self.band_count not in {4, 10, 20}:
            raise ValueError("band_count debe ser 4, 10 o 20")
        if self.welch_nperseg > self.window_size:
            raise ValueError("welch_nperseg no puede superar window_size")
        if self.analysis_window not in {"hann", "hamming", "blackman"}:
            raise ValueError("analysis_window debe ser hann, hamming o blackman")


def load_dataset_manifest(path: str | Path) -> dict[str, object]:
    """Carga un manifiesto listo y comprueba sus campos mínimos."""

    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if payload.get("status") != "local_subset_ready":
        raise ValueError("El manifiesto no corresponde a un subset local listo")
    rows = payload.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("El manifiesto no contiene archivos")
    required = {
        "relative_path",
        "sha256",
        "code",
        "activity",
        "model",
        "mode",
        "part",
        "segment",
        "group_id",
        "partition",
        "sample_rate_hz",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Faltan campos explícitos en el manifiesto: {sorted(missing)}")
    return payload


def validate_materialized_files(
    manifest: dict[str, object], dataset_root: str | Path, partitions: set[str]
) -> list[dict[str, object]]:
    """Resuelve rutas sin inferir etiquetas y falla ante cualquier archivo ausente."""

    root = Path(dataset_root).expanduser().resolve()
    selected = [
        dict(row)
        for row in manifest["files"]
        if str(row["partition"]) in partitions
    ]
    if not selected:
        raise ValueError(f"No hay archivos para las particiones {sorted(partitions)}")
    for row in selected:
        path = root / str(row["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_size = int(row.get("size_bytes", 0))
        if expected_size and path.stat().st_size != expected_size:
            raise ValueError(f"Tamaño inesperado para {row['relative_path']}")
        row["absolute_path"] = str(path)
    return selected


def extract_file_features(row: dict[str, object], config: FeatureConfig) -> pd.DataFrame:
    """Carga un archivo, toma ventanas y devuelve features con sus etiquetas explícitas."""

    config.validate()
    fs_hz = float(row["sample_rate_hz"])
    if not np.isclose(fs_hz, config.fs_hz):
        raise ValueError(f"Frecuencia de muestreo incompatible: {fs_hz:g} Hz")
    signal = load_signal_file(str(row["absolute_path"]))
    signal = preprocess_signal(
        signal,
        remove_mean=config.remove_mean,
        normalization=config.normalization,
    )
    return extract_preprocessed_features(signal, row, config)


def extract_preprocessed_features(
    signal: np.ndarray,
    row: dict[str, object],
    config: FeatureConfig,
) -> pd.DataFrame:
    """Extrae features de una señal ya preprocesada para reutilizar una lectura."""

    config.validate()
    fs_hz = float(row["sample_rate_hz"])
    if not np.isclose(fs_hz, config.fs_hz):
        raise ValueError(f"Frecuencia de muestreo incompatible: {fs_hz:g} Hz")
    windows, metadata = sample_signal_windows(
        signal,
        config.window_size,
        config.hop_size,
        config.windows_per_part,
        group_id=str(row["group_id"]),
        label=str(row["activity"]),
        part=str(row["part"]),
    )
    table = build_feature_table(
        windows,
        metadata,
        fs_hz=config.fs_hz,
        bands_hz=equal_frequency_bands(config.fs_hz, config.band_count),
        nperseg=config.welch_nperseg,
        analysis_window=config.analysis_window,
    )
    table["code"] = str(row["code"])
    table["activity"] = str(row["activity"])
    table["model"] = str(row["model"])
    table["mode"] = str(row["mode"])
    table["segment"] = int(row["segment"])
    table["partition"] = str(row["partition"])
    table["source_path"] = str(row["relative_path"])
    table["fs_hz"] = config.fs_hz
    return table


def relative_band_columns(count: int) -> list[str]:
    """Nombres estables de potencias relativas por bandas."""

    return [f"potencia_relativa_banda_{index}" for index in range(1, count + 1)]


def features_for_band_count(features: pd.DataFrame, target_count: int) -> pd.DataFrame:
    """Agrega las 20 bandas de 1 MHz en 10 o 4 bandas sin releer las señales."""

    if target_count not in {4, 10, 20}:
        raise ValueError("target_count debe ser 4, 10 o 20")
    source_columns = relative_band_columns(20)
    missing = [column for column in source_columns if column not in features]
    if missing:
        if target_count == 20:
            raise ValueError(f"Faltan bandas: {missing[:3]}")
        direct = relative_band_columns(target_count)
        if all(column in features for column in direct):
            return features.copy()
        raise ValueError("Se requieren las 20 bandas base para poder agregarlas")
    if target_count == 20:
        return features.copy()

    output = features.drop(columns=source_columns).copy()
    source = features[source_columns].to_numpy(dtype=float)
    width = 20 // target_count
    aggregated = source.reshape(len(features), target_count, width).sum(axis=2)
    for index, column in enumerate(relative_band_columns(target_count)):
        output[column] = aggregated[:, index]
    return output


def classifier_feature_columns(band_count: int) -> list[str]:
    """Lista interpretable usada por los clasificadores lineales."""

    return [*BASE_FEATURE_COLUMNS, *relative_band_columns(band_count)]


def config_dict(config: FeatureConfig) -> dict[str, object]:
    return asdict(config)
