"""Entrada/salida de datos RF locales, con stubs seguros para DroneRF."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from aps_drone_rf.estilo import etiqueta_clase

SUPPORTED_EXTENSIONS = {".npy", ".npz", ".mat", ".csv", ".txt"}
DRONERF_FILENAME_RE = re.compile(r"^(?P<code>\d{5})(?P<part>[LH])_(?P<segment>\d+)$", re.IGNORECASE)
GENERATED_DATA_DIRS = {"interim", "processed"}
GENERATED_DATA_FILES = {"signal_manifest.csv"}


@dataclass
class SignalRecord:
    """Registro cargado desde archivo local."""

    signal: np.ndarray
    fs_hz: float
    label: str
    group_id: str
    source_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


def infer_label_from_path(path: Path) -> str:
    """Inferencia conservadora de etiqueta a partir del nombre de archivo/carpeta."""

    dronerf_info = parse_dronerf_filename(path)
    if dronerf_info is not None:
        return "fondo" if dronerf_info["code"] == "00000" else "dron"

    file_text = path.name.lower()
    parent_text = path.parent.name.lower()
    background_terms = ["background", "no drone", "no_drone", "nodrone", "sin_dron", "fondo"]
    if any(term in file_text or term in parent_text for term in background_terms):
        return "fondo"
    model_terms = ["bebop", "ardrone", "ar_drone", "ar-drone", "phantom", "mambo"]
    if "drone" in file_text or any(
        term in file_text or term in parent_text for term in model_terms
    ):
        return "dron"
    return "desconocido"


def parse_dronerf_filename(path: str | Path) -> dict[str, str] | None:
    """Parsea nombres DroneRF como `11000L_3` o `11000H_3`."""

    file_path = Path(path)
    candidate = file_path.name if file_path.suffix == "" else file_path.stem
    match = DRONERF_FILENAME_RE.match(candidate)
    if match is None:
        return None
    code = match.group("code")
    segment = match.group("segment")
    return {
        "code": code,
        "part": match.group("part").upper(),
        "segment": segment,
        "segment_id": f"{code}_{segment}",
    }


def is_supported_signal_file(path: Path) -> bool:
    """Identifica archivos de señal soportados, incluyendo partes DroneRF sin extension."""

    return path.suffix.lower() in SUPPORTED_EXTENSIONS or parse_dronerf_filename(path) is not None


def list_signal_files(data_dir: str | Path) -> list[Path]:
    """Lista archivos de señal soportados bajo `data_dir`."""

    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        return []
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or not is_supported_signal_file(path):
            continue
        relative = path.relative_to(root)
        relative_parts = {part.lower() for part in relative.parts[:-1]}
        if relative_parts.intersection(GENERATED_DATA_DIRS):
            continue
        if path.name.lower() in GENERATED_DATA_FILES:
            continue
        files.append(path)
    return sorted(files)


def discover_signal_manifest(data_dir: str | Path) -> pd.DataFrame:
    """Construye un manifiesto liviano de archivos locales."""

    rows = []
    root = Path(data_dir).expanduser().resolve()
    for path in list_signal_files(data_dir):
        dronerf_info = parse_dronerf_filename(path)
        if dronerf_info is None:
            relative_id = path.relative_to(root).with_suffix("").as_posix()
            group_id = relative_id
            extra = {"dronerf_code": "", "dronerf_part": "", "dronerf_segment": ""}
        else:
            group_id = f"dronerf_{dronerf_info['segment_id']}"
            extra = {
                "dronerf_code": dronerf_info["code"],
                "dronerf_part": dronerf_info["part"],
                "dronerf_segment": dronerf_info["segment"],
            }
        row = {
            "path": str(path),
            "label": etiqueta_clase(infer_label_from_path(path)),
            "group_id": group_id,
            "suffix": path.suffix.lower(),
            "size_bytes": int(path.stat().st_size),
        }
        row.update(extra)
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "path",
            "label",
            "group_id",
            "suffix",
            "size_bytes",
            "dronerf_code",
            "dronerf_part",
            "dronerf_segment",
        ],
    )


def _largest_numeric_mat_array(mat: dict[str, Any]) -> np.ndarray:
    candidates = [
        value
        for key, value in mat.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
    ]
    if not candidates:
        raise ValueError("No se encontro ningun array numerico en el archivo .mat")
    return max(candidates, key=lambda arr: arr.size)


def _load_text_array(file_path: Path, delimiter: str | None) -> np.ndarray:
    attempts = [
        {"delimiter": delimiter, "skiprows": 0},
        {"delimiter": None, "skiprows": 0},
        {"delimiter": delimiter, "skiprows": 1},
        {"delimiter": None, "skiprows": 1},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return np.loadtxt(file_path, **kwargs)
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"No se pudo cargar archivo numerico de texto: {file_path}") from last_error


def load_dronerf_integer_csv(path: str | Path) -> np.ndarray:
    """Lee rápidamente el CSV entero y de una sola línea publicado por DroneRF.

    Los archivos oficiales guardan valores como ``-12.000000`` separados por comas.
    Se valida ese formato antes de convertirlo; un archivo distinto debe usar el lector
    genérico y no entrar silenciosamente por esta ruta.
    """

    file_path = Path(path).expanduser().resolve()
    raw_bytes = file_path.read_bytes().rstrip()
    if not raw_bytes:
        raise ValueError(f"Archivo DroneRF vacío: {file_path}")
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    commas = np.flatnonzero(raw == ord(","))
    if raw[-1] == ord(","):
        ends = commas
    else:
        ends = np.concatenate([commas, np.array([len(raw)], dtype=commas.dtype)])
    if len(ends) == 0:
        raise ValueError(f"CSV DroneRF sin separadores: {file_path}")

    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    dot_positions = ends - 7
    if np.any(dot_positions < starts):
        raise ValueError(f"Formato numérico inesperado en {file_path}")
    if not np.all(raw[dot_positions] == ord(".")):
        raise ValueError(f"DroneRF no usa el formato decimal esperado en {file_path}")
    for offset in range(1, 7):
        if not np.all(raw[dot_positions + offset] == ord("0")):
            raise ValueError(f"DroneRF contiene decimales no enteros en {file_path}")

    negative = raw[starts] == ord("-")
    digit_counts = dot_positions - starts - negative.astype(np.int64)
    if np.any((digit_counts < 1) | (digit_counts > 6)):
        raise ValueError(f"Cantidad de dígitos no soportada en {file_path}")

    values = np.zeros(len(ends), dtype=np.int32)
    for offset in range(1, int(digit_counts.max()) + 1):
        mask = digit_counts >= offset
        digits = raw[dot_positions[mask] - offset].astype(np.int32) - ord("0")
        if np.any((digits < 0) | (digits > 9)):
            raise ValueError(f"Carácter no numérico en {file_path}")
        values[mask] += digits * (10 ** (offset - 1))
    values[negative] *= -1
    if values.min() >= np.iinfo(np.int16).min and values.max() <= np.iinfo(np.int16).max:
        return values.astype(np.int16)
    return values


def load_signal_file(path: str | Path, variable: str | None = None) -> np.ndarray:
    """Carga un archivo de señal soportado y devuelve un array 1D."""

    file_path = Path(path).expanduser().resolve()
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(file_path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(file_path, allow_pickle=False)
        key = variable or archive.files[0]
        data = archive[key]
    elif suffix == ".mat":
        mat = loadmat(file_path)
        data = mat[variable] if variable else _largest_numeric_mat_array(mat)
    elif suffix in {".csv", ".txt"} or parse_dronerf_filename(file_path) is not None:
        is_dronerf_text = parse_dronerf_filename(file_path) is not None
        if is_dronerf_text:
            try:
                data = load_dronerf_integer_csv(file_path)
            except ValueError:
                data = _load_text_array(file_path, delimiter=",")
        else:
            delimiter = "," if suffix == ".csv" else None
            data = _load_text_array(file_path, delimiter=delimiter)
    else:
        raise ValueError(f"Extension no soportada: {suffix}")

    array = np.asarray(data)
    if array.ndim == 2 and 2 in array.shape:
        reshaped = array if array.shape[1] == 2 else array.T
        if reshaped.shape[1] == 2:
            array = reshaped[:, 0] + 1j * reshaped[:, 1]
    return array.reshape(-1)


def load_record(row: pd.Series, fs_hz: float, normalize: bool = False) -> SignalRecord:
    """Carga una fila de manifiesto como `SignalRecord`."""

    from aps_drone_rf.preprocessing import normalize_signal

    signal = load_signal_file(row["path"])
    if normalize:
        signal = normalize_signal(signal)
    return SignalRecord(
        signal=signal,
        fs_hz=fs_hz,
        label=str(row.get("label", "unknown")),
        group_id=str(row.get("group_id", Path(row["path"]).stem)),
        source_path=Path(row["path"]),
        metadata=row.to_dict(),
    )
