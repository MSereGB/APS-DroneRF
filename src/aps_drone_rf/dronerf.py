"""Catálogo y particiones congeladas para el subconjunto multiclase de DroneRF."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

DRONERF_FS_HZ = 40_000_000.0
DRONERF_PARTS = ("L", "H")


@dataclass(frozen=True)
class DroneRFClass:
    """Etiqueta explícita de un código publicado por DroneRF."""

    code: str
    activity: str
    model: str
    mode: str
    max_segment: int


DRONERF_CLASSES = {
    item.code: item
    for item in (
        DroneRFClass("00000", "fondo", "sin_dron", "sin_dron", 20),
        DroneRFClass("10000", "dron", "bebop", "conectado", 20),
        DroneRFClass("10001", "dron", "bebop", "hovering", 20),
        DroneRFClass("10010", "dron", "bebop", "vuelo", 20),
        DroneRFClass("10011", "dron", "bebop", "video", 20),
        DroneRFClass("10100", "dron", "ar", "conectado", 20),
        DroneRFClass("10101", "dron", "ar", "hovering", 20),
        DroneRFClass("10110", "dron", "ar", "vuelo", 20),
        DroneRFClass("10111", "dron", "ar", "video", 17),
        DroneRFClass("11000", "dron", "phantom", "conectado", 20),
    )
}

DISPLAY_NAMES = {
    "fondo": "Sin dron",
    "dron": "Actividad de dron",
    "sin_dron": "Sin dron",
    "bebop": "Bebop",
    "ar": "AR",
    "phantom": "Phantom",
    "conectado": "Conectado",
    "hovering": "Hovering",
    "vuelo": "Vuelo",
    "video": "Video",
}


def split_segments(code: str) -> dict[str, tuple[int, ...]]:
    """Devuelve desarrollo, evaluación y demo ciega para un código."""

    if code not in DRONERF_CLASSES:
        raise ValueError(f"Código DroneRF no reconocido: {code}")
    if code == "00000":
        return {
            "desarrollo": tuple(range(10)),
            "evaluacion": (15, 16, 17),
            "demo": (18, 19, 20),
        }
    if code == "10111":
        return {"desarrollo": tuple(range(5)), "evaluacion": (16,), "demo": (17,)}
    return {"desarrollo": tuple(range(5)), "evaluacion": (19,), "demo": (20,)}


def group_id(code: str, segment: int) -> str:
    """Construye el identificador común de las partes L/H."""

    return f"dronerf_{code}_{int(segment)}"


def selection_rows(raw_root: str | Path = "raw") -> list[dict[str, object]]:
    """Construye las 158 filas congeladas, una por archivo L/H seleccionado."""

    root = Path(raw_root)
    rows: list[dict[str, object]] = []
    for code, class_info in DRONERF_CLASSES.items():
        for partition, segments in split_segments(code).items():
            for segment in segments:
                if segment > class_info.max_segment:
                    raise ValueError(f"Segmento {segment} fuera de rango para {code}")
                for part in DRONERF_PARTS:
                    filename = f"{code}{part}_{segment}.csv"
                    rows.append(
                        {
                            **asdict(class_info),
                            "part": part,
                            "segment": segment,
                            "group_id": group_id(code, segment),
                            "partition": partition,
                            "filename": filename,
                            "relative_path": (root / code / part / filename).as_posix(),
                            "sample_rate_hz": DRONERF_FS_HZ,
                        }
                    )
    return rows


def validate_selection(rows: list[dict[str, object]]) -> None:
    """Falla si la selección deja de cumplir sus invariantes académicas."""

    if len(rows) != 158:
        raise ValueError(f"Se esperaban 158 archivos y se obtuvieron {len(rows)}")
    unique_groups = {str(row["group_id"]) for row in rows}
    if len(unique_groups) != 79:
        raise ValueError(f"Se esperaban 79 grupos y se obtuvieron {len(unique_groups)}")

    partitions_by_group: dict[str, set[str]] = {}
    parts_by_group: dict[str, set[str]] = {}
    for row in rows:
        current_group = str(row["group_id"])
        partitions_by_group.setdefault(current_group, set()).add(str(row["partition"]))
        parts_by_group.setdefault(current_group, set()).add(str(row["part"]))
    mixed = [key for key, value in partitions_by_group.items() if len(value) != 1]
    if mixed:
        raise ValueError(f"Grupos presentes en más de una partición: {mixed[:5]}")
    incomplete = [key for key, value in parts_by_group.items() if value != {"L", "H"}]
    if incomplete:
        raise ValueError(f"Grupos sin el par L/H completo: {incomplete[:5]}")


def code_metadata(code: str) -> dict[str, str]:
    """Devuelve etiquetas explícitas para un código DroneRF."""

    try:
        info = DRONERF_CLASSES[code]
    except KeyError as exc:
        raise ValueError(f"Código DroneRF no reconocido: {code}") from exc
    return {
        "activity": info.activity,
        "model": info.model,
        "mode": info.mode,
    }
