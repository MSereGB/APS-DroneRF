"""Configuracion central del proyecto."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RESULTS_DIR = PROJECT_ROOT / "resultados"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"
FEATURES_DIR = RESULTS_DIR / "features"

DEFAULT_RANDOM_STATE = 42
DEFAULT_SYNTHETIC_FS_HZ = 2_000.0

# Parametros usados por los scripts oficiales de DroneRF para archivos crudos L/H.
DRONERF_TIME_FS_HZ = 40_000_000.0
DRONERF_WINDOW_SIZE = 100_000


def ensure_project_dirs() -> None:
    """Crea las carpetas de salida livianas si no existen."""

    for path in [
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        METRICS_DIR,
        FEATURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
