"""Revisa el entorno local antes de preparar datos, calibrar o abrir la demo."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REQUIRED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "joblib",
    "gradio",
    "ipykernel",
    "reportlab",
    "python-pptx",
)


def package_versions() -> tuple[dict[str, str], list[str]]:
    """Devuelve versiones instaladas y dependencias ausentes."""

    installed: dict[str, str] = {}
    missing: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            missing.append(package)
    return installed, missing


def folder_has_files(path: Path, suffixes: set[str] | None = None) -> bool:
    """Indica si una carpeta contiene al menos un archivo de interés."""

    if not path.exists():
        return False
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        if suffixes is None or item.suffix.lower() in suffixes:
            return True
    return False


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", project_root / "data")),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=project_root / "resultados" / "runs" / "actual" / "modelos" / "bundle.joblib",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path(os.environ.get("MUESTRAS_DIR", project_root / "muestras_demo")),
    )
    parser.add_argument("--minimum-free-gb", type=float, default=18.0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Falla si todavía no están disponibles datos, bundle o muestras.",
    )
    args = parser.parse_args()

    versions, missing = package_versions()
    data_dir = args.data_dir.expanduser().resolve()
    bundle_path = args.bundle.expanduser().resolve()
    samples_dir = args.samples_dir.expanduser().resolve()
    free_gb = shutil.disk_usage(project_root.anchor).free / (1024**3)
    python_ok = sys.version_info[:3] == (3, 13, 4)
    data_ok = folder_has_files(data_dir, {".csv", ".txt", ".npy", ".npz", ".mat"})
    samples_ok = folder_has_files(samples_dir, {".npz"})
    bundle_ok = bundle_path.is_file()
    space_ok = free_gb >= args.minimum_free_gb

    print("Chequeo del entorno APS DroneRF")
    print(f"- Python: {sys.version.split()[0]} ({'OK' if python_ok else 'revisar'})")
    print(f"- Entorno virtual: {sys.prefix}")
    print(f"- Dependencias: {'OK' if not missing else 'faltan ' + ', '.join(missing)}")
    for package, package_version in sorted(versions.items()):
        print(f"  {package}: {package_version}")
    print(f"- DATA_DIR: {data_dir} ({'OK' if data_ok else 'sin señales'})")
    print(f"- Bundle: {bundle_path} ({'OK' if bundle_ok else 'pendiente'})")
    print(f"- Muestras: {samples_dir} ({'OK' if samples_ok else 'pendientes'})")
    print(f"- Espacio libre: {free_gb:.2f} GB ({'OK' if space_ok else 'insuficiente'})")

    base_failures = bool(missing) or not python_ok or not space_ok
    strict_failures = args.strict and (not data_ok or not bundle_ok or not samples_ok)
    if base_failures or strict_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
