"""Helpers livianos para registrar procedencia de datos y corridas."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula SHA-256 byte a byte para un archivo local."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: str | Path, relative_to: str | Path | None = None) -> dict[str, object]:
    """Devuelve nombre relativo, tamaño y hash de un archivo existente."""

    file_path = Path(path).resolve()
    if relative_to is None:
        visible_path = file_path.name
    else:
        root = Path(relative_to).resolve()
        try:
            visible_path = file_path.relative_to(root).as_posix()
        except ValueError:
            visible_path = file_path.name
    return {
        "path": visible_path,
        "size_bytes": int(file_path.stat().st_size),
        "sha256": sha256_file(file_path),
    }


def git_commit(project_root: str | Path) -> str | None:
    """Obtiene el commit actual sin fallar fuera de un checkout Git."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def git_worktree_status(project_root: str | Path) -> list[str]:
    """Devuelve cambios locales resumidos para no ocultarlos en un manifest."""

    root = Path(project_root).resolve()
    try:
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(top_level).resolve() != root:
            return []
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


def runtime_versions(
    packages: tuple[str, ...] = ("numpy", "scipy", "pandas", "scikit-learn"),
) -> dict[str, str]:
    """Registra versiones de Python y librerías usadas en una corrida."""

    values = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "no_instalado"
    return values


def utc_now() -> str:
    """Marca temporal ISO-8601 en UTC, sin microsegundos."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def write_json(path: str | Path, payload: dict[str, object]) -> Path:
    """Escribe un JSON UTF-8 con indentación estable."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output
