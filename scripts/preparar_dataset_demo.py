"""Prepara el subconjunto multiclase de DroneRF sin descomprimir el dataset completo."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.dronerf import selection_rows, validate_selection
from aps_drone_rf.provenance import sha256_file, utc_now

SOURCE_SHA256 = "f3cd8a1dfe14f51edc40c8012f12cdf34c3d7e1c51b62ef4539505d9d3f32c0d"

ARCHIVES: dict[tuple[str, str, str], str] = {
    ("00000", "L", "all"): r"DroneRF\Background RF activites\RF Data_00000_L1.rar",
    ("00000", "H", "all"): r"DroneRF\Background RF activites\RF Data_00000_H1.rar",
    ("10000", "L", "all"): r"DroneRF\Bepop drone\RF Data_10000_L.rar",
    ("10000", "H", "all"): r"DroneRF\Bepop drone\RF Data_10000_H.rar",
    ("10001", "L", "all"): r"DroneRF\Bepop drone\RF Data_10001_L.rar",
    ("10001", "H", "all"): r"DroneRF\Bepop drone\RF Data_10001_H.rar",
    ("10010", "L", "all"): r"DroneRF\Bepop drone\RF Data_10010_L.rar",
    ("10010", "H", "all"): r"DroneRF\Bepop drone\RF Data_10010_H.rar",
    ("10011", "L", "all"): r"DroneRF\Bepop drone\RF Data_10011_L.rar",
    ("10011", "H", "all"): r"DroneRF\Bepop drone\RF Data_10011_H.rar",
    ("10100", "L", "all"): r"DroneRF\AR drone\RF Data_10100_L.rar",
    ("10100", "H", "all"): r"DroneRF\AR drone\RF Data_10100_H.rar",
    ("10101", "L", "all"): r"DroneRF\AR drone\RF Data_10101_L.rar",
    ("10101", "H", "all"): r"DroneRF\AR drone\RF Data_10101_H.rar",
    ("10110", "L", "all"): r"DroneRF\AR drone\RF Data_10110_L.rar",
    ("10110", "H", "all"): r"DroneRF\AR drone\RF Data_10110_H.rar",
    ("10111", "L", "all"): r"DroneRF\AR drone\RF Data_10111_L.rar",
    ("10111", "H", "all"): r"DroneRF\AR drone\RF Data_10111_H.rar",
    ("11000", "L", "first"): r"DroneRF\Phantom drone\RF Data_11000_L1.rar",
    ("11000", "L", "last"): r"DroneRF\Phantom drone\RF Data_11000_L2.rar",
    ("11000", "H", "all"): r"DroneRF\Phantom drone\RF Data_11000_H.rar",
}


def archive_for(row: dict[str, object]) -> str:
    """Ubica el RAR que contiene una fila seleccionada."""

    code = str(row["code"])
    part = str(row["part"])
    if code == "11000" and part == "L":
        section = "first" if int(row["segment"]) <= 9 else "last"
    else:
        section = "all"
    return ARCHIVES[(code, part, section)]


def resolve_seven_zip(value: str | None) -> str:
    """Encuentra 7-Zip o devuelve un error accionable."""

    if value:
        candidate = Path(value).expanduser().resolve()
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"No existe el ejecutable indicado: {candidate}")
    for name in ("7z", "7zz", "7z.exe"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError("No se encontró 7-Zip. Usar --seven-zip <ruta>.")


def run_7zip(seven_zip: str, arguments: list[str]) -> None:
    """Ejecuta 7-Zip y conserva su salida solo cuando ocurre un error."""

    completed = subprocess.run(
        [seven_zip, *arguments],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        message = completed.stdout[-2000:] + completed.stderr[-2000:]
        raise RuntimeError(f"7-Zip falló ({completed.returncode}):\n{message}")


def extract_archive_rows(
    source_zip: Path,
    dataset_root: Path,
    seven_zip: str,
    archive_member: str,
    rows: list[dict[str, object]],
) -> None:
    """Extrae un RAR temporal y solamente los CSV seleccionados dentro de él."""

    pending = []
    for row in rows:
        target = dataset_root / str(row["relative_path"])
        if not target.is_file() or target.stat().st_size == 0:
            pending.append(row)
    if not pending:
        return

    staging_parent = dataset_root / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_parent, prefix="rar_") as temporary:
        temp_root = Path(temporary)
        run_7zip(seven_zip, ["x", str(source_zip), archive_member, f"-o{temp_root}", "-y"])
        rar_name = Path(archive_member).name
        rar_files = list(temp_root.rglob(rar_name))
        if len(rar_files) != 1:
            raise RuntimeError(f"No se pudo ubicar {rar_name} dentro del ZIP")

        extracted = temp_root / "csv"
        extracted.mkdir()
        include_args = [f"-ir!{row['filename']}" for row in pending]
        run_7zip(seven_zip, ["e", str(rar_files[0]), f"-o{extracted}", *include_args, "-y"])

        for row in pending:
            filename = str(row["filename"])
            source_file = extracted / filename
            if not source_file.is_file():
                raise RuntimeError(f"El archivo seleccionado no apareció en el RAR: {filename}")
            target = dataset_root / str(row["relative_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_file), str(target))


def file_description(path: Path, row: dict[str, object]) -> dict[str, object]:
    """Agrega tamaño y hash a una fila materializada."""

    return {
        **row,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seven-zip")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--minimum-free-gb-after", type=float, default=3.0)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "dronerf_demo_v2_plan.json",
    )
    args = parser.parse_args()

    rows = selection_rows()
    validate_selection(rows)
    plan_payload: dict[str, object] = {
        "schema_version": "2.0",
        "status": "selection_frozen",
        "dataset": "DroneRF",
        "dataset_version": "f4c2b4n755/1",
        "source_url": "https://data.mendeley.com/datasets/f4c2b4n755/1",
        "official_label_reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6727013/",
        "selection_frozen_at_utc": utc_now(),
        "group_count": 79,
        "file_count": 158,
        "files": rows,
    }

    if not args.extract:
        write_manifest(args.manifest_output, plan_payload)
        print(f"Plan congelado en {args.manifest_output}")
        return
    if args.source_zip is None or args.output_root is None:
        parser.error("--extract requiere --source-zip y --output-root")

    source_zip = args.source_zip.expanduser().resolve()
    dataset_root = args.output_root.expanduser().resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    if args.verify_source and sha256_file(source_zip) != SOURCE_SHA256:
        raise ValueError("El hash del ZIP no coincide con el archivo DroneRF ya registrado")

    expected_bytes = 158 * 105_000_000
    existing_bytes = sum(
        (dataset_root / str(row["relative_path"])).stat().st_size
        for row in rows
        if (dataset_root / str(row["relative_path"])).is_file()
    )
    missing_estimate = max(expected_bytes - existing_bytes, 0)
    disk_reference = dataset_root.parent if dataset_root.parent.exists() else Path.cwd()
    free_bytes = shutil.disk_usage(disk_reference).free
    required_free = missing_estimate + int(args.minimum_free_gb_after * 1024**3)
    if free_bytes < required_free:
        raise OSError(
            f"Espacio insuficiente: libres {free_bytes / 1024**3:.2f} GB, "
            f"estimados {required_free / 1024**3:.2f} GB"
        )

    seven_zip = resolve_seven_zip(args.seven_zip)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[archive_for(row)].append(row)
    for index, (archive_member, archive_rows) in enumerate(grouped.items(), start=1):
        print(f"[{index}/{len(grouped)}] {Path(archive_member).name}")
        extract_archive_rows(source_zip, dataset_root, seven_zip, archive_member, archive_rows)

    materialized = []
    for row in rows:
        path = dataset_root / str(row["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        materialized.append(file_description(path, row))

    manifest = {
        **plan_payload,
        "status": "local_subset_ready",
        "registered_at_utc": utc_now(),
        "source_archive": {
            "name": source_zip.name,
            "size_bytes": int(source_zip.stat().st_size),
            "sha256": SOURCE_SHA256,
        },
        "files": materialized,
    }
    write_manifest(args.manifest_output, manifest)
    external_manifest = dataset_root / "manifest.json"
    write_manifest(external_manifest, manifest)
    print(f"Dataset listo en {dataset_root}")
    print(f"Manifiesto: {args.manifest_output}")


if __name__ == "__main__":
    main()
