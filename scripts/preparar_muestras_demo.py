"""Crea paquetes livianos desde los grupos ciegos, con etiquetas en catálogo separado."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.demo import IGNORED_LABEL_KEYS
from aps_drone_rf.io import load_signal_file
from aps_drone_rf.pipeline import load_dataset_manifest, validate_materialized_files
from aps_drone_rf.preprocessing import preprocess_signal, sample_signal_windows
from aps_drone_rf.provenance import sha256_file, utc_now, write_json


def destination_folder(row: dict[str, object]) -> Path:
    if row["activity"] == "fondo":
        return Path("sin_dron")
    return Path(str(row["model"])) / str(row["mode"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "dronerf_demo_v2_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "muestras_demo")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--windows-per-part", type=int, default=100)
    args = parser.parse_args()

    manifest = load_dataset_manifest(args.manifest)
    rows = validate_materialized_files(manifest, args.data_dir, {"demo"})
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(row)
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = defaultdict(int)
    catalog_entries = []
    for group_id, group_rows in sorted(groups.items()):
        if {str(row["part"]) for row in group_rows} != {"L", "H"}:
            raise ValueError(f"El grupo ciego {group_id} no tiene L/H")
        first = group_rows[0]
        folder = destination_folder(first)
        folder_key = folder.as_posix()
        counters[folder_key] += 1
        sample_id = f"muestra_{counters[folder_key]:02d}"
        target_dir = output_root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{sample_id}.npz"

        window_sets = {}
        starts = {}
        source_files = []
        for row in group_rows:
            raw_signal = load_signal_file(row["absolute_path"])
            processed = preprocess_signal(raw_signal)
            windows, metadata = sample_signal_windows(
                processed,
                args.window_size,
                args.hop_size,
                args.windows_per_part,
                group_id=str(row["group_id"]),
                label="oculta",
                part=str(row["part"]),
            )
            window_sets[str(row["part"])] = windows.astype(np.float32)
            starts[str(row["part"])] = metadata["start"].astype(int).tolist()
            source_files.append(
                {
                    "relative_path": row["relative_path"],
                    "part": row["part"],
                    "sha256": row["sha256"],
                }
            )
        np.savez_compressed(
            target,
            ventanas_l=window_sets["L"],
            ventanas_h=window_sets["H"],
            fs_hz=np.array([float(first["sample_rate_hz"])]),
            sample_id=np.array([sample_id]),
            preprocessed=np.array([1], dtype=np.uint8),
            window_size=np.array([args.window_size]),
            selection_protocol=np.array(["distributed_windows_v2"]),
        )
        with np.load(target, allow_pickle=False) as archive:
            forbidden = IGNORED_LABEL_KEYS.intersection(key.lower() for key in archive.files)
            if forbidden:
                raise ValueError(f"El paquete contiene etiquetas: {sorted(forbidden)}")
        catalog_entries.append(
            {
                "sample_id": sample_id,
                "relative_path": target.relative_to(output_root).as_posix(),
                "package_sha256": sha256_file(target),
                "expected": {
                    "activity": first["activity"],
                    "model": first["model"],
                    "mode": first["mode"],
                    "code": first["code"],
                },
                "source_group_id": group_id,
                "source_partition": "demo",
                "source_files": source_files,
                "window_starts": starts,
                "window_size": args.window_size,
                "windows_per_part": args.windows_per_part,
                "selection_protocol": "distributed_windows_v2",
            }
        )

    catalog = {
        "schema_version": "1.0",
        "created_at_utc": utc_now(),
        "purpose": "Etiquetas de verificación separadas de la función de inferencia.",
        "sample_count": len(catalog_entries),
        "samples": catalog_entries,
    }
    catalog_path = output_root / "catalogo_privado.json"
    write_json(catalog_path, catalog)
    print(f"Muestras: {len(catalog_entries)} en {output_root}")
    print(f"Catálogo separado: {catalog_path}")


if __name__ == "__main__":
    main()
