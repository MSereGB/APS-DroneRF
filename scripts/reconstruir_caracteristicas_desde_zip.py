"""Reconstruye features DroneRF por RAR sin materializar el subconjunto completo."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preparar_dataset_demo import archive_for, resolve_seven_zip, run_7zip

from aps_drone_rf.pipeline import FeatureConfig, config_dict, extract_file_features
from aps_drone_rf.provenance import git_commit, sha256_file, utc_now, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "dronerf_demo_v2_manifest.json",
    )
    parser.add_argument(
        "--run-id",
        default="dronerf_demo_v2_final_n1024_hann_b20_seed42",
    )
    parser.add_argument("--seven-zip")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--windows-per-part", type=int, default=100)
    parser.add_argument("--band-count", type=int, default=20)
    parser.add_argument("--welch-nperseg", type=int, default=256)
    parser.add_argument("--analysis-window", default="hann")
    parser.add_argument("--verify-hashes", action="store_true")
    return parser.parse_args()


def selected_rows(manifest_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rows = [
        dict(row)
        for row in manifest["files"]
        if str(row["partition"]) in {"desarrollo", "evaluacion"}
    ]
    return manifest, rows


def extract_archive(
    source_zip: Path,
    archive_member: str,
    rows: list[dict[str, object]],
    seven_zip: str,
    destination: Path,
) -> Path:
    run_7zip(seven_zip, ["x", str(source_zip), archive_member, f"-o{destination}", "-y"])
    rar_name = Path(archive_member).name
    rar_files = list(destination.rglob(rar_name))
    if len(rar_files) != 1:
        raise RuntimeError(f"No se pudo ubicar {rar_name} dentro del ZIP")

    csv_dir = destination / "csv"
    csv_dir.mkdir()
    include_args = [f"-ir!{row['filename']}" for row in rows]
    run_7zip(seven_zip, ["e", str(rar_files[0]), f"-o{csv_dir}", *include_args, "-y"])
    return csv_dir


def main() -> None:
    args = parse_args()
    source_zip = args.source_zip.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    config = FeatureConfig(
        window_size=args.window_size,
        hop_size=args.hop_size,
        windows_per_part=args.windows_per_part,
        band_count=args.band_count,
        welch_nperseg=args.welch_nperseg,
        analysis_window=args.analysis_window,
    )
    config.validate()
    manifest, rows = selected_rows(manifest_path)
    seven_zip = resolve_seven_zip(args.seven_zip)

    run_root = PROJECT_ROOT / "resultados" / "runs" / args.run_id
    partial_dir = run_root / "features" / "por_archivo"
    partial_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[archive_for(row)].append(row)

    for archive_index, (archive_member, archive_rows) in enumerate(grouped.items(), start=1):
        print(f"[{archive_index}/{len(grouped)}] {Path(archive_member).name}")
        with tempfile.TemporaryDirectory(prefix="aps_dronerf_") as temporary:
            csv_dir = extract_archive(
                source_zip,
                archive_member,
                archive_rows,
                seven_zip,
                Path(temporary),
            )
            for file_index, row in enumerate(archive_rows, start=1):
                source = csv_dir / str(row["filename"])
                if not source.is_file():
                    raise FileNotFoundError(source)
                if int(row["size_bytes"]) != source.stat().st_size:
                    raise ValueError(f"Tamaño inesperado para {row['filename']}")
                if args.verify_hashes and sha256_file(source) != row["sha256"]:
                    raise ValueError(f"Hash inesperado para {row['filename']}")

                materialized_row = {**row, "absolute_path": str(source)}
                table = extract_file_features(materialized_row, config)
                output = partial_dir / (
                    f"{row['partition']}_{row['code']}_{row['part']}_"
                    f"{int(row['segment']):02d}.csv"
                )
                table.to_csv(output, index=False)
                print(f"  {file_index:02d}/{len(archive_rows):02d} {row['filename']}")
                del table
                gc.collect()

    partial_paths = [
        partial_dir
        / f"{row['partition']}_{row['code']}_{row['part']}_{int(row['segment']):02d}.csv"
        for row in rows
    ]
    features = pd.concat((pd.read_csv(path) for path in partial_paths), ignore_index=True)
    features = features.sort_values(["partition", "code", "segment", "part", "start"])
    output = run_root / "features" / "features.csv"
    features.to_csv(output, index=False)

    config_payload = {
        "dataset": "DroneRF",
        "dataset_version": manifest["dataset_version"],
        "dataset_manifest": manifest_path.name,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "partitions": ["desarrollo", "evaluacion"],
        "feature_config": config_dict(config),
    }
    write_json(run_root / "config.json", config_payload)
    write_json(
        run_root / "manifest.json",
        {
            "schema_version": "1.1",
            "run_id": args.run_id,
            "status": "features_ready",
            "created_at_utc": utc_now(),
            "git_commit": git_commit(PROJECT_ROOT),
            "source": "DroneRF",
            "source_archive": source_zip.name,
            "config": config_payload,
            "counts": {
                "files": len(rows),
                "groups": int(features["group_id"].nunique()),
                "windows": int(len(features)),
                "partitions": features["partition"].value_counts().to_dict(),
            },
            "artifacts": {
                "features": {
                    "path": output.relative_to(PROJECT_ROOT).as_posix(),
                    "size_bytes": int(output.stat().st_size),
                    "sha256": sha256_file(output),
                }
            },
        },
    )
    print(f"Features regeneradas: {output}")


if __name__ == "__main__":
    main()
