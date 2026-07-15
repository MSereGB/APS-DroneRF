"""Extrae features APS reales en una corrida aislada y reanudable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.pipeline import (
    FeatureConfig,
    config_dict,
    extract_file_features,
    load_dataset_manifest,
    validate_materialized_files,
)
from aps_drone_rf.provenance import git_commit, sha256_file, utc_now, write_json


def read_or_create_config(path: Path, expected: dict[str, object]) -> None:
    """Impide reanudar una corrida con parámetros diferentes."""

    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8-sig"))
        if current != expected:
            raise ValueError("La corrida ya existe con otra configuración")
        return
    write_json(path, expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "dronerf_demo_v2_manifest.json",
    )
    parser.add_argument("--run-id", default="dronerf_demo_v2_baseline_n512_hann_b20_seed42")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--windows-per-part", type=int, default=100)
    parser.add_argument("--band-count", type=int, choices=[4, 10, 20], default=20)
    parser.add_argument("--welch-nperseg", type=int, default=256)
    parser.add_argument(
        "--analysis-window",
        choices=["hann", "hamming", "blackman"],
        default="hann",
    )
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=["desarrollo", "evaluacion", "demo"],
        default=["desarrollo", "evaluacion"],
    )
    parser.add_argument(
        "--allow-demo",
        action="store_true",
        help="Permite extraer demo para inspección separada; nunca para calibración.",
    )
    args = parser.parse_args()

    partitions = set(args.partitions)
    if "demo" in partitions and not args.allow_demo:
        raise ValueError("La partición demo requiere --allow-demo y no debe calibrar modelos")

    config = FeatureConfig(
        window_size=args.window_size,
        hop_size=args.hop_size,
        windows_per_part=args.windows_per_part,
        band_count=args.band_count,
        welch_nperseg=args.welch_nperseg,
        analysis_window=args.analysis_window,
    )
    config.validate()
    manifest = load_dataset_manifest(args.manifest)
    rows = validate_materialized_files(manifest, args.data_dir, partitions)

    run_root = PROJECT_ROOT / "resultados" / "runs" / args.run_id
    features_dir = run_root / "features"
    partial_dir = features_dir / "por_archivo"
    partial_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "dataset": "DroneRF",
        "dataset_version": manifest["dataset_version"],
        "dataset_manifest": args.manifest.name,
        "dataset_manifest_sha256": sha256_file(args.manifest),
        "partitions": sorted(partitions),
        "feature_config": config_dict(config),
    }
    read_or_create_config(run_root / "config.json", run_config)

    partial_paths = []
    for row in tqdm(rows, desc="Archivos DroneRF", unit="archivo"):
        filename = (
            f"{row['partition']}_{row['code']}_{row['part']}_{int(row['segment']):02d}.csv"
        )
        output = partial_dir / filename
        partial_paths.append(output)
        if output.is_file():
            previous = pd.read_csv(output)
            if len(previous) == config.windows_per_part:
                continue
            raise ValueError(f"Salida parcial incompleta: {output}")
        if args.verify_hashes and sha256_file(row["absolute_path"]) != row["sha256"]:
            raise ValueError(f"Hash inesperado para {row['relative_path']}")
        table = extract_file_features(row, config)
        if not table.replace([float("inf"), float("-inf")], pd.NA).notna().all().all():
            raise ValueError(f"Features no finitas para {row['relative_path']}")
        table.to_csv(output, index=False)

    tables = [pd.read_csv(path) for path in partial_paths]
    features = pd.concat(tables, ignore_index=True)
    features = features.sort_values(["partition", "code", "segment", "part", "start"])
    output = features_dir / "features.csv"
    features.to_csv(output, index=False)

    group_partitions = features.groupby("group_id")["partition"].nunique()
    if (group_partitions != 1).any():
        raise ValueError("Un group_id apareció en más de una partición")
    group_parts = features.groupby("group_id")["part"].apply(set)
    if any(parts != {"L", "H"} for parts in group_parts):
        raise ValueError("Hay grupos sin las partes L/H completas")

    run_manifest = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "status": "features_ready",
        "created_at_utc": utc_now(),
        "git_commit": git_commit(PROJECT_ROOT),
        "source": "DroneRF",
        "config": run_config,
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
    }
    write_json(run_root / "manifest.json", run_manifest)
    print(f"Features: {output}")
    print(f"Filas: {len(features)}, grupos: {features['group_id'].nunique()}")


if __name__ == "__main__":
    main()
