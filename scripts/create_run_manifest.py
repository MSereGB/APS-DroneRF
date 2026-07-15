"""Crea un manifest liviano para una corrida final del TPF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def default_artifacts(project_root: Path) -> list[Path]:
    """Devuelve los artefactos ligeros de la corrida DroneRF v1 que existan."""

    relative_paths = [
        "resultados/features/features_dronerf_subset_v1.csv",
        "resultados/metrics/validation_metrics.json",
        "resultados/tables/validation_summary.csv",
        "resultados/tables/nb02_resumen_temporal.csv",
        "resultados/tables/nb02_fft_frecuencias_dominantes.csv",
        "resultados/tables/nb03_potencia_relativa_bandas.csv",
        "resultados/tables/nb04_filtros.csv",
        "resultados/tables/nb04_potencia_filtrado.csv",
        "resultados/tables/nb04_transformada_z.csv",
        "resultados/tables/nb05_resumen_caracteristicas_por_clase.csv",
        "resultados/figures/nb02_tiempo_fondo_vs_dron.png",
        "resultados/figures/nb02_fft_fondo_vs_dron.png",
        "resultados/figures/nb02_welch_fondo_vs_dron.png",
        "resultados/figures/nb03_potencia_relativa_bandas.png",
        "resultados/figures/nb04_respuesta_filtros.png",
        "resultados/figures/nb04_polos_ceros_iir.png",
        "resultados/figures/nb04_psd_filtrada.png",
        "resultados/figures/nb05_caracteristicas_clave.png",
        "informe/memoria_final.md",
        "informe/memoria_final.tex",
        "output/pdf/memoria_final_aps_dronerf.pdf",
        "output/presentations/defensa_aps_dronerf.pptx",
    ]
    return [project_root / path for path in relative_paths if (project_root / path).exists()]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))

    from aps_drone_rf.provenance import (
        describe_file,
        git_commit,
        git_worktree_status,
        runtime_versions,
        utc_now,
        write_json,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="dronerf_subset_v1")
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "drone_rf_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "resultados" / "manifests" / "dronerf_subset_v1_run.json",
    )
    parser.add_argument("--fs", type=float, default=40_000_000.0)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--max-windows-per-record", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--command", action="append", default=[])
    args = parser.parse_args()

    dataset_manifest = args.dataset_manifest.resolve()
    if not dataset_manifest.exists():
        raise FileNotFoundError(f"No existe el manifest de datos: {dataset_manifest}")
    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))

    artifacts = [dataset_manifest]
    requested_artifacts = args.artifact or default_artifacts(project_root)
    artifacts.extend(path.resolve() for path in requested_artifacts if path.exists())
    output_records = [describe_file(path, relative_to=project_root) for path in artifacts]
    commands = args.command or [
        (
            "python scripts/prepare_dataset.py --data-dir <DATA_DIR> "
            "--output data/interim/manifest_inicial.csv"
        ),
        (
            "python scripts/build_features.py --data-dir <DATA_DIR> --no-synthetic-fallback "
            "--max-records 24 --max-windows-per-record 200"
        ),
        (
            "python scripts/run_validation.py "
            "--features resultados/features/features_dronerf_subset_v1.csv "
            "--preferred-splits 5 --max-windows-per-group 200"
        ),
        "python scripts/export_figures.py --data-dir <DATA_DIR> --dronerf-fs 40000000",
        "python scripts/run_sensitivity.py",
    ]

    payload = {
        "schema_version": "1.0",
        "experiment_id": args.experiment_id,
        "created_at_utc": utc_now(),
        "git": {
            "commit": git_commit(project_root),
            "worktree_changes": git_worktree_status(project_root),
        },
        "dataset": {
            "name": dataset_payload.get("dataset"),
            "version": dataset_payload.get("dataset_version"),
            "data_root_name": dataset_payload.get("data_root_name"),
            "file_count": dataset_payload.get("file_count"),
            "group_count": dataset_payload.get("group_count"),
            "class_counts": dataset_payload.get("class_counts"),
            "manifest_sha256": describe_file(dataset_manifest)["sha256"],
        },
        "parameters": {
            "fs_hz": float(args.fs),
            "window_size": int(args.window_size),
            "hop_size": int(args.hop_size),
            "max_windows_per_record": int(args.max_windows_per_record),
            "seed": int(args.seed),
        },
        "commands": commands,
        "runtime": runtime_versions(),
        "artifacts": output_records,
        "interpretation": "Resultados con amplitudes normalizadas y métricas relativas.",
    }
    output = write_json(args.output, payload)
    print(f"Manifest de corrida guardado en {output}")


if __name__ == "__main__":
    main()
