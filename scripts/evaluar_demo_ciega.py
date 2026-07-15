"""Ejecuta una sola pasada ciega y consulta las etiquetas recién al terminar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.demo import (
    extract_input_features,
    load_bundle,
    load_signal_input,
    predict_hierarchy,
)
from aps_drone_rf.metrics import classification_report_dict
from aps_drone_rf.provenance import sha256_file, utc_now, write_json


def stage_metrics(table: pd.DataFrame, truth_column: str, prediction_column: str):
    truth = table[truth_column].astype(str)
    prediction = table[prediction_column].fillna("no_concluyente").astype(str)
    report = classification_report_dict(truth, prediction)
    report["coverage"] = float((prediction != "no_concluyente").mean())
    report["n_groups"] = int(len(table))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "resultados" / "runs" / "actual" / "modelos" / "bundle.joblib",
    )
    parser.add_argument("--samples-dir", type=Path, default=PROJECT_ROOT / "muestras_demo")
    parser.add_argument("--run-id", default="dronerf_demo_v2_baseline_n512_hann_b20_seed42")
    parser.add_argument("--evaluation-id", default="ventanas_distribuidas_v2")
    args = parser.parse_args()

    samples_root = args.samples_dir.expanduser().resolve()
    bundle = load_bundle(args.bundle)
    package_paths = sorted(samples_root.rglob("*.npz"))
    if not package_paths:
        raise FileNotFoundError("No hay paquetes ciegos para evaluar")

    predictions = []
    for path in package_paths:
        signal_input = load_signal_input(path)
        features = extract_input_features(signal_input, bundle)
        result = predict_hierarchy(bundle, features)
        predictions.append(
            {
                "relative_path": path.relative_to(samples_root).as_posix(),
                "package_sha256": sha256_file(path),
                "pred_activity": (
                    "fondo"
                    if result["state"] == "fondo"
                    else "dron"
                    if result["state"] == "dron"
                    else "no_concluyente"
                ),
                "pred_model": result.get("model") or "no_concluyente",
                "pred_mode": result.get("mode") or "no_concluyente",
                "drone_score": result["drone_score"],
                "margin": result["margin"],
                "last_reliable_level": result["last_reliable_level"],
            }
        )

    catalog_path = samples_root / "catalogo_privado.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    expected_by_path = {item["relative_path"]: item for item in catalog["samples"]}
    for row in predictions:
        expected = expected_by_path[row["relative_path"]]
        row["group_id"] = expected["source_group_id"]
        row["expected_activity"] = expected["expected"]["activity"]
        row["expected_model"] = expected["expected"]["model"]
        row["expected_mode"] = expected["expected"]["mode"]

    table = pd.DataFrame(predictions)
    drone_rows = table[table["expected_activity"] == "dron"].copy()
    metrics = {
        "schema_version": "1.0",
        "executed_at_utc": utc_now(),
        "method": "Primero inferencia sin etiquetas; después unión con catálogo separado.",
        "bundle_sha256": sha256_file(args.bundle),
        "activity": stage_metrics(table, "expected_activity", "pred_activity"),
        "model": stage_metrics(drone_rows, "expected_model", "pred_model"),
        "mode": stage_metrics(drone_rows, "expected_mode", "pred_mode"),
        "exact_hierarchy_rate": float(
            (
                (table["expected_activity"] == table["pred_activity"])
                & (
                    (table["expected_activity"] == "fondo")
                    | (
                        (table["expected_model"] == table["pred_model"])
                        & (table["expected_mode"] == table["pred_mode"])
                    )
                )
            ).mean()
        ),
    }

    run_root = PROJECT_ROOT / "resultados" / "runs" / args.run_id
    table_path = run_root / "tablas" / f"predicciones_demo_{args.evaluation_id}.csv"
    metrics_path = run_root / "metricas" / f"metricas_demo_{args.evaluation_id}.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    write_json(metrics_path, metrics)

    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest["status"] = "demo_protocol_checked"
    manifest["calibration"]["demo_groups_accessed"] = int(table["group_id"].nunique())
    manifest.setdefault("demo_protocols", {})[args.evaluation_id] = {
        "executed_at_utc": metrics["executed_at_utc"],
        "samples": int(len(table)),
    }
    manifest["artifacts"].update(
        {
            f"demo_metrics_{args.evaluation_id}": {
                "path": metrics_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": int(metrics_path.stat().st_size),
                "sha256": sha256_file(metrics_path),
            },
            f"demo_predictions_{args.evaluation_id}": {
                "path": table_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": int(table_path.stat().st_size),
                "sha256": sha256_file(table_path),
            },
        }
    )
    write_json(manifest_path, manifest)
    visible_columns = ["relative_path", "pred_activity", "pred_model", "pred_mode"]
    print(table[visible_columns].to_string(index=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
