"""Evalúa fondos DroneRF no incluidos en la selección v2 ni en la demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.demo import SignalInput, extract_input_features, load_bundle, predict_hierarchy
from aps_drone_rf.io import load_signal_file, parse_dronerf_filename
from aps_drone_rf.provenance import sha256_file, utc_now, write_json

DEFAULT_RUN_ID = "dronerf_demo_v2_final_n1024_hann_b20_seed42"


def background_pairs(root: Path) -> list[tuple[int, Path, Path]]:
    """Encuentra pares L/H del desafío sin incorporar una etiqueta a la señal."""

    low_dir = root / "raw" / "00000" / "L"
    high_dir = root / "raw" / "00000" / "H"
    pairs = []
    for low_path in sorted(low_dir.glob("00000L_*.csv")):
        parsed = parse_dronerf_filename(low_path)
        if parsed is None:
            continue
        segment = int(parsed["segment"])
        high_path = high_dir / f"00000H_{segment}.csv"
        if not high_path.is_file():
            raise FileNotFoundError(f"Falta la parte H del segmento {segment}")
        pairs.append((segment, low_path, high_path))
    if not pairs:
        raise FileNotFoundError("No se encontraron pares L/H de fondo")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Carpeta del desafío de fondo con la estructura raw/00000/L y H.",
    )
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    run_root = PROJECT_ROOT / "resultados" / "runs" / args.run_id
    conservative = (
        PROJECT_ROOT
        / "resultados"
        / "runs"
        / "actual"
        / "modelos"
        / "bundle_demo_conservador.joblib"
    )
    bundle_path = args.bundle or conservative
    if not bundle_path.is_file():
        raise FileNotFoundError("Falta el bundle conservador de la demostración")
    bundle = load_bundle(bundle_path)
    pairs = background_pairs(args.data_dir.expanduser().resolve())

    # Primero se procesan arrays sin pasar ninguna etiqueta al backend.
    rows = []
    for segment, low_path, high_path in pairs:
        signal_input = SignalInput(
            signals={"L": load_signal_file(low_path), "H": load_signal_file(high_path)},
            fs_hz=40_000_000.0,
            source_name=f"desafio_fondo_{segment}",
        )
        features = extract_input_features(signal_input, bundle)
        result = predict_hierarchy(bundle, features)
        rows.append(
            {
                "segment": segment,
                "source_l_sha256": sha256_file(low_path),
                "source_h_sha256": sha256_file(high_path),
                "prediction": (
                    "dron"
                    if result["state"] == "dron"
                    else "fondo"
                    if result["state"] == "fondo"
                    else "no_concluyente"
                ),
                "drone_score": float(result["drone_score"]),
                "threshold": float(result["threshold"]),
                "margin": float(result["margin"]),
                "domain_compatible": bool(result["domain_compatible"]),
                "domain_distance": float(result["domain_distance"]),
                "domain_threshold": float(result["domain_threshold"]),
            }
        )

    # La condición de fondo se registra solo después de concluir todas las inferencias.
    table = pd.DataFrame(rows)
    table["expected"] = "fondo"
    false_alarms = table["prediction"] == "dron"
    metrics = {
        "schema_version": "1.0",
        "executed_at_utc": utc_now(),
        "method": (
            "Inferencia congelada sobre cinco pares de fondo extraídos de los segmentos "
            "21-25; condición esperada agregada después de inferir."
        ),
        "bundle_sha256": sha256_file(bundle_path),
        "input_root": str(args.data_dir.expanduser().resolve()),
        "n_background_groups": int(len(table)),
        "false_alarm_count": int(false_alarms.sum()),
        "false_alarm_rate": float(false_alarms.mean()),
        "no_conclusive_count": int((table["prediction"] == "no_concluyente").sum()),
        "out_of_domain_count": int((~table["domain_compatible"]).sum()),
        "true_negative_count": int((table["prediction"] == "fondo").sum()),
        "drone_score_min": float(table["drone_score"].min()),
        "drone_score_max": float(table["drone_score"].max()),
    }

    tables_dir = run_root / "tablas"
    metrics_dir = run_root / "metricas"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / "desafio_fondos_externos.csv"
    metrics_path = metrics_dir / "desafio_fondos_externos.json"
    table.to_csv(table_path, index=False)
    write_json(metrics_path, metrics)

    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for name, path in {
        "desafio_fondos_externos_tabla": table_path,
        "desafio_fondos_externos_metricas": metrics_path,
    }.items():
        manifest["artifacts"][name] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    manifest["external_background_challenge_completed_at_utc"] = metrics["executed_at_utc"]
    write_json(manifest_path, manifest)
    print(
        table[
            ["segment", "prediction", "drone_score", "threshold", "domain_compatible"]
        ].to_string(index=False)
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
