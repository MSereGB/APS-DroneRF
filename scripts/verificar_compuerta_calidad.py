"""Verifica la compuerta de transitorios sobre particiones ya separadas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.demo import load_bundle
from aps_drone_rf.dominio import check_signal_quality
from aps_drone_rf.provenance import sha256_file, utc_now, write_json

DEFAULT_RUN_ID = "dronerf_demo_v2_final_n1024_hann_b20_seed42"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=(
            PROJECT_ROOT
            / "resultados"
            / "runs"
            / "actual"
            / "modelos"
            / "bundle_demo_conservador.joblib"
        ),
    )
    args = parser.parse_args()

    run_root = PROJECT_ROOT / "resultados" / "runs" / args.run_id
    bundle_path = args.bundle.expanduser().resolve()
    bundle = load_bundle(bundle_path)
    features = pd.read_csv(run_root / "features" / "features.csv")
    rows = []
    for partition in ("desarrollo", "evaluacion"):
        selected = features[features["partition"] == partition]
        for group_id, group in selected.groupby("group_id", sort=True):
            # La compuerta recibe únicamente features y partes, no las etiquetas del grupo.
            input_features = group[["group_id", "part", "factor_cresta"]].copy()
            result = check_signal_quality(input_features, bundle["signal_quality_guard"])
            rows.append(
                {
                    "particion": partition,
                    "group_id": group_id,
                    "compatible": bool(result["compatible"]),
                    "factor_cresta_maximo": float(result["maximum"]),
                    "limite": float(result["upper_limit"]),
                    "partes_fuera": ",".join(result["parts_exceeding"]),
                }
            )

    table = pd.DataFrame(rows)
    output_dir = run_root / "auditorias" / "compuerta_calidad_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "compuerta_calidad_particiones.csv"
    metrics_path = output_dir / "compuerta_calidad_particiones.json"
    table.to_csv(table_path, index=False)
    metrics = {
        "schema_version": "1.0",
        "executed_at_utc": utc_now(),
        "purpose": (
            "Verificar una compuerta de transitorios calibrada solo con desarrollo. "
            "No se recalibra con evaluación."
        ),
        "bundle_sha256": sha256_file(bundle_path),
        "development_groups": int((table["particion"] == "desarrollo").sum()),
        "evaluation_groups": int((table["particion"] == "evaluacion").sum()),
        "development_rejected": int(
            ((table["particion"] == "desarrollo") & ~table["compatible"]).sum()
        ),
        "evaluation_rejected": int(
            ((table["particion"] == "evaluacion") & ~table["compatible"]).sum()
        ),
        "quality_guard": bundle["signal_quality_guard"],
    }
    write_json(metrics_path, metrics)
    print(table.groupby("particion")["compatible"].agg(["count", "sum"]).to_string())
    print(f"Resultados: {output_dir}")


if __name__ == "__main__":
    main()
