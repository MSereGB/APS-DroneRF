"""Somete entradas excluidas a estresores sintéticos sin recalibrar el detector."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.demo import (
    SignalInput,
    extract_input_features,
    load_bundle,
    load_signal_input,
    predict_hierarchy,
)
from aps_drone_rf.provenance import sha256_file, utc_now, write_json
from aps_drone_rf.robustness import (
    agregar_impulsos_controlados,
    agregar_tono_controlado,
    plot_perturbation_summary,
)

DEFAULT_RUN_ID = "dronerf_demo_v2_final_n1024_hann_b20_seed42"


def _tono(entry: SignalInput, *, frequency_hz: float, sir_db: float) -> SignalInput:
    """Suma un tono relativo a las ventanas ya normalizadas de la demostración."""

    if entry.window_sets is None:
        raise ValueError("La auditoría necesita paquetes NPZ con partes L/H")
    changed = agregar_tono_controlado(
        entry.window_sets,
        fs_hz=entry.fs_hz,
        frequency_hz=frequency_hz,
        sir_db=sir_db,
    )
    return SignalInput(
        signals={part: values.reshape(-1) for part, values in changed.items()},
        fs_hz=entry.fs_hz,
        source_name=entry.source_name,
        window_sets=changed,
    )


def _impulsos(entry: SignalInput, *, seed: int) -> SignalInput:
    """Inyecta impulsos poco frecuentes a las ventanas de la demostración."""

    if entry.window_sets is None:
        raise ValueError("La auditoría necesita paquetes NPZ con partes L/H")
    changed = agregar_impulsos_controlados(
        entry.window_sets,
        fraction=0.005,
        amplitude_rms=8.0,
        seed=seed,
    )
    return SignalInput(
        signals={part: values.reshape(-1) for part, values in changed.items()},
        fs_hz=entry.fs_hz,
        source_name=entry.source_name,
        window_sets=changed,
    )


def _activity_state(result: dict[str, object]) -> str:
    """Reduce la salida de la jerarquía al nivel binario auditado."""

    return str(result["state"])


def _infer_without_labels(
    sample_paths: list[Path],
    samples_root: Path,
    bundle: dict[str, object],
    transform: Callable[[SignalInput], SignalInput],
) -> list[dict[str, object]]:
    """Infiere todos los paquetes antes de consultar el catálogo privado."""

    rows = []
    for path in sample_paths:
        transformed = transform(load_signal_input(path))
        features = extract_input_features(transformed, bundle)
        result = predict_hierarchy(bundle, features)
        rows.append(
            {
                "relative_path": path.relative_to(samples_root).as_posix(),
                "estado": _activity_state(result),
                "puntaje_dron": float(result["drone_score"]),
                "umbral": float(result["threshold"]),
                "margen": float(result["margin"]),
                "compatible_con_dominio": result.get("domain_compatible"),
                "motivo": result.get("stopped_reason"),
            }
        )
    return rows


def _catalog_expected(samples_root: Path) -> dict[str, str]:
    """Extrae la actividad esperada cuando ya se completaron las inferencias."""

    catalog = json.loads(
        (samples_root / "catalogo_privado.json").read_text(encoding="utf-8-sig")
    )
    return {
        str(item["relative_path"]): str(item["expected"]["activity"])
        for item in catalog["samples"]
    }


def _summarize_condition(table: pd.DataFrame, condition: str) -> dict[str, object]:
    """Cuenta falsas alarmas, pérdidas y rechazos sin ocultar no conclusiones."""

    background = table["esperado"] == "fondo"
    drone = table["esperado"] == "dron"
    predicted_drone = table["estado"] == "dron"
    predicted_background = table["estado"] == "fondo"
    false_alarms = int((background & predicted_drone).sum())
    missed = int((drone & ~predicted_drone).sum())
    background_accuracy = float(predicted_background[background].mean())
    detection_rate = float(predicted_drone[drone].mean())
    return {
        "condicion": condition,
        "grupos": int(len(table)),
        "grupos_fondo": int(background.sum()),
        "grupos_dron": int(drone.sum()),
        "exactitud_balanceada": float((background_accuracy + detection_rate) / 2.0),
        "tasa_deteccion": detection_rate,
        "falsas_alarmas": false_alarms,
        "tasa_falsa_alarma": float(false_alarms / max(int(background.sum()), 1)),
        "perdidas": missed,
        "tasa_perdida": float(missed / max(int(drone.sum()), 1)),
        "no_concluyentes": int((table["estado"] == "no_concluyente").sum()),
        "tasa_no_concluyente": float((table["estado"] == "no_concluyente").mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--samples-dir", type=Path, default=PROJECT_ROOT / "muestras_demo")
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
    parser.add_argument("--seed", type=int, default=20_260_714)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    samples_root = args.samples_dir.expanduser().resolve()
    sample_paths = sorted(samples_root.rglob("*.npz"))
    if not sample_paths:
        raise FileNotFoundError("No se encontraron paquetes NPZ de demostración")
    bundle_path = args.bundle.expanduser().resolve()
    bundle = load_bundle(bundle_path)
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "resultados"
        / "runs"
        / args.run_id
        / "auditorias"
        / "estresores_entrada_v1"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions: list[tuple[str, Callable[[SignalInput], SignalInput], str]] = [
        ("referencia", lambda entry: entry, "Paquete excluido sin perturbación."),
        (
            "tono_10_5_mhz_20_db",
            lambda entry: _tono(entry, frequency_hz=10.5e6, sir_db=20.0),
            "Tono agregado a 10,5 MHz con relación señal/interferencia relativa de 20 dB.",
        ),
        (
            "tono_10_5_mhz_10_db",
            lambda entry: _tono(entry, frequency_hz=10.5e6, sir_db=10.0),
            "Tono agregado a 10,5 MHz con relación señal/interferencia relativa de 10 dB.",
        ),
        (
            "impulsos_0_5_pct",
            lambda entry: _impulsos(entry, seed=args.seed),
            "Impulsos bipolares en 0,5 % de muestras, amplitud relativa de 8 RMS.",
        ),
    ]

    all_rows = []
    for condition, transform, _ in conditions:
        for row in _infer_without_labels(sample_paths, samples_root, bundle, transform):
            all_rows.append({"condicion": condition, **row})

    predictions = pd.DataFrame(all_rows)
    expected = _catalog_expected(samples_root)
    predictions["esperado"] = predictions["relative_path"].map(expected)
    if predictions["esperado"].isna().any():
        raise ValueError("Alguna muestra auditada no figura en el catálogo privado")

    summary = pd.DataFrame(
        [
            _summarize_condition(
                predictions[predictions["condicion"] == condition], condition
            )
            for condition, _, _ in conditions
        ]
    )
    descriptions = {condition: description for condition, _, description in conditions}
    summary["descripcion"] = summary["condicion"].map(descriptions)

    predictions_path = output_dir / "predicciones_estresores_entrada.csv"
    summary_path = output_dir / "resumen_estresores_entrada.csv"
    metrics_path = output_dir / "metricas_estresores_entrada.json"
    figure_path = output_dir / "estresores_entrada.png"
    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)
    metrics = {
        "schema_version": "1.0",
        "executed_at_utc": utc_now(),
        "purpose": (
            "Auditoría postcongelación con interferencias sintéticas inyectadas sobre "
            "ventanas de demostración ya normalizadas. No se usa para ajustar modelo, "
            "umbral, dominio ni control de calidad."
        ),
        "bundle_sha256": sha256_file(bundle_path),
        "seed": args.seed,
        "samples": len(sample_paths),
        "label_accessed_after_inference": True,
        "conditions": {condition: description for condition, _, description in conditions},
        "summary": summary.to_dict(orient="records"),
    }
    write_json(metrics_path, metrics)

    figure = plot_perturbation_summary(summary)
    figure.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(summary.drop(columns="descripcion").to_string(index=False))
    print(f"Resultados: {output_dir}")


if __name__ == "__main__":
    main()
