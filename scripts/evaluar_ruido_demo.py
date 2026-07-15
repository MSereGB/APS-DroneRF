"""Mide sensibilidad del bundle congelado ante ruido blanco controlado."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
from aps_drone_rf.estilo import COLORES, aplicar_estilo_matplotlib
from aps_drone_rf.metrics import classification_report_dict
from aps_drone_rf.provenance import sha256_file, utc_now, write_json

DEFAULT_RUN_ID = "dronerf_demo_v2_final_n1024_hann_b20_seed42"


def add_noise(signal_input: SignalInput, snr_db: float, rng) -> SignalInput:
    """Agrega ruido a las ventanas ya normalizadas sin consultar etiquetas."""

    if signal_input.window_sets is None:
        raise ValueError("La prueba de ruido requiere paquetes de ventanas L/H")
    noisy_sets = {}
    for part, windows in signal_input.window_sets.items():
        values = np.asarray(windows, dtype=float)
        signal_power = float(np.mean(values**2))
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noisy_sets[part] = values + rng.normal(
            scale=np.sqrt(noise_power),
            size=values.shape,
        )
    return SignalInput(
        signals={part: values.reshape(-1) for part, values in noisy_sets.items()},
        fs_hz=signal_input.fs_hz,
        source_name=signal_input.source_name,
        window_sets=noisy_sets,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "resultados" / "runs" / "actual" / "modelos" / "bundle.joblib",
    )
    parser.add_argument("--samples-dir", type=Path, default=PROJECT_ROOT / "muestras_demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples_root = args.samples_dir.expanduser().resolve()
    catalog = json.loads(
        (samples_root / "catalogo_privado.json").read_text(encoding="utf-8-sig")
    )
    expected = {item["relative_path"]: item["expected"]["activity"] for item in catalog["samples"]}
    bundle = load_bundle(args.bundle)
    snr_values = [-5, 0, 5, 10, 20]
    rows = []
    for snr_db in snr_values:
        for sample_index, path in enumerate(sorted(samples_root.rglob("*.npz"))):
            relative = path.relative_to(samples_root).as_posix()
            clean_input = load_signal_input(path)
            rng = np.random.default_rng(args.seed + sample_index + int((snr_db + 5) * 100))
            noisy_input = add_noise(clean_input, snr_db, rng)
            features = extract_input_features(noisy_input, bundle)
            result = predict_hierarchy(bundle, features)
            prediction = (
                "fondo"
                if result["state"] == "fondo"
                else "dron"
                if result["state"] == "dron"
                else "no_concluyente"
            )
            rows.append(
                {
                    "snr_db": snr_db,
                    "relative_path": relative,
                    "prediction": prediction,
                    "expected": expected[relative],
                    "drone_score": result["drone_score"],
                }
            )

    predictions = pd.DataFrame(rows)
    summaries = []
    for snr_db, table in predictions.groupby("snr_db", sort=True):
        report = classification_report_dict(table["expected"], table["prediction"])
        summaries.append(
            {
                "snr_db": int(snr_db),
                "accuracy": report["accuracy"],
                "balanced_accuracy": report["balanced_accuracy"],
                "f1_macro": report["f1_macro"],
                "coverage": float((table["prediction"] != "no_concluyente").mean()),
            }
        )
    summary = pd.DataFrame(summaries).sort_values("snr_db")

    run_root = PROJECT_ROOT / "resultados" / "runs" / args.run_id
    table_path = run_root / "tablas" / "sensibilidad_ruido.csv"
    prediction_path = run_root / "tablas" / "predicciones_ruido.csv"
    metrics_path = run_root / "metricas" / "sensibilidad_ruido.json"
    figure_path = run_root / "figuras" / "05_sensibilidad_ruido.png"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    metrics = {
        "schema_version": "1.0",
        "executed_at_utc": utc_now(),
        "bundle_sha256": sha256_file(args.bundle),
        "seed": args.seed,
        "interpretation": "Ruido relativo agregado después de normalizar; no potencia física.",
        "results": summary.to_dict(orient="records"),
    }
    write_json(metrics_path, metrics)

    aplicar_estilo_matplotlib()
    fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
    ax.plot(
        summary["snr_db"],
        summary["balanced_accuracy"],
        marker="o",
        linewidth=2,
        color=COLORES["rosa"],
        label="Exactitud balanceada",
    )
    ax.plot(
        summary["snr_db"],
        summary["coverage"],
        marker="s",
        linewidth=2,
        color=COLORES["violeta"],
        label="Cobertura",
    )
    ax.set(
        title="Sensibilidad de fondo/dron al ruido agregado",
        xlabel="SNR controlada [dB]",
        ylabel="Valor por grupo",
        xticks=snr_values,
        ylim=(0, 1.05),
    )
    ax.legend(loc="lower right")
    fig.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for name, path in {
        "noise_metrics": metrics_path,
        "noise_summary": table_path,
        "noise_predictions": prediction_path,
        "noise_figure": figure_path,
    }.items():
        manifest["artifacts"][name] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    manifest["noise_sensitivity_completed_at_utc"] = metrics["executed_at_utc"]
    write_json(manifest_path, manifest)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
