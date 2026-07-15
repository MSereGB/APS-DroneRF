"""Prueba controles sintéticos fuera de DroneRF contra el bundle de demostración."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.demo import (
    SignalInput,
    extract_input_features,
    load_bundle,
    predict_hierarchy,
)
from aps_drone_rf.estilo import COLORES, aplicar_estilo_matplotlib
from aps_drone_rf.provenance import sha256_file, utc_now, write_json


def controles_sinteticos(fs_hz: float, samples: int, seed: int) -> dict[str, np.ndarray]:
    """Genera señales conocidas que no representan registros DroneRF."""

    tt = np.arange(samples) / fs_hz
    rng = np.random.default_rng(seed)
    chirp_rate = 12e6 / (samples / fs_hz)
    return {
        "ruido_blanco": rng.normal(size=samples),
        "tono_1_mhz": np.sin(2 * np.pi * 1e6 * tt),
        "tono_10_5_mhz": np.sin(2 * np.pi * 10.5e6 * tt),
        "dos_tonos": np.sin(2 * np.pi * 4e6 * tt) + 0.7 * np.sin(2 * np.pi * 12e6 * tt),
        "am_10_5_mhz": (1 + 0.6 * np.sin(2 * np.pi * 40e3 * tt))
        * np.sin(2 * np.pi * 10.5e6 * tt),
        "chirp_4_a_16_mhz": np.sin(2 * np.pi * (4e6 * tt + 0.5 * chirp_rate * tt**2)),
    }


def add_artifact(manifest: dict[str, object], name: str, path: Path) -> None:
    """Registra una salida con ubicación, tamaño y hash."""

    manifest.setdefault("artifacts", {})[name] = {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="dronerf_demo_v2_final_n1024_hann_b20_seed42")
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
    parser.add_argument("--samples", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=20_260_714)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    fs_hz = float(bundle["feature_config"]["feature_config"]["fs_hz"])
    rng = np.random.default_rng(args.seed + 1)
    rows = []
    for name, signal in controles_sinteticos(fs_hz, args.samples, args.seed).items():
        signal_input = SignalInput(
            signals={"L": signal, "H": 0.98 * signal + 0.02 * rng.normal(size=len(signal))},
            fs_hz=fs_hz,
            source_name=name,
        )
        features = extract_input_features(signal_input, bundle)
        result = predict_hierarchy(bundle, features)
        rows.append(
            {
                "control": name,
                "estado": result["state"],
                "puntaje_dron": float(result["drone_score"]),
                "distancia_dominio": float(result["domain_distance"]),
                "limite_dominio": float(result["domain_threshold"]),
                "compatible": bool(result["domain_compatible"]),
            }
        )

    table = pd.DataFrame(rows)
    run_root = PROJECT_ROOT / "resultados" / "runs" / args.run_id
    tables_dir = run_root / "tablas"
    metrics_dir = run_root / "metricas"
    figures_dir = run_root / "figuras"
    for folder in (tables_dir, metrics_dir, figures_dir):
        folder.mkdir(parents=True, exist_ok=True)

    table_path = tables_dir / "controles_sinteticos_dominio.csv"
    table.to_csv(table_path, index=False)
    metrics_path = metrics_dir / "controles_sinteticos_dominio.json"
    metrics = {
        "schema_version": "1.0",
        "executed_at_utc": utc_now(),
        "purpose": "Control negativo fuera de DroneRF; no se usa para reentrenar.",
        "bundle_sha256": sha256_file(args.bundle),
        "controls": int(len(table)),
        "rejected_as_outside_domain": int((~table["compatible"]).sum()),
        "drone_decisions": int((table["estado"] == "dron").sum()),
        "rows": table.to_dict(orient="records"),
    }
    write_json(metrics_path, metrics)

    aplicar_estilo_matplotlib()
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.bar(
        table["control"],
        table["distancia_dominio"],
        color=COLORES["rosa"],
        label="Distancia al dominio",
    )
    axis.axhline(
        float(table["limite_dominio"].iloc[0]),
        color=COLORES["violeta"],
        linestyle="--",
        label="Límite calibrado con desarrollo",
    )
    axis.set(
        title="Controles sintéticos fuera del dominio DroneRF",
        xlabel="Control",
        ylabel="Distancia relativa",
        yscale="log",
    )
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    fig.tight_layout()
    figure_path = figures_dir / "10_controles_sinteticos_dominio.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    add_artifact(manifest, "controles_sinteticos_tabla", table_path)
    add_artifact(manifest, "controles_sinteticos_metricas", metrics_path)
    add_artifact(manifest, "controles_sinteticos_figura", figure_path)
    write_json(manifest_path, manifest)

    print(table.to_string(index=False))
    if args.strict and ((table["estado"] != "no_concluyente").any() or table["compatible"].any()):
        raise SystemExit("Algún control sintético no fue bloqueado por el dominio")


if __name__ == "__main__":
    main()
