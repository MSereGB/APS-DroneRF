"""Ensaya el recorrido técnico de la demo sin consultar etiquetas esperadas."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.demo import (
    extract_input_features,
    load_bundle,
    load_signal_input,
    predict_hierarchy,
)
from aps_drone_rf.provenance import sha256_file, utc_now, write_json
from aps_drone_rf.robustness import agregar_impulsos_controlados


def resumir_resultado(name: str, path: Path, bundle: dict[str, object]) -> dict[str, object]:
    """Procesar una entrada y devolver solo datos disponibles para la inferencia."""

    signal_input = load_signal_input(path)
    features = extract_input_features(signal_input, bundle)
    result = predict_hierarchy(bundle, features)
    return {
        "caso": name,
        "archivo": path.name,
        "partes": sorted(signal_input.signals),
        "estado": result["state"],
        "modelo": result.get("model"),
        "modo": result.get("mode"),
        "puntaje_dron": float(result["drone_score"]),
        "umbral": float(result["threshold"]),
        "margen": float(result["margin"]),
        "compatible_con_dominio": result.get("domain_compatible"),
        "motivo": result.get("stopped_reason"),
    }


def crear_entrada_l_aislada(source: Path, destination: Path) -> None:
    """Crear un archivo legible con L solamente para comprobar la política L/H."""

    with np.load(source, allow_pickle=False) as package:
        np.savez(
            destination,
            senal_l=np.asarray(package["ventanas_l"]).reshape(-1),
            fs_hz=np.asarray(package["fs_hz"]),
        )


def crear_entrada_impulsiva(source: Path, destination: Path) -> None:
    """Crear una entrada L/H con transitorios para comprobar el control de calidad."""

    with np.load(source, allow_pickle=False) as package:
        window_sets = {
            "L": np.asarray(package["ventanas_l"], dtype=float),
            "H": np.asarray(package["ventanas_h"], dtype=float),
        }
        changed = agregar_impulsos_controlados(
            window_sets,
            fraction=0.005,
            amplitude_rms=8.0,
            seed=20_260_714,
        )
        np.savez_compressed(
            destination,
            ventanas_l=changed["L"].astype(np.float32),
            ventanas_h=changed["H"].astype(np.float32),
            fs_hz=np.asarray(package["fs_hz"]),
            preprocessed=np.array([1], dtype=np.uint8),
        )


def _comparar_resultados(reference: dict[str, object], copied: dict[str, object]) -> bool:
    """Comprobar que copiar la muestra fuera de la biblioteca no altera la inferencia."""

    same_text = all(
        reference[key] == copied[key]
        for key in ("partes", "estado", "modelo", "modo", "motivo")
    )
    same_scores = all(
        abs(float(reference[key]) - float(copied[key])) < 1e-12
        for key in ("puntaje_dron", "umbral", "margen")
    )
    return bool(same_text and same_scores)


def main() -> None:
    """Ejecutar fondo, actividad, parte aislada, carga externa y transitorios."""

    parser = argparse.ArgumentParser(
        description="Ensaya el recorrido técnico de la demo APS DroneRF sin leer etiquetas."
    )
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
    parser.add_argument("--samples-dir", type=Path, default=PROJECT_ROOT / "muestras_demo")
    parser.add_argument("--background", type=Path)
    parser.add_argument("--activity", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON opcional para dejar constancia local del ensayo.",
    )
    args = parser.parse_args()

    samples_dir = args.samples_dir.expanduser().resolve()
    background = (
        args.background.expanduser().resolve()
        if args.background
        else samples_dir / "sin_dron" / "muestra_01.npz"
    )
    activity = (
        args.activity.expanduser().resolve()
        if args.activity
        else samples_dir / "phantom" / "conectado" / "muestra_01.npz"
    )
    for path in (background, activity):
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró la muestra: {path}")

    bundle_path = args.bundle.expanduser().resolve()
    bundle = load_bundle(bundle_path)
    background_result = resumir_resultado("biblioteca_fondo", background, bundle)
    activity_result = resumir_resultado("biblioteca_actividad", activity, bundle)

    with tempfile.TemporaryDirectory(prefix="aps_dronerf_ensayo_") as temporary_name:
        temporary_dir = Path(temporary_name)
        only_l = temporary_dir / "muestra_l_aislada.npz"
        uploaded_copy = temporary_dir / "muestra_subida.npz"
        impulsive = temporary_dir / "muestra_impulsiva.npz"
        crear_entrada_l_aislada(activity, only_l)
        crear_entrada_impulsiva(background, impulsive)
        shutil.copy2(activity, uploaded_copy)
        only_l_result = resumir_resultado("parte_l_aislada", only_l, bundle)
        uploaded_result = resumir_resultado("archivo_subido", uploaded_copy, bundle)
        impulsive_result = resumir_resultado("transitorios_controlados", impulsive, bundle)

    checks = {
        "fondo_resuelto": background_result["estado"] == "fondo",
        "actividad_resuelta": activity_result["estado"] == "dron",
        "parte_l_no_concluyente": only_l_result["estado"] == "no_concluyente",
        "copia_externa_igual": _comparar_resultados(activity_result, uploaded_result),
        "transitorios_no_concluyente": impulsive_result["estado"] == "no_concluyente",
    }
    payload = {
        "executed_at_utc": utc_now(),
        "purpose": (
            "Ensayo técnico local. No consulta catalogo_privado.json ni usa etiquetas "
            "esperadas durante la inferencia."
        ),
        "bundle": {
            "name": bundle_path.name,
            "sha256": sha256_file(bundle_path),
        },
        "cases": [
            background_result,
            activity_result,
            only_l_result,
            uploaded_result,
            impulsive_result,
        ],
        "checks": checks,
    }
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, payload)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit("El ensayo técnico no pasó todos los controles")


if __name__ == "__main__":
    main()
