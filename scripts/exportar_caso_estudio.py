"""Exporta un caso excluido con análisis APS e inferencia jerárquica trazable."""

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

from aps_drone_rf.demo import (  # noqa: E402
    extract_input_features,
    load_bundle,
    load_signal_input,
    predict_hierarchy,
)
from aps_drone_rf.dronerf import DISPLAY_NAMES  # noqa: E402
from aps_drone_rf.estilo import COLORES, aplicar_estilo_matplotlib  # noqa: E402
from aps_drone_rf.provenance import describe_file, sha256_file, utc_now, write_json  # noqa: E402
from aps_drone_rf.spectral import (  # noqa: E402
    fft_magnitude,
    relative_psd_diagnostics,
    welch_psd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="dronerf_demo_v2_final_n1024_hann_b20_seed42",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("muestras_demo/phantom/conectado/muestra_01.npz"),
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--window-index", type=int, default=50)
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def median_welch(
    windows: np.ndarray,
    fs_hz: float,
    nperseg: int,
    analysis_window: str,
) -> tuple[np.ndarray, np.ndarray]:
    spectra = []
    frequencies = None
    for window in windows:
        frequencies, psd = welch_psd(
            window,
            fs_hz,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            window=analysis_window,
        )
        spectra.append(psd)
    if frequencies is None:
        raise ValueError("No hay ventanas para calcular Welch")
    return frequencies, np.median(np.asarray(spectra), axis=0)


def lookup_expected(sample_path: Path, samples_root: Path) -> dict[str, str] | None:
    """Consulta la etiqueta después de inferir y solo para muestras de la biblioteca."""

    catalog_path = samples_root / "catalogo_privado.json"
    if not catalog_path.is_file():
        return None
    try:
        relative_path = sample_path.relative_to(samples_root).as_posix()
    except ValueError:
        return None
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    entry = next(
        (item for item in catalog["samples"] if item["relative_path"] == relative_path),
        None,
    )
    return None if entry is None else entry["expected"]


def display_name(value: object) -> str:
    if value is None:
        return "No concluyente"
    return DISPLAY_NAMES.get(str(value), str(value).replace("_", " ").title())


def create_aps_figure(
    window_sets: dict[str, np.ndarray],
    fs_hz: float,
    window_index: int,
    nperseg: int,
    analysis_window: str,
    output_path: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    colors = {"L": COLORES["azul_pastel"], "H": COLORES["rosa"]}
    line_styles = {"L": "-", "H": "--"}
    diagnostics: dict[str, dict[str, float]] = {}
    median_psd: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10), constrained_layout=True)
    for column, part in enumerate(("L", "H")):
        windows = np.asarray(window_sets[part])
        if not 0 <= window_index < len(windows):
            raise IndexError(f"window-index fuera de rango para {part}: {window_index}")
        x = windows[window_index]
        time_us = np.arange(len(x)) / fs_hz * 1e6

        axes[0, column].plot(time_us, x, color=colors[part], lw=1.15)
        axes[0, column].set_title(f"Parte {part}: ventana temporal {window_index}")
        axes[0, column].set_xlabel("Tiempo [µs]")
        axes[0, column].set_ylabel("Amplitud normalizada")

        fft_f, fft_mag, _ = fft_magnitude(
            x,
            fs_hz,
            n_fft=8192,
            window=analysis_window,
            detrend=True,
        )
        fft_db = 20.0 * np.log10(np.maximum(fft_mag, 1e-15))
        fft_peak = int(np.argmax(fft_mag))
        peak_frequency = float(fft_f[fft_peak])
        axes[1, column].plot(
            fft_f / 1e6,
            fft_db,
            color=colors[part],
            ls=line_styles[part],
            lw=1.1,
        )
        axes[1, column].axvline(peak_frequency / 1e6, color=COLORES["violeta"], lw=0.9)
        axes[1, column].set_title(f"Parte {part}: FFT con Hann y zero padding")
        axes[1, column].set_xlabel("Frecuencia [MHz]")
        axes[1, column].set_ylabel("Magnitud [dB rel.] ")
        axes[1, column].set_xlim(0.0, fs_hz / 2e6)
        inset_fft = axes[1, column].inset_axes([0.53, 0.53, 0.43, 0.4])
        inset_fft.plot(fft_f / 1e6, fft_db, color=colors[part], lw=1.0)
        inset_fft.set_xlim(
            max(0.0, peak_frequency / 1e6 - 0.6),
            min(fs_hz / 2e6, peak_frequency / 1e6 + 0.6),
        )
        inset_fft.set_title(f"Pico: {peak_frequency / 1e6:.3f} MHz", fontsize=8)
        inset_fft.tick_params(labelsize=7)
        inset_fft.grid(True, alpha=0.35)

        welch_f, psd = median_welch(windows, fs_hz, nperseg, analysis_window)
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-30))
        psd_peak = int(np.argmax(psd))
        psd_peak_frequency = float(welch_f[psd_peak])
        diagnostics[part] = relative_psd_diagnostics(psd)
        diagnostics[part]["frecuencia_pico_fft_hz"] = peak_frequency
        diagnostics[part]["frecuencia_pico_welch_hz"] = psd_peak_frequency
        median_psd[part] = (welch_f, psd)

        axes[2, column].plot(welch_f / 1e6, psd_db, color=colors[part], lw=1.15)
        axes[2, column].axvline(psd_peak_frequency / 1e6, color=COLORES["violeta"], lw=0.9)
        contrast = diagnostics[part]["contraste_pico_piso_db"]
        axes[2, column].set_title(f"Parte {part}: PSD mediana, contraste {contrast:.1f} dB")
        axes[2, column].set_xlabel("Frecuencia [MHz]")
        axes[2, column].set_ylabel("PSD [dB rel./Hz]")
        axes[2, column].set_xlim(0.0, fs_hz / 2e6)
        inset_psd = axes[2, column].inset_axes([0.53, 0.53, 0.43, 0.4])
        inset_psd.plot(welch_f / 1e6, psd_db, color=colors[part], lw=1.0)
        inset_psd.set_xlim(
            max(0.0, psd_peak_frequency / 1e6 - 0.6),
            min(fs_hz / 2e6, psd_peak_frequency / 1e6 + 0.6),
        )
        inset_psd.set_title(f"Pico: {psd_peak_frequency / 1e6:.3f} MHz", fontsize=8)
        inset_psd.tick_params(labelsize=7)
        inset_psd.grid(True, alpha=0.35)

    fig.suptitle(
        "Caso excluido: análisis temporal y espectral\n"
        f"fs={fs_hz / 1e6:.1f} MHz · N=1024 · Hann · NFFT=8192 · Welch nperseg={nperseg}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return diagnostics, median_psd


def create_decision_figure(
    features: pd.DataFrame,
    result: dict[str, object],
    output_path: Path,
) -> None:
    colors = {"L": COLORES["azul_pastel"], "H": COLORES["rosa"]}
    band_columns = [
        column for column in features if column.startswith("potencia_relativa_banda_")
    ]
    band_columns.sort(key=lambda value: int(value.rsplit("_", 1)[1]))
    by_part = features.groupby("part")[band_columns].median()
    x = np.arange(len(band_columns))

    fig, (axis_bands, axis_text) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.2),
        gridspec_kw={"width_ratios": [1.8, 1.0]},
        constrained_layout=True,
    )
    width = 0.38
    for offset, part in zip((-width / 2, width / 2), ("L", "H"), strict=True):
        axis_bands.bar(
            x + offset,
            100.0 * by_part.loc[part].to_numpy(dtype=float),
            width,
            color=colors[part],
            edgecolor=COLORES["gris_texto"],
            linewidth=0.45,
            label=f"Parte {part}",
        )
    axis_bands.set_title("Potencia relativa mediana en 20 bandas")
    axis_bands.set_xlabel("Banda de 1 MHz")
    axis_bands.set_ylabel("Potencia relativa [%]")
    axis_bands.set_xticks(x, [f"{index}-{index + 1}" for index in range(20)], rotation=55)
    axis_bands.legend()

    axis_text.axis("off")
    state = display_name(result["state"])
    model = display_name(result.get("model"))
    mode = display_name(result.get("mode"))
    lines = [
        "Resultado jerárquico",
        "",
        f"Actividad: {state}",
        f"Modelo: {model}",
        f"Modo: {mode}",
        "",
        f"Puntaje de dron: {float(result['drone_score']):.3f}",
        f"Umbral: {float(result['threshold']):.3f}",
        f"Margen: {float(result['margin']):.3f}",
        "",
        "La etiqueta esperada no se consulta",
        "durante esta inferencia.",
        "",
        "Las amplitudes y la PSD son relativas.",
    ]
    if result.get("model") is None:
        lines.append("Modelo y modo: no concluyentes con la evidencia reservada.")
    else:
        lines.append("Phantom tiene un único modo en DroneRF.")
    axis_text.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=11.5, linespacing=1.35)
    fig.suptitle("Caso excluido: bandas e interpretación de la decisión", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    aplicar_estilo_matplotlib()
    run_root = PROJECT_ROOT / "resultados" / "runs" / args.run_id
    sample_path = resolve_project_path(args.sample)
    conservative_bundle = (
        PROJECT_ROOT
        / "resultados"
        / "runs"
        / "actual"
        / "modelos"
        / "bundle_demo_conservador.joblib"
    )
    if args.bundle is not None:
        bundle_path = resolve_project_path(args.bundle)
    elif conservative_bundle.is_file():
        bundle_path = conservative_bundle
    else:
        bundle_path = run_root / "modelos" / "bundle.joblib"

    bundle = load_bundle(bundle_path)
    signal_input = load_signal_input(sample_path)
    features = extract_input_features(signal_input, bundle)
    result = predict_hierarchy(bundle, features)

    samples_root = (PROJECT_ROOT / "muestras_demo").resolve()
    expected = lookup_expected(sample_path, samples_root)
    if signal_input.window_sets is None:
        raise ValueError("El caso de estudio requiere un paquete NPZ con ventanas L/H")

    feature_config = bundle.get("feature_config", {}).get("feature_config", {})
    nperseg = int(feature_config.get("welch_nperseg", 256))
    analysis_window = str(feature_config.get("analysis_window", "hann"))
    figures_dir = run_root / "figuras"
    tables_dir = run_root / "tablas"
    metrics_dir = run_root / "metricas"
    figure_aps = figures_dir / "07_caso_phantom_analisis_aps.png"
    figure_decision = figures_dir / "08_caso_phantom_decision.png"

    diagnostics, _ = create_aps_figure(
        signal_input.window_sets,
        signal_input.fs_hz,
        args.window_index,
        nperseg,
        analysis_window,
        figure_aps,
    )
    create_decision_figure(features, result, figure_decision)

    summary_columns = [
        "rms",
        "energia",
        "potencia_media",
        "pico",
        "factor_cresta",
        "frecuencia_dominante_fft_hz",
        "frecuencia_dominante_hz",
        "centroide_espectral_hz",
        "ancho_banda_espectral_hz",
    ]
    summary = features.groupby("part")[summary_columns].median().reset_index()
    for key in (
        "piso_psd_db_rel",
        "pico_psd_db_rel",
        "contraste_pico_piso_db",
        "frecuencia_pico_fft_hz",
        "frecuencia_pico_welch_hz",
    ):
        by_part = {part: values[key] for part, values in diagnostics.items()}
        summary[key] = summary["part"].map(by_part)
    table_path = tables_dir / "caso_phantom_caracteristicas.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)

    payload: dict[str, object] = {
        "schema_version": "1.0",
        "generated_at_utc": utc_now(),
        "method": "Inferencia sin etiqueta seguida de consulta separada del catálogo.",
        "sample": describe_file(sample_path, PROJECT_ROOT),
        "bundle": describe_file(bundle_path, PROJECT_ROOT),
        "window_index": int(args.window_index),
        "fs_hz": float(signal_input.fs_hz),
        "inference_before_label": result,
        "expected_after_inference": expected,
        "matches_expected": (
            None
            if expected is None
            else {
                "activity": result["state"] == expected["activity"],
                "model": (
                    None
                    if result.get("model") is None
                    else result.get("model") == expected["model"]
                ),
                "mode": (
                    None
                    if result.get("mode") is None
                    else result.get("mode") == expected["mode"]
                ),
            }
        ),
        "diagnostics_by_part": diagnostics,
    }
    metrics_path = metrics_dir / "caso_phantom_conectado.json"
    write_json(metrics_path, payload)

    manifest_path = run_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        artifacts = manifest.setdefault("artifacts", {})
        for key, path in {
            "case_study_aps_figure": figure_aps,
            "case_study_decision_figure": figure_decision,
            "case_study_table": table_path,
            "case_study_metrics": metrics_path,
        }.items():
            artifacts[key] = describe_file(path, PROJECT_ROOT)
        manifest["case_study"] = {
            "sample_sha256": sha256_file(sample_path),
            "label_accessed_after_inference": True,
            "result": {
                "state": result["state"],
                "model": result.get("model"),
                "mode": result.get("mode"),
            },
        }
        write_json(manifest_path, manifest)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
