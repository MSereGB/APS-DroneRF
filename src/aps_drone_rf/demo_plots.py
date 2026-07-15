"""Figuras compactas para la aplicación de demostración."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aps_drone_rf.demo import SignalInput
from aps_drone_rf.dronerf import DISPLAY_NAMES
from aps_drone_rf.estilo import COLORES, aplicar_estilo_matplotlib
from aps_drone_rf.preprocessing import preprocess_signal
from aps_drone_rf.spectral import fft_magnitude, welch_psd

SERIES_COLORS = [
    COLORES["azul_pastel"],
    COLORES["rosa"],
    COLORES["lila"],
    COLORES["violeta"],
]


def _new_figure(figsize=(8, 3.2)):
    aplicar_estilo_matplotlib()
    return plt.subplots(figsize=figsize, constrained_layout=True)


def temporal_figure(signal_input: SignalInput, max_samples: int = 5000):
    fig, ax = _new_figure()
    for index, (part, raw_signal) in enumerate(signal_input.signals.items()):
        signal = preprocess_signal(raw_signal)
        visible = signal[:max_samples]
        time_us = np.arange(len(visible)) / signal_input.fs_hz * 1e6
        ax.plot(time_us, visible, color=SERIES_COLORS[index], linewidth=1.0, label=part)
    ax.set(title="Señal temporal", xlabel="Tiempo [µs]", ylabel="Amplitud normalizada")
    ax.legend(title="Parte")
    return fig


def fft_figure(signal_input: SignalInput, max_samples: int = 65_536):
    fig, ax = _new_figure()
    for index, (part, raw_signal) in enumerate(signal_input.signals.items()):
        signal = preprocess_signal(raw_signal)[:max_samples]
        frequencies, magnitude, _ = fft_magnitude(
            signal,
            fs_hz=signal_input.fs_hz,
            window="hann",
        )
        ax.plot(
            frequencies / 1e6,
            magnitude,
            color=SERIES_COLORS[index],
            linewidth=1.0,
            label=part,
        )
    ax.set(title="FFT de magnitud", xlabel="Frecuencia [MHz]", ylabel="Magnitud relativa")
    ax.legend(title="Parte")
    return fig


def welch_figure(signal_input: SignalInput, max_samples: int = 65_536):
    fig, ax = _new_figure()
    for index, (part, raw_signal) in enumerate(signal_input.signals.items()):
        signal = preprocess_signal(raw_signal)[:max_samples]
        frequencies, psd = welch_psd(
            signal,
            fs_hz=signal_input.fs_hz,
            nperseg=min(2048, len(signal)),
            window="hann",
        )
        psd_db = 10.0 * np.log10(np.maximum(psd, np.finfo(float).tiny))
        ax.plot(
            frequencies / 1e6,
            psd_db,
            color=SERIES_COLORS[index],
            linewidth=1.0,
            label=part,
        )
    ax.set(title="PSD estimada por Welch", xlabel="Frecuencia [MHz]", ylabel="PSD [dB rel.] ")
    ax.legend(title="Parte")
    return fig


def band_figure(features: pd.DataFrame, band_count: int):
    columns = [f"potencia_relativa_banda_{index}" for index in range(1, band_count + 1)]
    values = features[columns].median().to_numpy(dtype=float)
    fig, ax = _new_figure()
    positions = np.arange(1, band_count + 1)
    colors = [SERIES_COLORS[index % len(SERIES_COLORS)] for index in range(band_count)]
    ax.bar(positions, values, color=colors, edgecolor="white")
    ax.set(
        title="Potencia relativa por bandas",
        xlabel="Banda",
        ylabel="Fracción de potencia",
        xticks=positions,
    )
    ax.set_ylim(0.0, max(float(values.max()) * 1.15, 0.05))
    return fig


def evolution_figure(values: list[float], threshold: float):
    fig, ax = _new_figure()
    if values:
        ax.plot(
            np.arange(1, len(values) + 1),
            values,
            color=COLORES["rosa"],
            linewidth=2.0,
            label="Puntaje de actividad",
        )
    ax.axhline(
        threshold,
        color=COLORES["violeta"],
        linestyle="--",
        linewidth=1.5,
        label="Umbral",
    )
    ax.set(
        title="Evolución del puntaje",
        xlabel="Actualización",
        ylabel="Puntaje relativo",
        ylim=(0.0, 1.0),
    )
    ax.legend()
    return fig


def feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    visible = {
        "RMS": "rms",
        "Energía": "energia",
        "Potencia media": "potencia_media",
        "Factor de cresta": "factor_cresta",
        "Frecuencia dominante [MHz]": "frecuencia_dominante_fft_hz",
        "Centroide [MHz]": "centroide_espectral_hz",
        "Ancho de banda [MHz]": "ancho_banda_espectral_hz",
    }
    rows = []
    for name, column in visible.items():
        value = float(features[column].median())
        if "MHz" in name:
            value /= 1e6
        if "MHz" in name:
            display_value = f"{value:.3f}"
        elif name in {"Energía", "Factor de cresta"}:
            display_value = f"{value:.3f}"
        else:
            display_value = f"{value:.5f}"
        rows.append({"Característica": name, "Valor mediano": display_value})
    return pd.DataFrame(rows)


def result_markdown(result: dict[str, object]) -> str:
    state_names = {
        "fondo": "Sin dron",
        "dron": "Actividad de dron",
        "no_concluyente": "No concluyente",
    }
    model = result.get("model")
    mode = result.get("mode")
    lines = [f"## {state_names[str(result['state'])]}"]
    lines.append(
        f"**Puntaje relativo:** {float(result['drone_score']):.3f}  |  "
        f"**Umbral:** {float(result['threshold']):.3f}  |  "
        f"**Margen:** {float(result['margin']):.3f}"
    )
    if "domain_distance" in result:
        lines.append(
            f"**Compatibilidad con la calibración:** "
            f"{float(result['domain_distance']):.3f} / "
            f"{float(result['domain_threshold']):.3f}"
        )
    if "quality_compatible" in result:
        quality = "compatible" if bool(result["quality_compatible"]) else "fuera de rango"
        lines.append(
            f"**Control de transitorios:** {quality} "
            f"({float(result['crest_maximum']):.3f} / "
            f"{float(result['crest_upper_limit']):.3f} en factor de cresta)"
        )
    if model:
        lines.append(f"**Modelo propuesto:** {DISPLAY_NAMES.get(str(model), model)}")
    if mode:
        lines.append(f"**Modo propuesto:** {DISPLAY_NAMES.get(str(mode), mode)}")
    if result["state"] == "dron" and model is None:
        lines.append("**Modelo y modo:** no concluyentes con la evidencia reservada disponible.")
    return "\n\n".join(lines)


def interpretation_text(result: dict[str, object]) -> str:
    stopped_reason = result.get("stopped_reason")
    if stopped_reason:
        return str(stopped_reason)
    if result["state"] == "no_concluyente":
        return (
            "El puntaje quedó demasiado cerca del umbral calibrado, "
            "así que no se fuerza una decisión."
        )
    if result["state"] == "fondo":
        return (
            "Las características temporales y espectrales se parecen más a los grupos de fondo "
            "usados durante el desarrollo."
        )
    if result.get("model") is None:
        return (
            "Hay evidencia relativa de actividad de dron, pero los puntajes de modelo quedaron "
            "demasiado próximos entre sí."
        )
    if result.get("mode") is None:
        return (
            "La actividad y el modelo resultan compatibles con la calibración, pero el modo no "
            "alcanza un margen suficiente."
        )
    return (
        "La decisión combina características de tiempo, FFT y PSD de las partes disponibles. "
        "Es una comparación relativa con DroneRF, no una medición de potencia física."
    )
