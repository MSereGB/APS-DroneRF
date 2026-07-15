"""Gráficos compactos para auditorías de robustez del detector binario."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aps_drone_rf.estilo import COLORES, aplicar_estilo_matplotlib


def normalizar_conjuntos_ventanas(
    window_sets: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Remueve media y normaliza cada parte para una perturbación controlada.

    Los paquetes de la demostración ya están preprocesados. Esta función permite
    aplicar una perturbación sobre sus ventanas y volver a pasar por el mismo paso
    básico de media cero y normalización relativa, sin acceder a etiquetas.
    """

    output: dict[str, np.ndarray] = {}
    for part, windows in window_sets.items():
        values = np.asarray(windows, dtype=float)
        if values.ndim != 2 or values.size == 0:
            raise ValueError("Cada parte debe contener una matriz no vacía de ventanas")
        flattened = values.reshape(-1)
        centered = flattened - float(np.mean(flattened))
        scale = max(float(np.max(np.abs(centered))), np.finfo(float).eps)
        output[str(part)] = (centered / scale).reshape(values.shape)
    return output


def agregar_tono_controlado(
    window_sets: dict[str, np.ndarray],
    *,
    fs_hz: float,
    frequency_hz: float,
    sir_db: float,
) -> dict[str, np.ndarray]:
    """Suma un tono con relación señal/interferencia relativa conocida.

    ``sir_db`` compara el RMS de cada parte con el RMS del tono agregado. No es una
    medición física de interferencia: sirve solamente como estresor sintético.
    """

    if fs_hz <= 0 or frequency_hz < 0 or frequency_hz > fs_hz / 2:
        raise ValueError("La frecuencia del tono debe pertenecer al semiespectro válido")
    output: dict[str, np.ndarray] = {}
    for part_index, (part, windows) in enumerate(sorted(window_sets.items())):
        values = np.asarray(windows, dtype=float)
        if values.ndim != 2 or values.size == 0:
            raise ValueError("Cada parte debe contener una matriz no vacía de ventanas")
        flattened = values.reshape(-1)
        rms_signal = float(np.sqrt(np.mean(flattened**2)))
        rms_tone = rms_signal / (10.0 ** (sir_db / 20.0))
        tt = np.arange(flattened.size, dtype=float) / fs_hz
        phase = part_index * np.pi / 7.0
        tone = np.sin(2.0 * np.pi * frequency_hz * tt + phase)
        observed_rms = float(np.sqrt(np.mean(tone**2)))
        if observed_rms > np.finfo(float).eps:
            tone *= rms_tone / observed_rms
        else:
            tone.fill(0.0)
        output[str(part)] = (flattened + tone).reshape(values.shape)
    return output


def agregar_impulsos_controlados(
    window_sets: dict[str, np.ndarray],
    *,
    fraction: float,
    amplitude_rms: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Agrega impulsos bipolares reproducibles para estudiar sensibilidad.

    ``fraction`` es la porción de muestras afectadas por parte y ``amplitude_rms``
    expresa su amplitud respecto del RMS de la parte original.
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction debe quedar entre cero y uno")
    if amplitude_rms <= 0:
        raise ValueError("amplitude_rms debe ser positivo")
    rng = np.random.default_rng(seed)
    output: dict[str, np.ndarray] = {}
    for part, windows in sorted(window_sets.items()):
        values = np.asarray(windows, dtype=float)
        if values.ndim != 2 or values.size == 0:
            raise ValueError("Cada parte debe contener una matriz no vacía de ventanas")
        flattened = values.reshape(-1).copy()
        count = max(1, int(round(flattened.size * fraction)))
        positions = rng.choice(flattened.size, size=count, replace=False)
        rms_signal = float(np.sqrt(np.mean(flattened**2)))
        signs = rng.choice(np.array([-1.0, 1.0]), size=count)
        flattened[positions] += signs * amplitude_rms * rms_signal
        output[str(part)] = flattened.reshape(values.shape)
    return output


def plot_robustness_summary(
    nested: pd.DataFrame,
    permutation: pd.DataFrame,
    noise_summary: pd.DataFrame,
    quantization: pd.DataFrame,
    chunks: pd.DataFrame,
    parts: pd.DataFrame,
):
    """Resume los cuatro controles cuantitativos principales en paneles legibles."""

    aplicar_estilo_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    axes[0, 0].plot(
        nested["repeat"],
        nested["balanced_accuracy"],
        marker="o",
        color=COLORES["rosa"],
    )
    axes[0, 0].set(
        title="CV anidada por grupos",
        xlabel="Repetición",
        ylabel="Exactitud balanceada",
        ylim=(0, 1.05),
    )

    axes[0, 1].scatter(
        permutation["permutation"],
        permutation["balanced_accuracy"],
        color=COLORES["lila"],
        label="Etiquetas permutadas",
    )
    axes[0, 1].axhline(0.5, color=COLORES["gris_texto"], linestyle="--", label="Azar binario")
    axes[0, 1].axhline(
        float(nested["balanced_accuracy"].mean()),
        color=COLORES["rosa"],
        linestyle="-.",
        label="Promedio con etiquetas reales",
    )
    axes[0, 1].set(
        title="Control con etiquetas permutadas",
        xlabel="Permutación",
        ylabel="Exactitud balanceada",
        ylim=(0, 1.05),
    )
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        noise_summary["snr_db"],
        noise_summary["balanced_accuracy_mean"],
        marker="o",
        color=COLORES["violeta"],
        label="Exactitud balanceada",
    )
    axes[1, 0].plot(
        noise_summary["snr_db"],
        noise_summary["false_alarm_rate_mean"],
        marker="s",
        color=COLORES["azul_pastel"],
        label="Falsas alarmas",
    )
    axes[1, 0].set(
        title="Ruido blanco relativo",
        xlabel="SNR controlada [dB]",
        ylabel="Valor por grupo",
        ylim=(0, 1.05),
    )
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        quantization["bits"],
        quantization["balanced_accuracy"],
        marker="o",
        color=COLORES["rosa"],
        label="Exactitud balanceada",
    )
    axes[1, 1].plot(
        quantization["bits"],
        quantization["false_alarm_rate"],
        marker="s",
        color=COLORES["azul_pastel"],
        label="Falsas alarmas",
    )
    axes[1, 1].set(
        title="Cuantización relativa",
        xlabel="Bits",
        ylabel="Valor por grupo",
        xticks=quantization["bits"],
        ylim=(0, 1.05),
    )
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Pruebas complementarias de estabilidad del detector binario", fontsize=13)
    return fig


def plot_perturbation_summary(summary: pd.DataFrame):
    """Grafica la decisión binaria ante estresores de entrada controlados."""

    required = {
        "condicion",
        "exactitud_balanceada",
        "tasa_deteccion",
        "tasa_falsa_alarma",
        "tasa_no_concluyente",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Faltan columnas para el gráfico: {sorted(missing)}")

    aplicar_estilo_matplotlib()
    positions = np.arange(len(summary))
    labels = summary["condicion"].str.replace("_", " ", regex=False)
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), sharex=True, constrained_layout=True)

    axes[0].plot(
        positions,
        summary["exactitud_balanceada"],
        marker="o",
        linewidth=2,
        color=COLORES["rosa"],
        label="Exactitud balanceada",
    )
    axes[0].plot(
        positions,
        summary["tasa_deteccion"],
        marker="s",
        linewidth=2,
        color=COLORES["violeta"],
        label="Tasa de detección",
    )
    axes[0].plot(
        positions,
        summary["tasa_falsa_alarma"],
        marker="^",
        linewidth=2,
        color=COLORES["azul_pastel"],
        label="Tasa de falsas alarmas",
    )
    axes[0].set(
        title="Respuesta del detector ante perturbaciones de entrada",
        ylabel="Valor por grupo",
        ylim=(-0.03, 1.05),
    )
    axes[0].legend(ncol=3, fontsize=8, loc="lower left")

    axes[1].bar(
        positions,
        summary["tasa_no_concluyente"],
        color=COLORES["lila"],
        width=0.62,
        label="No concluyente",
    )
    axes[1].set(
        xlabel="Condición sintética",
        ylabel="Tasa",
        ylim=(-0.03, 1.05),
        xticks=positions,
        xticklabels=labels,
    )
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].legend(fontsize=8, loc="upper left")
    return fig
