"""Funciones de graficacion con matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aps_drone_rf.estilo import (
    COLORES,
    aplicar_estilo_matplotlib,
    colormap_violeta,
    nombre_clase,
)


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    """Guarda una figura creando la carpeta destino."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    return output


def plot_time_signal(t, x, title: str = "Señal temporal"):
    """Grafica una señal temporal."""

    aplicar_estilo_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, np.real(x), linewidth=1.1, color=COLORES["violeta"])
    ax.set_title(title)
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Amplitud normalizada")
    return fig, ax


def plot_spectrum(freqs, magnitude, title: str = "Espectro de magnitud"):
    """Grafica magnitud espectral."""

    aplicar_estilo_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freqs, magnitude, linewidth=1.1, color=COLORES["rosa"])
    ax.set_title(title)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Magnitud")
    return fig, ax


def plot_psd(freqs, psd, title: str = "PSD"):
    """Grafica PSD en escala dB relativa."""

    aplicar_estilo_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(
        freqs,
        10.0 * np.log10(np.maximum(psd, 1e-20)),
        linewidth=1.1,
        color=COLORES["violeta"],
    )
    ax.set_title(title)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("PSD [dB rel.]")
    return fig, ax


def plot_confusion_matrix(matrix, labels, title: str = "Matriz de confusion"):
    """Grafica matriz de confusion simple."""

    aplicar_estilo_matplotlib()
    visible_labels = [nombre_clase(label) for label in labels]
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(matrix, cmap=colormap_violeta())
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels=visible_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=visible_labels)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i][j]
            color = "white" if value > np.max(matrix) * 0.55 else COLORES["gris_texto"]
            ax.text(j, i, str(value), ha="center", va="center", color=color)
    for spine in ax.spines.values():
        spine.set_color(COLORES["violeta"])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax
