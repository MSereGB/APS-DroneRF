"""Nombres simples en español y paleta visual del trabajo."""

from __future__ import annotations

from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

COLORES = {
    "rosa": "#E85D9E",
    "rosa_claro": "#F7B4D4",
    "lila": "#B084F5",
    "lila_claro": "#D7C2FF",
    "violeta": "#6F42C1",
    "violeta_oscuro": "#4C1D95",
    "azul_pastel": "#74BDE8",
    "azul_claro": "#B7E0FA",
    "gris_texto": "#2A2630",
    "gris_suave": "#E7DFEF",
    "fondo_pagina": "#FCF8FF",
}

COLOR_CLASE = {
    "fondo": COLORES["azul_pastel"],
    "dron": COLORES["rosa"],
    "background": COLORES["azul_pastel"],
    "drone": COLORES["rosa"],
}

NOMBRE_CLASE = {
    "fondo": "Fondo",
    "dron": "Dron",
    "background": "Fondo",
    "drone": "Dron",
    "desconocido": "Desconocido",
    "unknown": "Desconocido",
}

ETIQUETA_NORMALIZADA = {
    "background": "fondo",
    "no_drone": "fondo",
    "sin_dron": "fondo",
    "fondo": "fondo",
    "drone": "dron",
    "dron": "dron",
    "unknown": "desconocido",
    "desconocido": "desconocido",
}

COLOR_SERIE = {
    "periodograma": COLORES["lila"],
    "welch": COLORES["violeta"],
    "rectangular": COLORES["azul_pastel"],
    "hann": COLORES["rosa"],
    "hamming": COLORES["lila"],
    "blackman": COLORES["violeta"],
    "fir": COLORES["azul_pastel"],
    "iir": COLORES["rosa"],
    "antes": COLORES["lila"],
    "despues": COLORES["violeta"],
}

NOMBRE_METRICA = {
    "accuracy": "Exactitud",
    "balanced_accuracy": "Exactitud balanceada",
    "precision_macro": "Precision macro",
    "recall_macro": "Recall macro",
    "f1_macro": "F1 macro",
    "splitter": "Metodo de validacion",
    "n_splits": "Cantidad de folds",
    "n_groups": "Cantidad de grupos",
}


def etiqueta_clase(label: str) -> str:
    """Devuelve etiqueta interna simple: fondo, dron o desconocido."""

    return ETIQUETA_NORMALIZADA.get(str(label).strip().lower(), str(label).strip().lower())


def nombre_clase(label: str) -> str:
    """Devuelve nombre visible para figuras y tablas."""

    simple = etiqueta_clase(label)
    return NOMBRE_CLASE.get(simple, str(label))


def color_clase(label: str) -> str:
    """Devuelve color estable para una clase."""

    simple = etiqueta_clase(label)
    return COLOR_CLASE.get(simple, COLORES["violeta"])


def aplicar_estilo_matplotlib() -> None:
    """Aplica estilo comun a las figuras del trabajo."""

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORES["gris_texto"],
            "axes.labelcolor": COLORES["gris_texto"],
            "axes.titlecolor": COLORES["gris_texto"],
            "axes.grid": True,
            "axes.prop_cycle": cycler(
                color=[
                    COLORES["azul_pastel"],
                    COLORES["rosa"],
                    COLORES["lila"],
                    COLORES["violeta"],
                    COLORES["azul_claro"],
                    COLORES["rosa_claro"],
                ]
            ),
            "grid.color": COLORES["gris_suave"],
            "grid.alpha": 0.65,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": COLORES["gris_suave"],
            "xtick.color": COLORES["gris_texto"],
            "ytick.color": COLORES["gris_texto"],
        }
    )


def colormap_violeta() -> LinearSegmentedColormap:
    """Colormap suave para matrices de confusion."""

    return LinearSegmentedColormap.from_list(
        "rosa_lila_violeta",
        [COLORES["fondo_pagina"], COLORES["lila_claro"], COLORES["lila"], COLORES["violeta"]],
    )
