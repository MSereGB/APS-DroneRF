"""Exporta figuras finales de la corrida jerárquica DroneRF v2."""

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

from aps_drone_rf.estilo import COLORES, aplicar_estilo_matplotlib
from aps_drone_rf.provenance import sha256_file, utc_now, write_json

DEFAULT_RUN = (
    PROJECT_ROOT
    / "resultados"
    / "runs"
    / "dronerf_demo_v2_final_n1024_hann_b20_seed42"
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def plot_dataset_split(manifest: dict[str, object], output_dir: Path) -> pd.DataFrame:
    files = pd.DataFrame(manifest["files"])
    groups = files.drop_duplicates("group_id")
    order = ["desarrollo", "evaluacion", "demo"]
    names = ["Desarrollo", "Evaluación", "Demo excluida"]
    counts = groups["partition"].value_counts().reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    bars = ax.bar(
        names,
        counts.to_numpy(),
        color=[COLORES["azul_pastel"], COLORES["lila"], COLORES["rosa"]],
        width=0.62,
    )
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.set(
        title="Grupos originales por partición",
        xlabel="Partición congelada antes de calibrar",
        ylabel="Cantidad de grupos",
        ylim=(0, max(counts) * 1.18),
    )
    _save(fig, output_dir / "01_particiones_dataset.png")
    return pd.DataFrame({"particion": names, "grupos": counts.to_numpy()})


def plot_band_comparison(metrics: dict[str, object], output_dir: Path) -> pd.DataFrame:
    table = pd.DataFrame(metrics["band_comparison"])
    positions = np.arange(len(table))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
    score = ax.bar(
        positions - width / 2,
        table["balanced_accuracy_mean"],
        width,
        color=COLORES["rosa"],
        label="Exactitud balanceada media",
    )
    spread = ax.bar(
        positions + width / 2,
        table["fold_dispersion_mean"],
        width,
        color=COLORES["azul_pastel"],
        label="Dispersión media entre folds",
    )
    ax.bar_label(score, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(spread, fmt="%.3f", padding=3, fontsize=9)
    ax.set(
        title="Comparación de resolución de bandas",
        xlabel="Cantidad de bandas entre 0 y 20 MHz",
        ylabel="Valor relativo",
        xticks=positions,
        xticklabels=table["band_count"].astype(str),
        ylim=(0, 0.72),
    )
    ax.legend(loc="upper right")
    ax.text(
        0.5,
        -0.24,
        "Se eligieron 10 bandas usando solamente los grupos de desarrollo.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    _save(fig, output_dir / "02_comparacion_bandas.png")
    return table[
        ["band_count", "balanced_accuracy_mean", "fold_dispersion_mean"]
    ].copy()


def plot_hierarchy(metrics: dict[str, object], output_dir: Path) -> pd.DataFrame:
    keys = ["actividad", "modelo", "modo_bebop", "modo_ar"]
    names = ["Fondo / dron", "Modelo", "Modo Bebop", "Modo AR"]
    dummy = [metrics["dummy_oof"][key]["balanced_accuracy"] for key in keys]
    development = [
        metrics["oof_development"][key]["balanced_accuracy"] for key in keys
    ]
    evaluation = [
        metrics["reserved_evaluation"][key]["balanced_accuracy"] for key in keys
    ]
    positions = np.arange(len(keys))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.0, 4.7), constrained_layout=True)
    ax.bar(
        positions - width,
        dummy,
        width,
        color=COLORES["azul_pastel"],
        label="Referencia",
    )
    ax.bar(
        positions,
        development,
        width,
        color=COLORES["lila"],
        label="Desarrollo",
    )
    ax.bar(
        positions + width,
        evaluation,
        width,
        color=COLORES["rosa"],
        label="Evaluación reservada",
    )
    ax.set(
        title="Separabilidad por nivel de la jerarquía",
        xlabel="Nivel analizado",
        ylabel="Exactitud balanceada por grupo",
        xticks=positions,
        xticklabels=names,
        ylim=(0, 1.08),
    )
    ax.legend(ncol=3, loc="upper center")
    ax.text(
        0.5,
        -0.23,
        "La separación fondo/dron es estable; modelo y modo no generalizan igual.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    _save(fig, output_dir / "03_resultados_jerarquia.png")
    return pd.DataFrame(
        {
            "nivel": names,
            "dummy": dummy,
            "desarrollo_oof": development,
            "evaluacion_reservada": evaluation,
        }
    )


def plot_demo(metrics: dict[str, object], output_dir: Path) -> pd.DataFrame:
    keys = ["activity", "model", "mode"]
    names = ["Fondo / dron", "Modelo", "Modo"]
    coverage = [metrics[key]["coverage"] for key in keys]
    accuracy = [
        metrics[key]["balanced_accuracy"] if current_coverage > 0 else np.nan
        for key, current_coverage in zip(keys, coverage, strict=True)
    ]
    positions = np.arange(len(keys))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.8, 4.5), constrained_layout=True)
    ax.bar(
        positions - width / 2,
        accuracy,
        width,
        color=COLORES["rosa"],
        label="Exactitud balanceada",
    )
    second = ax.bar(
        positions + width / 2,
        coverage,
        width,
        color=COLORES["lila"],
        label="Cobertura",
    )
    for position, value in enumerate(accuracy):
        if np.isfinite(value):
            ax.text(
                position - width / 2,
                value + 0.035,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
            ax.text(
                position - width / 2,
                0.06,
                "No concluyente",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#4c5568",
                rotation=90,
            )
    ax.bar_label(second, fmt="%.3f", padding=3, fontsize=9)
    ax.set(
        title="Muestras excluidas del entrenamiento",
        xlabel="Nivel de la decisión",
        ylabel="Valor por grupo",
        xticks=positions,
        xticklabels=names,
        ylim=(0, 1.08),
    )
    ax.legend(loc="upper right")
    ax.text(
        0.5,
        -0.23,
        "Modelo y modo quedan sin decisión por la política conservadora.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    _save(fig, output_dir / "04_resultados_muestras_excluidas.png")
    return pd.DataFrame(
        {"nivel": names, "exactitud_balanceada": accuracy, "cobertura": coverage}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--demo-metrics",
        type=Path,
        default=None,
        help="JSON de muestras excluidas; por defecto usa la política conservadora final.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "dronerf_demo_v2_manifest.json",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = run_dir / "figuras"
    table_dir = run_dir / "tablas"
    table_dir.mkdir(parents=True, exist_ok=True)
    aplicar_estilo_matplotlib()

    hierarchy = _read_json(run_dir / "metricas" / "metricas_jerarquicas.json")
    demo_path = args.demo_metrics
    if demo_path is None:
        demo_path = run_dir / "metricas" / "metricas_demo_conservadora_final.json"
    demo = _read_json(demo_path.expanduser().resolve())
    manifest = _read_json(args.manifest.expanduser().resolve())

    tables = {
        "particiones_dataset.csv": plot_dataset_split(manifest, output_dir),
        "comparacion_bandas_final.csv": plot_band_comparison(hierarchy, output_dir),
        "resultados_jerarquia.csv": plot_hierarchy(hierarchy, output_dir),
        "resultados_muestras_excluidas.csv": plot_demo(demo, output_dir),
    }
    for filename, table in tables.items():
        table.to_csv(table_dir / filename, index=False)

    manifest_path = run_dir / "manifest.json"
    run_manifest = _read_json(manifest_path)
    for path in sorted(output_dir.glob("*.png")):
        run_manifest["artifacts"][f"figura_{path.stem}"] = _artifact(path)
    for filename in tables:
        path = table_dir / filename
        run_manifest["artifacts"][f"tabla_{path.stem}"] = _artifact(path)
    lock_path = PROJECT_ROOT / "requirements-lock.txt"
    if lock_path.is_file():
        run_manifest["artifacts"]["environment_lock"] = _artifact(lock_path)
    run_manifest["status"] = "figures_exported"
    run_manifest["updated_at_utc"] = utc_now()
    write_json(manifest_path, run_manifest)

    print(f"Figuras exportadas en: {output_dir}")


if __name__ == "__main__":
    main()
