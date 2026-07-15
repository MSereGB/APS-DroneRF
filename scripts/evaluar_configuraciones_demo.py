"""Compara tamaños y ventanas usando solamente la partición de desarrollo."""

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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.calibration import (
    SCORE_PREFIX,
    STAGES,
    choose_binary_threshold,
    grouped_oof_scores,
    top_predictions,
)
from aps_drone_rf.estilo import COLORES, aplicar_estilo_matplotlib
from aps_drone_rf.io import load_signal_file
from aps_drone_rf.metrics import classification_report_dict
from aps_drone_rf.pipeline import (
    FeatureConfig,
    classifier_feature_columns,
    config_dict,
    extract_preprocessed_features,
    features_for_band_count,
    load_dataset_manifest,
    validate_materialized_files,
)
from aps_drone_rf.preprocessing import preprocess_signal
from aps_drone_rf.provenance import git_commit, sha256_file, utc_now, write_json

BASELINE_RUN = "dronerf_demo_v2_baseline_n512_hann_b20_seed42"
EXPERIMENT_ROOT = PROJECT_ROOT / "resultados" / "runs" / "sensibilidad_configuraciones"

CANDIDATES = {
    "n256_hann": FeatureConfig(window_size=256, hop_size=128, welch_nperseg=256),
    "n1024_hann": FeatureConfig(window_size=1024, hop_size=512, welch_nperseg=256),
    "n2048_hann": FeatureConfig(window_size=2048, hop_size=1024, welch_nperseg=256),
    "n512_hamming": FeatureConfig(analysis_window="hamming"),
    "n512_blackman": FeatureConfig(analysis_window="blackman"),
}


def _partial_name(row: dict[str, object]) -> str:
    return f"{row['code']}_{row['part']}_{int(row['segment']):02d}.csv"


def _stage_summary(scores: pd.DataFrame, stage_name: str) -> tuple[float, float, list[float]]:
    spec = STAGES[stage_name]
    truth = scores[spec.label_column].astype(str).to_numpy()
    threshold = None
    if stage_name == "actividad":
        threshold = choose_binary_threshold(truth, scores[f"{SCORE_PREFIX}dron"])
        prediction = np.where(
            scores[f"{SCORE_PREFIX}dron"].to_numpy(dtype=float) >= threshold,
            "dron",
            "fondo",
        )
    else:
        prediction, _ = top_predictions(scores)
    score = float(classification_report_dict(truth, prediction)["balanced_accuracy"])

    fold_scores = []
    for _, fold in scores.groupby("fold"):
        fold_truth = fold[spec.label_column].astype(str).to_numpy()
        if stage_name == "actividad":
            fold_prediction = np.where(
                fold[f"{SCORE_PREFIX}dron"].to_numpy(dtype=float) >= threshold,
                "dron",
                "fondo",
            )
        else:
            fold_prediction, _ = top_predictions(fold)
        fold_scores.append(
            float(
                classification_report_dict(fold_truth, fold_prediction)[
                    "balanced_accuracy"
                ]
            )
        )
    return score, float(np.std(fold_scores)), fold_scores


def _evaluate_configuration(
    features: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
) -> dict[str, object]:
    band_results = []
    for band_count in (4, 10, 20):
        transformed = features_for_band_count(features, band_count)
        feature_columns = classifier_feature_columns(band_count)
        stage_results = {}
        scores = []
        dispersions = []
        for stage_name, spec in STAGES.items():
            oof, _ = grouped_oof_scores(
                transformed,
                feature_columns,
                spec,
                n_splits=n_splits,
                random_state=seed,
                model_kind="lineal",
            )
            score, dispersion, fold_values = _stage_summary(oof, stage_name)
            stage_results[stage_name] = {
                "balanced_accuracy": score,
                "fold_dispersion": dispersion,
                "fold_values": fold_values,
            }
            scores.append(score)
            dispersions.append(dispersion)
        band_results.append(
            {
                "band_count": band_count,
                "balanced_accuracy_mean": float(np.mean(scores)),
                "fold_dispersion_mean": float(np.mean(dispersions)),
                "stages": stage_results,
            }
        )

    best_score = max(item["balanced_accuracy_mean"] for item in band_results)
    tied = [
        item
        for item in band_results
        if best_score - item["balanced_accuracy_mean"] <= 0.02
    ]
    selected = min(
        tied,
        key=lambda item: (item["fold_dispersion_mean"], item["band_count"]),
    )
    return {"band_comparison": band_results, "selected": selected}


def _combine_features(run_root: Path, rows: list[dict[str, object]]) -> Path:
    partial_dir = run_root / "features" / "por_archivo"
    paths = [partial_dir / _partial_name(row) for row in rows]
    features = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    features = features.sort_values(["code", "segment", "part", "start"])
    output = run_root / "features" / "features.csv"
    features.to_csv(output, index=False)
    return output


def _plot_summary(table: pd.DataFrame, output: Path) -> None:
    aplicar_estilo_matplotlib()
    order = [
        "n256_hann",
        "n512_hann",
        "n1024_hann",
        "n2048_hann",
        "n512_hamming",
        "n512_blackman",
    ]
    ordered = table.set_index("configuration").loc[order].reset_index()
    positions = np.arange(len(ordered))
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.6), sharex=True, constrained_layout=True)
    axes[0].plot(
        positions,
        ordered["balanced_accuracy_mean"],
        color=COLORES["rosa"],
        marker="o",
        linewidth=2,
    )
    for position, value in zip(positions, ordered["balanced_accuracy_mean"], strict=True):
        axes[0].annotate(f"{value:.3f}", (position, value), xytext=(0, 7),
                         textcoords="offset points", ha="center", fontsize=8)
    axes[0].set(
        title="Desempeño medio en desarrollo",
        ylabel="Exactitud balanceada",
        ylim=(0.45, 0.68),
    )
    bars = axes[1].bar(
        positions,
        ordered["fold_dispersion_mean"],
        color=COLORES["azul_pastel"],
        edgecolor=COLORES["gris_texto"],
        width=0.62,
    )
    axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[1].set(
        title="Dispersión entre particiones (menor es mejor)",
        xlabel="Configuración",
        ylabel="Desviación media",
        xticks=positions,
        xticklabels=ordered["display_name"],
        ylim=(0, max(ordered["fold_dispersion_mean"]) * 1.25),
    )
    fig.suptitle("Sensibilidad a longitud y ventana de análisis", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "dronerf_demo_v2_manifest.json",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()

    manifest = load_dataset_manifest(args.manifest)
    rows = validate_materialized_files(manifest, args.data_dir, {"desarrollo"})
    run_roots = {name: EXPERIMENT_ROOT / name for name in CANDIDATES}
    for name, config in CANDIDATES.items():
        config.validate()
        run_root = run_roots[name]
        (run_root / "features" / "por_archivo").mkdir(parents=True, exist_ok=True)
        expected = {
            "dataset": "DroneRF",
            "dataset_manifest_sha256": sha256_file(args.manifest),
            "partition": "desarrollo",
            "feature_config": config_dict(config),
            "selection_data": "desarrollo solamente",
        }
        config_path = run_root / "config.json"
        if config_path.exists():
            current = json.loads(config_path.read_text(encoding="utf-8-sig"))
            if current != expected:
                raise ValueError(f"Configuración incompatible en {config_path}")
        else:
            write_json(config_path, expected)

    for row in tqdm(rows, desc="CSV leídos una sola vez", unit="archivo"):
        pending = [
            name
            for name, run_root in run_roots.items()
            if not (run_root / "features" / "por_archivo" / _partial_name(row)).is_file()
        ]
        if not pending:
            continue
        if args.verify_hashes and sha256_file(row["absolute_path"]) != row["sha256"]:
            raise ValueError(f"Hash inesperado para {row['relative_path']}")
        signal = preprocess_signal(load_signal_file(row["absolute_path"]))
        for name in pending:
            table = extract_preprocessed_features(signal, row, CANDIDATES[name])
            if not np.isfinite(table.select_dtypes(include=[np.number])).all().all():
                raise ValueError(f"Features no finitas en {row['relative_path']}")
            output = run_roots[name] / "features" / "por_archivo" / _partial_name(row)
            table.to_csv(output, index=False)

    baseline_path = (
        PROJECT_ROOT / "resultados" / "runs" / BASELINE_RUN / "features" / "features.csv"
    )
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Falta la configuración base: {baseline_path}")
    feature_paths = {"n512_hann": baseline_path}
    for name, run_root in run_roots.items():
        feature_paths[name] = _combine_features(run_root, rows)

    labels = {
        "n256_hann": "N=256\nHann",
        "n512_hann": "N=512\nHann",
        "n1024_hann": "N=1024\nHann",
        "n2048_hann": "N=2048\nHann",
        "n512_hamming": "N=512\nHamming",
        "n512_blackman": "N=512\nBlackman",
    }
    summary_rows = []
    details = {}
    for name, path in feature_paths.items():
        features = pd.read_csv(path)
        features = features[features["partition"] == "desarrollo"].copy()
        result = _evaluate_configuration(
            features,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        details[name] = result
        selected = result["selected"]
        summary_rows.append(
            {
                "configuration": name,
                "display_name": labels[name],
                "selected_band_count": selected["band_count"],
                "balanced_accuracy_mean": selected["balanced_accuracy_mean"],
                "fold_dispersion_mean": selected["fold_dispersion_mean"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    best_score = float(summary["balanced_accuracy_mean"].max())
    tied = summary[
        best_score - summary["balanced_accuracy_mean"] <= 0.02
    ].copy()
    simplicity = {
        "n512_hann": 0,
        "n256_hann": 1,
        "n1024_hann": 2,
        "n2048_hann": 3,
        "n512_hamming": 4,
        "n512_blackman": 5,
    }
    tied["simplicity"] = tied["configuration"].map(simplicity)
    selected_row = tied.sort_values(
        ["fold_dispersion_mean", "simplicity"]
    ).iloc[0]

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = EXPERIMENT_ROOT / "resumen_configuraciones.csv"
    summary.to_csv(summary_path, index=False)
    figure_path = EXPERIMENT_ROOT / "sensibilidad_configuraciones.png"
    _plot_summary(summary, figure_path)
    report = {
        "schema_version": "1.0",
        "executed_at_utc": utc_now(),
        "git_commit": git_commit(PROJECT_ROOT),
        "selection_partition": "desarrollo",
        "evaluation_groups_accessed": 0,
        "demo_groups_accessed": 0,
        "rule": (
            "Mayor exactitud balanceada media; dentro de 0.02, menor dispersión; "
            "después, configuración más simple."
        ),
        "selected_configuration": str(selected_row["configuration"]),
        "selected_band_count": int(selected_row["selected_band_count"]),
        "summary": summary.to_dict(orient="records"),
        "details": details,
        "artifacts": {
            "summary": {
                "path": summary_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(summary_path),
            },
            "figure": {
                "path": figure_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(figure_path),
            },
        },
    }
    write_json(EXPERIMENT_ROOT / "resultado.json", report)
    print(summary.to_string(index=False))
    print(f"Configuración seleccionada con desarrollo: {report['selected_configuration']}")


if __name__ == "__main__":
    main()
