"""Calibra la jerarquía fondo/dron, modelo y modo usando solamente desarrollo."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.calibration import (
    SCORE_PREFIX,
    STAGES,
    choose_binary_threshold,
    choose_rejection_margin,
    fit_final_model,
    grouped_oof_scores,
    score_with_model,
    summarize_group_scores,
    top_predictions,
)
from aps_drone_rf.deployment_policy import apply_conservative_policy
from aps_drone_rf.dominio import calibrate_domain_guard, calibrate_signal_quality_guard
from aps_drone_rf.metrics import classification_report_dict
from aps_drone_rf.pipeline import (
    classifier_feature_columns,
    features_for_band_count,
)
from aps_drone_rf.provenance import (
    git_commit,
    runtime_versions,
    sha256_file,
    utc_now,
    write_json,
)


def raw_stage_report(scores: pd.DataFrame, stage_name: str) -> dict[str, object]:
    """Mide separabilidad sin rechazo para comparar configuraciones."""

    spec = STAGES[stage_name]
    truth = scores[spec.label_column].astype(str).to_numpy()
    if stage_name == "actividad":
        threshold = choose_binary_threshold(truth, scores[f"{SCORE_PREFIX}dron"])
        prediction = np.where(scores[f"{SCORE_PREFIX}dron"] >= threshold, "dron", "fondo")
    else:
        prediction, _ = top_predictions(scores)
        threshold = None
    report = classification_report_dict(truth, prediction)
    report["threshold"] = threshold
    return report


def fold_dispersion(scores: pd.DataFrame, stage_name: str) -> tuple[float, list[float]]:
    """Calcula dispersión descriptiva por fold sobre grupos."""

    spec = STAGES[stage_name]
    binary_threshold = None
    if stage_name == "actividad":
        binary_threshold = choose_binary_threshold(
            scores[spec.label_column], scores[f"{SCORE_PREFIX}dron"]
        )
    values = []
    for _, fold_scores in scores.groupby("fold"):
        truth = fold_scores[spec.label_column].astype(str).to_numpy()
        if stage_name == "actividad":
            prediction = np.where(
                fold_scores[f"{SCORE_PREFIX}dron"] >= binary_threshold,
                "dron",
                "fondo",
            )
        else:
            prediction, _ = top_predictions(fold_scores)
        values.append(float(classification_report_dict(truth, prediction)["balanced_accuracy"]))
    return float(np.std(values)), values


def calibrate_rejection(scores: pd.DataFrame, stage_name: str) -> tuple[float | None, float]:
    """Obtiene umbral y margen usando exclusivamente puntajes OOF."""

    spec = STAGES[stage_name]
    truth = scores[spec.label_column].astype(str).to_numpy()
    if stage_name == "actividad":
        drone_scores = scores[f"{SCORE_PREFIX}dron"].to_numpy(dtype=float)
        threshold = choose_binary_threshold(truth, drone_scores)
        predictions = np.where(drone_scores >= threshold, "dron", "fondo")
        margins = np.abs(drone_scores - threshold)
    else:
        threshold = None
        predictions, margins = top_predictions(scores)
    rejection = choose_rejection_margin(truth, predictions, margins)
    return threshold, rejection


def select_band_count(comparison: list[dict[str, object]]) -> int:
    """Aplica prioridad, tolerancia de 0.02, dispersión y simplicidad."""

    best_score = max(float(item["balanced_accuracy_mean"]) for item in comparison)
    tied = [
        item
        for item in comparison
        if best_score - float(item["balanced_accuracy_mean"]) <= 0.02
    ]
    selected = min(
        tied,
        key=lambda item: (float(item["fold_dispersion_mean"]), int(item["band_count"])),
    )
    return int(selected["band_count"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="dronerf_demo_v2_baseline_n512_hann_b20_seed42")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="No copia el bundle candidato a resultados/runs/actual.",
    )
    args = parser.parse_args()

    run_root = PROJECT_ROOT / "resultados" / "runs" / args.run_id
    feature_path = run_root / "features" / "features.csv"
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"No existen features reales en {feature_path}. "
            "Ejecutar construir_caracteristicas_demo.py"
        )
    features = pd.read_csv(feature_path)
    present_partitions = set(features["partition"].astype(str))
    if "demo" in present_partitions:
        raise ValueError("La calibración rechaza tablas que contengan la partición demo")
    if not {"desarrollo", "evaluacion"}.issubset(present_partitions):
        raise ValueError("Faltan desarrollo o evaluación en la tabla de features")
    development_source = features[features["partition"] == "desarrollo"].copy()
    evaluation_source = features[features["partition"] == "evaluacion"].copy()

    comparison = []
    oof_cache: dict[int, dict[str, pd.DataFrame]] = {}
    fold_cache: dict[int, dict[str, list[dict[str, object]]]] = {}
    for band_count in (4, 10, 20):
        transformed = features_for_band_count(development_source, band_count)
        feature_columns = classifier_feature_columns(band_count)
        stage_scores = {}
        stage_folds = {}
        accuracies = []
        dispersions = []
        for stage_name, spec in STAGES.items():
            scores, folds = grouped_oof_scores(
                transformed,
                feature_columns,
                spec,
                n_splits=args.n_splits,
                random_state=args.seed,
                model_kind="lineal",
            )
            report = raw_stage_report(scores, stage_name)
            dispersion, _ = fold_dispersion(scores, stage_name)
            stage_scores[stage_name] = scores
            stage_folds[stage_name] = folds
            accuracies.append(float(report["balanced_accuracy"]))
            dispersions.append(dispersion)
        oof_cache[band_count] = stage_scores
        fold_cache[band_count] = stage_folds
        comparison.append(
            {
                "band_count": band_count,
                "balanced_accuracy_mean": float(np.mean(accuracies)),
                "fold_dispersion_mean": float(np.mean(dispersions)),
                "stage_balanced_accuracies": accuracies,
            }
        )

    selected_bands = select_band_count(comparison)
    feature_columns = classifier_feature_columns(selected_bands)
    development = features_for_band_count(development_source, selected_bands)
    evaluation = features_for_band_count(evaluation_source, selected_bands)
    domain_guard = calibrate_domain_guard(development, feature_columns)
    signal_quality_guard = calibrate_signal_quality_guard(development)
    models = {}
    thresholds: dict[str, dict[str, float | None]] = {}
    oof_reports = {}
    dummy_reports = {}
    evaluation_reports = {}
    prediction_tables = []
    stage_reliability = {}

    for stage_name, spec in STAGES.items():
        oof_scores = oof_cache[selected_bands][stage_name]
        binary_threshold, rejection_margin = calibrate_rejection(oof_scores, stage_name)
        thresholds[stage_name] = {
            "decision": binary_threshold,
            "rejection_margin": rejection_margin,
        }
        oof_report, oof_predictions = summarize_group_scores(
            oof_scores,
            spec,
            binary_threshold=binary_threshold,
            rejection_margin=rejection_margin,
        )
        dispersion, fold_values = fold_dispersion(oof_scores, stage_name)
        oof_report["fold_balanced_accuracy"] = fold_values
        oof_report["fold_dispersion"] = dispersion
        oof_reports[stage_name] = oof_report

        dummy_scores, _ = grouped_oof_scores(
            development,
            feature_columns,
            spec,
            n_splits=args.n_splits,
            random_state=args.seed,
            model_kind="dummy",
        )
        dummy_reports[stage_name] = raw_stage_report(dummy_scores, stage_name)
        dummy_balanced_accuracy = float(dummy_reports[stage_name]["balanced_accuracy"])
        improved_folds = sum(
            value > dummy_balanced_accuracy for value in oof_report["fold_balanced_accuracy"]
        )
        stage_reliability[stage_name] = {
            "enabled": bool(
                oof_report["balanced_accuracy"] > dummy_balanced_accuracy
                and improved_folds >= args.n_splits - 1
            ),
            "rule": "Superar al Dummy en al menos 4 de 5 folds de desarrollo.",
            "improved_folds": int(improved_folds),
            "total_folds": int(args.n_splits),
            "dummy_balanced_accuracy": dummy_balanced_accuracy,
        }

        model = fit_final_model(
            development,
            feature_columns,
            spec,
            random_state=args.seed,
        )
        models[stage_name] = model
        evaluation_scores = score_with_model(model, evaluation, feature_columns, spec)
        evaluation_report, evaluation_predictions = summarize_group_scores(
            evaluation_scores,
            spec,
            binary_threshold=binary_threshold,
            rejection_margin=rejection_margin,
        )
        evaluation_reports[stage_name] = evaluation_report
        evaluation_predictions["stage"] = stage_name
        evaluation_predictions["source"] = "evaluacion_reservada"
        prediction_tables.append(evaluation_predictions)

        oof_predictions["stage"] = stage_name
        oof_predictions["source"] = "desarrollo_oof"
        prediction_tables.append(oof_predictions)

    bundle = {
        "schema_version": "1.1",
        "run_id": args.run_id,
        "source": "DroneRF",
        "training_partition": "desarrollo",
        "evaluation_partition": "evaluacion",
        "blind_partition": "demo",
        "created_at_utc": utc_now(),
        "git_commit": git_commit(PROJECT_ROOT),
        "band_count": selected_bands,
        "feature_columns": feature_columns,
        "feature_config": json.loads((run_root / "config.json").read_text(encoding="utf-8-sig")),
        "thresholds": thresholds,
        "stage_reliability": stage_reliability,
        "domain_guard": domain_guard,
        "signal_quality_guard": signal_quality_guard,
        "models": models,
        "runtime_versions": runtime_versions(
            ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "gradio")
        ),
    }
    models_dir = run_root / "modelos"
    metrics_dir = run_root / "metricas"
    tables_dir = run_root / "tablas"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = models_dir / "bundle.joblib"
    joblib.dump(bundle, bundle_path)

    metrics = {
        "schema_version": "1.1",
        "run_id": args.run_id,
        "interpretation": "Separabilidad exploratoria por grupos; no detector operacional.",
        "selection_used_only": "desarrollo",
        "selected_band_count": selected_bands,
        "band_comparison": comparison,
        "oof_development": oof_reports,
        "dummy_oof": dummy_reports,
        "reserved_evaluation": evaluation_reports,
        "thresholds": thresholds,
        "stage_reliability": stage_reliability,
        "domain_guard": domain_guard,
        "signal_quality_guard": signal_quality_guard,
    }
    bundle = apply_conservative_policy(bundle, metrics)
    metrics["stage_reliability"] = bundle["stage_reliability"]
    metrics["deployment_policy"] = bundle["deployment_policy"]
    joblib.dump(bundle, bundle_path)
    metrics_path = metrics_dir / "metricas_jerarquicas.json"
    write_json(metrics_path, metrics)
    comparison_path = tables_dir / "comparacion_bandas.csv"
    pd.DataFrame(comparison).drop(columns=["stage_balanced_accuracies"]).to_csv(
        comparison_path, index=False
    )
    predictions_path = tables_dir / "predicciones_por_grupo.csv"
    pd.concat(prediction_tables, ignore_index=True).to_csv(predictions_path, index=False)

    if not args.no_promote:
        actual_root = PROJECT_ROOT / "resultados" / "runs" / "actual"
        actual_models = actual_root / "modelos"
        actual_models.mkdir(parents=True, exist_ok=True)
        actual_bundle = actual_models / "bundle.joblib"
        shutil.copy2(bundle_path, actual_bundle)
        write_json(
            actual_root / "origen.json",
            {
                "run_id": args.run_id,
                "bundle_sha256": sha256_file(bundle_path),
                "copied_at_utc": utc_now(),
            },
        )

    manifest_path = run_root / "manifest.json"
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    run_manifest["status"] = "calibrated_and_evaluated"
    run_manifest["calibration"] = {
        "development_groups": int(development["group_id"].nunique()),
        "evaluation_groups": int(evaluation["group_id"].nunique()),
        "demo_groups_accessed": 0,
        "n_splits": args.n_splits,
        "selected_band_count": selected_bands,
    }
    run_manifest["artifacts"].update(
        {
            "bundle": {
                "path": bundle_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": int(bundle_path.stat().st_size),
                "sha256": sha256_file(bundle_path),
            },
            "metrics": {
                "path": metrics_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": int(metrics_path.stat().st_size),
                "sha256": sha256_file(metrics_path),
            },
            "predictions": {
                "path": predictions_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": int(predictions_path.stat().st_size),
                "sha256": sha256_file(predictions_path),
            },
        }
    )
    write_json(manifest_path, run_manifest)
    print(f"Configuración elegida: {selected_bands} bandas")
    print(f"Bundle: {bundle_path}")
    print(f"Métricas: {metrics_path}")


if __name__ == "__main__":
    main()
