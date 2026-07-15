"""Somete el detector binario congelado a controles reproducibles de robustez."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.calibration import (
    SCORE_PREFIX,
    STAGES,
    choose_binary_threshold,
    grouped_oof_scores,
    nested_binary_group_scores,
    summarize_group_scores,
)
from aps_drone_rf.demo import (
    SignalInput,
    extract_input_features,
    load_bundle,
    load_signal_input,
    predict_hierarchy,
)
from aps_drone_rf.metrics import classification_report_dict
from aps_drone_rf.pipeline import classifier_feature_columns, features_for_band_count
from aps_drone_rf.provenance import sha256_file, utc_now, write_json
from aps_drone_rf.robustness import plot_robustness_summary
from aps_drone_rf.synthetic import quantize_uniform

DEFAULT_RUN_ID = "dronerf_demo_v2_final_n1024_hann_b20_seed42"


def activity_prediction(result: dict[str, object]) -> str:
    """Reduce la salida jerárquica al único nivel evaluado en esta auditoría."""

    if result["state"] == "fondo":
        return "fondo"
    if result["state"] == "dron":
        return "dron"
    return "no_concluyente"


def binary_metrics(truth, prediction) -> dict[str, object]:
    """Calcula tasas y conteos explícitos de falsas alarmas y pérdidas."""

    labels = np.asarray(truth).astype(str)
    predicted = np.asarray(prediction).astype(str)
    report = classification_report_dict(labels, predicted)
    background = labels == "fondo"
    drone = labels == "dron"
    false_alarm = predicted == "dron"
    detected = predicted == "dron"
    report.update(
        {
            "n_groups": int(len(labels)),
            "background_groups": int(np.sum(background)),
            "drone_groups": int(np.sum(drone)),
            "false_alarm_count": int(np.sum(false_alarm[background])),
            "false_alarm_rate": float(np.mean(false_alarm[background])),
            "missed_drone_count": int(np.sum(~detected[drone])),
            "missed_drone_rate": float(np.mean(~detected[drone])),
            "detection_rate": float(np.mean(detected[drone])),
            "coverage": float(np.mean(predicted != "no_concluyente")),
        }
    )
    return report


def with_windows(
    signal_input: SignalInput,
    *,
    start: int = 0,
    stop: int | None = None,
    parts: tuple[str, ...] | None = None,
    noise_snr_db: float | None = None,
    bits: int | None = None,
    rng: np.random.Generator | None = None,
) -> SignalInput:
    """Crea una entrada perturbada sin añadir etiqueta ni metadatos de catálogo."""

    if signal_input.window_sets is None:
        raise ValueError("La auditoría requiere paquetes con ventanas L/H")
    selected = parts or tuple(signal_input.window_sets)
    window_sets = {}
    for part in selected:
        values = np.asarray(signal_input.window_sets[part][start:stop], dtype=float).copy()
        if noise_snr_db is not None:
            if rng is None:
                raise ValueError("Falta generador para el ruido")
            power = float(np.mean(values**2))
            noise_power = power / (10.0 ** (noise_snr_db / 10.0))
            values += rng.normal(scale=np.sqrt(noise_power), size=values.shape)
        if bits is not None:
            values = quantize_uniform(values, bits=bits, full_scale=1.0)
        window_sets[part] = values
    return SignalInput(
        signals={part: values.reshape(-1) for part, values in window_sets.items()},
        fs_hz=signal_input.fs_hz,
        source_name=signal_input.source_name,
        window_sets=window_sets,
    )


def infer_samples_without_labels(samples_root: Path, bundle: dict[str, object], transform):
    """Infiere primero; las etiquetas del catálogo se unen por separado después."""

    rows = []
    for path in sorted(samples_root.rglob("*.npz")):
        signal_input = transform(load_signal_input(path))
        features = extract_input_features(signal_input, bundle)
        result = predict_hierarchy(bundle, features)
        rows.append(
            {
                "relative_path": path.relative_to(samples_root).as_posix(),
                "prediction": activity_prediction(result),
                "drone_score": float(result["drone_score"]),
                "threshold": float(result["threshold"]),
            }
        )
    return pd.DataFrame(rows)


def infer_samples_sin_politica_lh(samples_root: Path, bundle: dict[str, object], transform):
    """Diagnosticar el clasificador binario sin aplicar la seguridad final L/H.

    Esta función existe solo para mostrar por qué la aplicación exige ambas partes.
    No se usa en la demo ni en una decisión presentada al usuario.
    """

    columns = list(bundle["feature_columns"])
    model = bundle["models"]["actividad"]
    threshold = float(bundle["thresholds"]["actividad"]["decision"])
    rows = []
    for path in sorted(samples_root.rglob("*.npz")):
        signal_input = transform(load_signal_input(path))
        features = extract_input_features(signal_input, bundle)
        probabilities = model.predict_proba(features[columns].to_numpy(dtype=float))
        scores = pd.DataFrame(probabilities, columns=[str(value) for value in model.classes_])
        scores["part"] = features["part"].to_numpy()
        drone_score = float(scores.groupby("part")["dron"].median().mean())
        rows.append(
            {
                "relative_path": path.relative_to(samples_root).as_posix(),
                "prediction": "dron" if drone_score >= threshold else "fondo",
                "drone_score": drone_score,
                "threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def attach_expected_activity(table: pd.DataFrame, samples_root: Path) -> pd.DataFrame:
    """Revela la actividad esperada solo una vez que terminaron las predicciones."""

    catalog = json.loads((samples_root / "catalogo_privado.json").read_text(encoding="utf-8-sig"))
    expected = {
        item["relative_path"]: item["expected"]["activity"] for item in catalog["samples"]
    }
    output = table.copy()
    output["expected"] = output["relative_path"].map(expected)
    if output["expected"].isna().any():
        raise ValueError("Una muestra no está registrada en el catálogo separado")
    return output


def summarize_conditions(rows: list[dict[str, object]], condition: str) -> pd.DataFrame:
    """Agrupa repeticiones de una perturbación sin ocultar su dispersión."""

    frame = pd.DataFrame(rows)
    numeric = [
        "accuracy",
        "balanced_accuracy",
        "false_alarm_rate",
        "missed_drone_rate",
        "coverage",
    ]
    summary = frame.groupby(condition, as_index=False)[numeric].agg(["mean", "min", "max"])
    summary.columns = [
        column if not statistic else f"{column}_{statistic}"
        for column, statistic in summary.columns.to_flat_index()
    ]
    return summary


def add_artifact(manifest: dict[str, object], name: str, path: Path) -> None:
    manifest["artifacts"][name] = {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--samples-dir", type=Path, default=PROJECT_ROOT / "muestras_demo")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.repetitions, args.permutations) < 1:
        raise ValueError("repetitions y permutations deben ser positivos")

    run_root = PROJECT_ROOT / "resultados" / "runs" / args.run_id
    guarded = (
        PROJECT_ROOT
        / "resultados"
        / "runs"
        / "actual"
        / "modelos"
        / "bundle_demo_conservador.joblib"
    )
    bundle_path = args.bundle or guarded
    if not bundle_path.is_file():
        bundle_path = run_root / "modelos" / "bundle.joblib"
    bundle = load_bundle(bundle_path)
    samples_root = args.samples_dir.expanduser().resolve()

    features = pd.read_csv(run_root / "features" / "features.csv")
    development = features[features["partition"] == "desarrollo"].copy()
    selected = features_for_band_count(development, int(bundle["band_count"]))
    feature_columns = classifier_feature_columns(int(bundle["band_count"]))

    nested_rows = []
    nested_trace = []
    for repeat in range(args.repetitions):
        seed = args.seed + repeat
        scores, folds = nested_binary_group_scores(
            selected,
            feature_columns,
            n_outer_splits=5,
            n_inner_splits=4,
            random_state=seed,
        )
        report = binary_metrics(scores["activity"], scores["prediction"])
        report.update({"repeat": repeat + 1, "seed": seed})
        nested_rows.append(report)
        nested_trace.append(
            {
                "repeat": repeat + 1,
                "seed": seed,
                "all_group_separations_valid": all(
                    not set(fold["test_group_ids"]).intersection(fold["train_group_ids"])
                    and not set(fold["test_group_ids"]).intersection(fold["threshold_group_ids"])
                    for fold in folds
                ),
                "folds": folds,
            }
        )

    permutation_rows = []
    group_labels = selected[["group_id", "activity"]].drop_duplicates()
    group_labels = group_labels.sort_values("group_id").reset_index(drop=True)
    for repeat in range(args.permutations):
        rng = np.random.default_rng(args.seed + 10_000 + repeat)
        shuffled = rng.permutation(group_labels["activity"].to_numpy())
        mapping = dict(zip(group_labels["group_id"], shuffled, strict=True))
        permuted = selected.copy()
        permuted["activity"] = permuted["group_id"].map(mapping)
        scores, _ = grouped_oof_scores(
            permuted,
            feature_columns,
            STAGES["actividad"],
            n_splits=5,
            random_state=args.seed + 20_000 + repeat,
        )
        threshold = choose_binary_threshold(scores["activity"], scores[f"{SCORE_PREFIX}dron"])
        _, predicted = summarize_group_scores(
            scores,
            STAGES["actividad"],
            binary_threshold=threshold,
        )
        report = binary_metrics(predicted["activity"], predicted["prediction"])
        report.update({"permutation": repeat + 1, "seed": args.seed + 10_000 + repeat})
        permutation_rows.append(report)

    chunk_rows = []
    for chunk in range(4):
        raw = infer_samples_without_labels(
            samples_root,
            bundle,
            lambda entry, start=chunk * 25: with_windows(entry, start=start, stop=start + 25),
        )
        checked = attach_expected_activity(raw, samples_root)
        report = binary_metrics(checked["expected"], checked["prediction"])
        report.update({"chunk": chunk + 1, "windows_per_part": 25})
        chunk_rows.append(report)

    part_policy_rows = []
    part_diagnostic_rows = []
    for part in ("L", "H", "L_H"):
        selected_parts = ("L", "H") if part == "L_H" else (part,)
        raw = infer_samples_without_labels(
            samples_root,
            bundle,
            lambda entry, current=selected_parts: with_windows(
                entry,
                parts=current,
            ),
        )
        checked = attach_expected_activity(raw, samples_root)
        report = binary_metrics(checked["expected"], checked["prediction"])
        report["partes"] = part
        part_policy_rows.append(report)

        diagnostic = infer_samples_sin_politica_lh(
            samples_root,
            bundle,
            lambda entry, current=selected_parts: with_windows(
                entry,
                parts=current,
            ),
        )
        diagnostic_checked = attach_expected_activity(diagnostic, samples_root)
        diagnostic_report = binary_metrics(
            diagnostic_checked["expected"], diagnostic_checked["prediction"]
        )
        diagnostic_report["partes"] = part
        part_diagnostic_rows.append(diagnostic_report)

    noise_rows = []
    for snr_db in (-5, 0, 5, 10, 20):
        for repeat in range(args.repetitions):
            rng = np.random.default_rng(args.seed + 30_000 + repeat * 100 + snr_db)
            raw = infer_samples_without_labels(
                samples_root,
                bundle,
                lambda entry, current_rng=rng, current_snr=snr_db: with_windows(
                    entry,
                    noise_snr_db=current_snr,
                    rng=current_rng,
                ),
            )
            checked = attach_expected_activity(raw, samples_root)
            report = binary_metrics(checked["expected"], checked["prediction"])
            report.update({"snr_db": snr_db, "repeat": repeat + 1})
            noise_rows.append(report)

    quantization_rows = []
    for bits in (4, 6, 8, 12):
        raw = infer_samples_without_labels(
            samples_root,
            bundle,
            lambda entry, current_bits=bits: with_windows(
                entry,
                bits=current_bits,
            ),
        )
        checked = attach_expected_activity(raw, samples_root)
        report = binary_metrics(checked["expected"], checked["prediction"])
        report["bits"] = bits
        quantization_rows.append(report)

    output_tables = run_root / "tablas"
    output_metrics = run_root / "metricas"
    output_figures = run_root / "figuras"
    output_tables.mkdir(parents=True, exist_ok=True)
    output_metrics.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)
    paths = {
        "nested": output_tables / "auditoria_robustez_cv_anidada.csv",
        "permutation": output_tables / "auditoria_robustez_permuta_etiquetas.csv",
        "chunks": output_tables / "auditoria_robustez_tramos.csv",
        "parts": output_tables / "auditoria_robustez_partes.csv",
        "parts_policy": output_tables / "auditoria_robustez_partes_politica.csv",
        "noise": output_tables / "auditoria_robustez_ruido.csv",
        "quantization": output_tables / "auditoria_robustez_cuantizacion.csv",
    }
    pd.DataFrame(nested_rows).to_csv(paths["nested"], index=False)
    pd.DataFrame(permutation_rows).to_csv(paths["permutation"], index=False)
    pd.DataFrame(chunk_rows).to_csv(paths["chunks"], index=False)
    pd.DataFrame(part_diagnostic_rows).to_csv(paths["parts"], index=False)
    pd.DataFrame(part_policy_rows).to_csv(paths["parts_policy"], index=False)
    pd.DataFrame(noise_rows).to_csv(paths["noise"], index=False)
    pd.DataFrame(quantization_rows).to_csv(paths["quantization"], index=False)

    noise_summary = summarize_conditions(noise_rows, "snr_db")
    metrics = {
        "schema_version": "1.0",
        "executed_at_utc": utc_now(),
        "bundle_sha256": sha256_file(bundle_path),
        "label_accessed_after_inference": True,
        "scope": (
            "Robustez de fondo/dron en el subset DroneRF v2 y perturbaciones relativas; "
            "no demuestra desempeño operacional ni identificación de modelo/modo."
        ),
        "nested_cv": pd.DataFrame(nested_rows).to_dict(orient="records"),
        "nested_trace": nested_trace,
        "label_permutation": pd.DataFrame(permutation_rows).to_dict(orient="records"),
        "temporal_chunks": pd.DataFrame(chunk_rows).to_dict(orient="records"),
        "parts_diagnostic_without_policy": pd.DataFrame(part_diagnostic_rows).to_dict(
            orient="records"
        ),
        "parts_with_final_policy": pd.DataFrame(part_policy_rows).to_dict(orient="records"),
        "noise_summary": noise_summary.to_dict(orient="records"),
        "quantization": pd.DataFrame(quantization_rows).to_dict(orient="records"),
    }
    metrics_path = output_metrics / "auditoria_robustez_detector.json"
    write_json(metrics_path, metrics)

    nested_frame = pd.DataFrame(nested_rows)
    permutation_frame = pd.DataFrame(permutation_rows)
    quantization = pd.DataFrame(quantization_rows)
    chunks = pd.DataFrame(chunk_rows)
    parts = pd.DataFrame(part_diagnostic_rows)
    fig = plot_robustness_summary(
        nested_frame,
        permutation_frame,
        noise_summary,
        quantization,
        chunks,
        parts,
    )
    figure_path = output_figures / "09_auditoria_robustez_detector.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.clear()

    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for name, path in paths.items():
        add_artifact(manifest, f"robustez_{name}", path)
    add_artifact(manifest, "robustez_metricas", metrics_path)
    add_artifact(manifest, "robustez_figura", figure_path)
    manifest["robustness_audit_completed_at_utc"] = metrics["executed_at_utc"]
    write_json(manifest_path, manifest)

    nested_visible = pd.DataFrame(nested_rows)[
        ["repeat", "balanced_accuracy", "false_alarm_count"]
    ]
    print(nested_visible.to_string(index=False))
    print(noise_summary.to_string(index=False))
    quantization_visible = pd.DataFrame(quantization_rows)[
        ["bits", "balanced_accuracy", "false_alarm_count"]
    ]
    print(quantization_visible.to_string(index=False))


if __name__ == "__main__":
    main()
