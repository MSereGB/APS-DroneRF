"""Regenera la figura de auditoría a partir de tablas ya calculadas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.provenance import sha256_file, write_json
from aps_drone_rf.robustness import plot_robustness_summary

DEFAULT_RUN_ID = "dronerf_demo_v2_final_n1024_hann_b20_seed42"


def noise_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Resume repeticiones de ruido como lo hace la auditoría completa."""

    columns = ["balanced_accuracy", "false_alarm_rate"]
    summary = rows.groupby("snr_db", as_index=False)[columns].mean()
    return summary.rename(columns={f"{name}": f"{name}_mean" for name in columns})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    run_root = PROJECT_ROOT / "resultados" / "runs" / args.run_id
    tables = run_root / "tablas"
    figure_path = run_root / "figuras" / "09_auditoria_robustez_detector.png"
    fig = plot_robustness_summary(
        pd.read_csv(tables / "auditoria_robustez_cv_anidada.csv"),
        pd.read_csv(tables / "auditoria_robustez_permuta_etiquetas.csv"),
        noise_summary(pd.read_csv(tables / "auditoria_robustez_ruido.csv")),
        pd.read_csv(tables / "auditoria_robustez_cuantizacion.csv"),
        pd.read_csv(tables / "auditoria_robustez_tramos.csv"),
        pd.read_csv(tables / "auditoria_robustez_partes.csv"),
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")

    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest["artifacts"]["robustez_figura"] = {
        "path": figure_path.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": int(figure_path.stat().st_size),
        "sha256": sha256_file(figure_path),
    }
    write_json(manifest_path, manifest)
    print(figure_path)


if __name__ == "__main__":
    main()
