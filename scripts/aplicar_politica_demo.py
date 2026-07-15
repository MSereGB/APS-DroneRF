"""Genera un bundle de demostración que no sobreafirma modelo ni modo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.deployment_policy import apply_conservative_policy
from aps_drone_rf.provenance import sha256_file, utc_now, write_json

DEFAULT_RUN_ID = "dronerf_demo_v2_final_n1024_hann_b20_seed42"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_root = PROJECT_ROOT / "resultados" / "runs" / args.run_id
    source = args.source_bundle or run_root / "modelos" / "bundle.joblib"
    output = args.output or run_root / "modelos" / "bundle_demo_conservador.joblib"
    metrics_path = run_root / "metricas" / "metricas_jerarquicas.json"
    bundle = joblib.load(source)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    guarded = apply_conservative_policy(bundle, metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(guarded, output)

    policy_path = run_root / "metricas" / "politica_demo_conservadora.json"
    policy = {
        "schema_version": "1.0",
        "created_at_utc": utc_now(),
        "source_bundle_sha256": sha256_file(source),
        "guarded_bundle_sha256": sha256_file(output),
        "deployment_policy": guarded["deployment_policy"],
    }
    write_json(policy_path, policy)

    actual_models = PROJECT_ROOT / "resultados" / "runs" / "actual" / "modelos"
    actual_models.mkdir(parents=True, exist_ok=True)
    actual_output = actual_models / "bundle_demo_conservador.joblib"
    joblib.dump(guarded, actual_output)
    write_json(
        PROJECT_ROOT / "resultados" / "runs" / "actual" / "politica_origen.json",
        {
            "run_id": args.run_id,
            "bundle": output.relative_to(PROJECT_ROOT).as_posix(),
            "bundle_sha256": sha256_file(output),
            "copied_at_utc": utc_now(),
        },
    )

    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest["deployment_policy"] = guarded["deployment_policy"]
    for name, path in {
        "bundle_demo_conservador": output,
        "politica_demo_conservadora": policy_path,
    }.items():
        manifest["artifacts"][name] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    write_json(manifest_path, manifest)
    print(f"Bundle conservador: {output}")
    for name, stage in guarded["stage_reliability"].items():
        print(f"{name}: {'habilitada' if stage['enabled'] else 'no concluyente'}")


if __name__ == "__main__":
    main()
