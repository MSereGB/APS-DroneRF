"""Política conservadora para presentar resultados en la demostración local."""

from __future__ import annotations

from copy import deepcopy


def conservative_stage_reliability(metrics: dict[str, object]) -> dict[str, dict[str, object]]:
    """Habilita una etapa solo si desarrollo y evaluación reservada la respaldan.

    La evaluación no cambia features, modelos ni umbrales. Solo controla qué nivel de
    detalle es responsable mostrar en la interfaz final.
    """

    development = metrics["stage_reliability"]
    evaluation = metrics["reserved_evaluation"]
    dummy = metrics["dummy_oof"]
    result: dict[str, dict[str, object]] = {}
    for name, previous in development.items():
        report = evaluation[name]
        baseline = float(dummy[name]["balanced_accuracy"])
        balanced_accuracy = float(report["balanced_accuracy"])
        group_count = int(report["n_groups"])
        if name == "actividad":
            evaluation_ok = (
                group_count >= 6
                and balanced_accuracy > baseline
                and float(report.get("false_alarm_rate", 1.0)) <= 0.10
                and float(report.get("detection_rate", 0.0)) >= 0.90
            )
            rule = (
                "Superar al Dummy en desarrollo y evaluación; en evaluación exigir al "
                "menos 6 grupos, detección >= 0,90 y falsas alarmas <= 0,10."
            )
        else:
            # Para identificación multiclase, cuatro grupos reservados no alcanzan para
            # asegurar dos ejemplos por clase ni sostener una salida para la demo.
            evaluation_ok = group_count >= 8 and balanced_accuracy > baseline
            rule = (
                "Superar al Dummy en desarrollo y evaluación, con al menos 8 grupos "
                "reservados para mostrar la etapa en la demo."
            )
        development_ok = bool(previous["enabled"])
        enabled = development_ok and evaluation_ok
        result[name] = {
            **previous,
            "enabled": enabled,
            "development_enabled": development_ok,
            "reserved_evaluation_balanced_accuracy": balanced_accuracy,
            "reserved_evaluation_groups": group_count,
            "reserved_evaluation_dummy": baseline,
            "reserved_evaluation_passed": evaluation_ok,
            "rule": rule,
        }
    return result


def apply_conservative_policy(
    bundle: dict[str, object], metrics: dict[str, object]
) -> dict[str, object]:
    """Devuelve una copia del bundle con una política de presentación verificable."""

    output = deepcopy(bundle)
    reliability = conservative_stage_reliability(metrics)
    output["stage_reliability"] = reliability
    output["deployment_policy"] = {
        "name": "conservadora_evaluacion_reservada_v1",
        "purpose": "Limitar la interfaz a los niveles que sostienen desarrollo y evaluación.",
        "required_parts_for_activity": ["L", "H"],
        "single_part_behavior": (
            "Conservar gráficos y características, pero informar No concluyente para "
            "evitar aplicar el umbral L/H a una única parte."
        ),
        "stage_reliability": reliability,
    }
    return output
