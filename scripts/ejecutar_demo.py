"""Inicia la aplicación local de demostración APS DroneRF."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aps_drone_rf.demo import (
    extract_input_features,
    load_bundle,
    load_signal_input,
    predict_hierarchy,
    progressive_feature_batches,
)
from aps_drone_rf.demo_plots import (
    band_figure,
    evolution_figure,
    feature_summary,
    fft_figure,
    interpretation_text,
    result_markdown,
    temporal_figure,
    welch_figure,
)
from aps_drone_rf.dronerf import DISPLAY_NAMES

CSS = """
:root {
  --aps-rosa: #E85D9E;
  --aps-lila: #B084F5;
  --aps-violeta: #6F42C1;
  --aps-azul: #74BDE8;
  --aps-texto: #27232D;
  --aps-borde: #E4DEE9;
  --aps-fondo: #F7F7FA;
  --primary-500: #E85D9E;
  --primary-600: #D94A8B;
}
.gradio-container {
  background: var(--aps-fondo);
  color: var(--aps-texto);
  width: 100% !important;
  max-width: 1500px !important;
  box-sizing: border-box;
  margin: 0 auto !important;
  padding: 20px 0 32px !important;
}
.instrument-title h1 {
  font-size: 1.55rem !important;
  margin: 0 !important;
  letter-spacing: 0 !important;
}
.instrument-status { color: #645E6B; font-size: 0.92rem; margin-bottom: 8px; }
.input-panel { border-right: 1px solid var(--aps-borde); padding-right: 18px; }
.result-strip { border-top: 3px solid var(--aps-rosa); padding-top: 10px; }
.result-strip h2 { font-size: 1.25rem !important; letter-spacing: 0 !important; }
.compact textarea, .compact input { font-size: 0.92rem !important; }
button.primary { background: var(--aps-rosa) !important; border-color: var(--aps-rosa) !important; }
button.secondary {
  border-color: var(--aps-violeta) !important;
  color: var(--aps-violeta) !important;
}
input[type="radio"], input[type="checkbox"], input[type="range"] {
  accent-color: var(--aps-rosa) !important;
}
button[role="tab"][aria-selected="true"] {
  color: var(--aps-violeta) !important;
  border-bottom-color: var(--aps-rosa) !important;
}
.tabs { border-radius: 8px !important; }
.block, .form { border-radius: 8px !important; }
"""


def selected_path(
    source_mode: str,
    library_value,
    upload_value,
    samples_dir: Path,
) -> Path:
    """Resuelve la entrada elegida sin consultar su etiqueta."""

    if source_mode == "Biblioteca de muestras":
        if isinstance(library_value, list):
            library_value = library_value[0] if library_value else None
        if not library_value:
            raise gr.Error("Seleccioná una muestra de la biblioteca.")
        path = Path(str(library_value))
        if not path.is_absolute():
            path = samples_dir / path
    else:
        if not upload_value:
            raise gr.Error("Seleccioná un archivo para analizar.")
        path = Path(str(upload_value))
    return path.expanduser().resolve()


def visible_result(result: dict[str, object]) -> tuple[str, str, str, str]:
    state_names = {
        "fondo": "Sin dron",
        "dron": "Actividad de dron",
        "no_concluyente": "No concluyente",
    }
    model = result.get("model")
    mode = result.get("mode")
    return (
        state_names[str(result["state"])],
        DISPLAY_NAMES.get(str(model), str(model)) if model else "No concluyente",
        DISPLAY_NAMES.get(str(mode), str(mode)) if mode else "No concluyente",
        (
            f"{float(result['drone_score']):.3f} / "
            f"{float(result['threshold']):.3f} / {float(result['margin']):.3f}"
        ),
    )


def make_outputs(signal_input, features, bundle, result, evolution):
    state, model, mode, score = visible_result(result)
    return (
        result_markdown(result),
        state,
        model,
        mode,
        score,
        interpretation_text(result),
        temporal_figure(signal_input),
        fft_figure(signal_input),
        welch_figure(signal_input),
        band_figure(features, int(bundle["band_count"])),
        evolution_figure(evolution, float(result["threshold"])),
        feature_summary(features),
    )


def build_app(bundle_path: Path, samples_dir: Path) -> gr.Blocks:
    """Construye el instrumento y carga el bundle una sola vez."""

    bundle = load_bundle(bundle_path)
    samples_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = samples_dir / "catalogo_privado.json"

    def toggle_source(mode):
        return (
            gr.update(visible=mode == "Biblioteca de muestras"),
            gr.update(visible=mode == "Subir archivo"),
        )

    def remember_library_selection(evt: gr.SelectData):
        value = str(evt.value) if evt.value is not None else ""
        if evt.selected and value.lower().endswith(".npz"):
            return value
        return ""

    def sampling_rate_override(fs_hz):
        value = str(fs_hz).strip()
        return float(value) if value else None

    def analyze_decision(source_mode, library_value, upload_value, fs_hz):
        path = selected_path(source_mode, library_value, upload_value, samples_dir)
        fs_override = sampling_rate_override(fs_hz)
        signal_input = load_signal_input(path, fs_hz=fs_override)
        features = extract_input_features(signal_input, bundle)
        result = predict_hierarchy(bundle, features)
        state, model, mode, score = visible_result(result)
        return (
            result_markdown(result),
            state,
            model,
            mode,
            score,
            interpretation_text(result),
            True,
            str(path),
            gr.update(interactive=True),
            signal_input,
            features,
            result,
        )

    def analyze_plots(signal_input, features, result):
        return (
            temporal_figure(signal_input),
            fft_figure(signal_input),
            welch_figure(signal_input),
            band_figure(features, int(bundle["band_count"])),
            evolution_figure([float(result["drone_score"])], float(result["threshold"])),
            feature_summary(features),
        )

    def replay(source_mode, library_value, upload_value, fs_hz, speed):
        path = selected_path(source_mode, library_value, upload_value, samples_dir)
        fs_override = sampling_rate_override(fs_hz)
        signal_input = load_signal_input(path, fs_hz=fs_override)
        evolution = []
        for features in progressive_feature_batches(signal_input, bundle):
            result = predict_hierarchy(bundle, features)
            evolution.append(float(result["drone_score"]))
            outputs = make_outputs(signal_input, features, bundle, result, evolution)
            yield (*outputs, True, str(path), gr.update(interactive=True))
            time.sleep(max(0.05, 0.25 / max(float(speed), 0.1)))

    def reset():
        empty_plot = evolution_figure([], 0.5)
        return (
            "## Sin análisis",
            "-",
            "-",
            "-",
            "-",
            "",
            None,
            None,
            None,
            None,
            empty_plot,
            None,
            False,
            "",
            gr.update(interactive=False),
            None,
            None,
            None,
            "",
        )

    def reveal_label(analysis_done, analyzed_path):
        if not analysis_done or not analyzed_path:
            raise gr.Error("Primero ejecutá el análisis.")
        path = Path(analyzed_path).resolve()
        if not catalog_path.is_file():
            return "La entrada no tiene una etiqueta de demo disponible."
        catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
        try:
            relative = path.relative_to(samples_dir.resolve()).as_posix()
        except ValueError:
            return "El archivo subido no pertenece a la biblioteca ciega."
        entry = next(
            (item for item in catalog["samples"] if item["relative_path"] == relative),
            None,
        )
        if entry is None:
            return "La entrada no tiene una etiqueta de demo disponible."
        expected = entry["expected"]
        return (
            f"Esperado: {DISPLAY_NAMES.get(expected['activity'], expected['activity'])}; "
            f"modelo {DISPLAY_NAMES.get(expected['model'], expected['model'])}; "
            f"modo {DISPLAY_NAMES.get(expected['mode'], expected['mode'])}."
        )

    with gr.Blocks(title="Análisis RF - DroneRF") as app:
        analysis_done = gr.State(False)
        analyzed_path = gr.State("")
        selected_library = gr.State("")
        signal_state = gr.State(None)
        features_state = gr.State(None)
        result_state = gr.State(None)
        gr.Markdown("# Análisis de señales RF - DroneRF", elem_classes="instrument-title")
        gr.Markdown(
            "Simulación de adquisición desde archivo · amplitudes normalizadas por registro",
            elem_classes="instrument-status",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=300, elem_classes="input-panel"):
                source_mode = gr.Radio(
                    ["Biblioteca de muestras", "Subir archivo"],
                    value="Biblioteca de muestras",
                    label="Fuente",
                )
                library = gr.FileExplorer(
                    glob="**/*.npz",
                    root_dir=samples_dir,
                    file_count="single",
                    label="Biblioteca",
                    height=220,
                )
                upload = gr.File(
                    file_types=[".csv", ".txt", ".npy", ".npz", ".mat"],
                    type="filepath",
                    label="Archivo RF",
                    visible=False,
                )
                fs_hz = gr.Textbox(
                    value="",
                    label="Frecuencia de muestreo [Hz]",
                    placeholder="Solo si el archivo no la incluye",
                    elem_classes="compact",
                )
                speed = gr.Slider(
                    minimum=0.5,
                    maximum=3.0,
                    value=1.0,
                    step=0.5,
                    label="Velocidad visual",
                )
                with gr.Row():
                    analyze_button = gr.Button(
                        "Analizar", variant="primary", scale=1, min_width=0
                    )
                    replay_button = gr.Button(
                        "Reproducir", variant="secondary", scale=1, min_width=0
                    )
                with gr.Row():
                    stop_button = gr.Button("Detener", scale=1, min_width=0)
                    reset_button = gr.Button("Reiniciar", scale=1, min_width=0)
                reveal_button = gr.Button("Mostrar etiqueta esperada", interactive=False)
                expected_label = gr.Markdown("")

            with gr.Column(scale=7, min_width=620):
                result_md = gr.Markdown("## Sin análisis", elem_classes="result-strip")
                with gr.Row():
                    state_box = gr.Textbox(label="Estado", value="-", interactive=False)
                    model_box = gr.Textbox(
                        label="Modelo (si hay respaldo)", value="-", interactive=False
                    )
                    mode_box = gr.Textbox(
                        label="Modo (si hay respaldo)", value="-", interactive=False
                    )
                score_box = gr.Textbox(
                    label="Puntaje relativo / umbral / margen",
                    value="-",
                    interactive=False,
                )
                interpretation = gr.Markdown("")
                with gr.Tabs():
                    with gr.Tab("Tiempo"):
                        temporal_plot = gr.Plot(label="Señal temporal")
                    with gr.Tab("FFT"):
                        fft_plot = gr.Plot(label="FFT")
                    with gr.Tab("Welch"):
                        welch_plot = gr.Plot(label="PSD por Welch")
                    with gr.Tab("Bandas"):
                        band_plot = gr.Plot(label="Potencia por bandas")
                    with gr.Tab("Evolución"):
                        evolution_plot = gr.Plot(
                            value=evolution_figure([], 0.5),
                            label="Puntaje de actividad",
                        )
                    with gr.Tab("Características"):
                        feature_table = gr.Dataframe(
                            headers=["Característica", "Valor mediano"],
                            interactive=False,
                            label="Características APS",
                        )

        source_mode.change(toggle_source, source_mode, [library, upload])
        library.select(
            remember_library_selection,
            inputs=None,
            outputs=selected_library,
            show_progress="hidden",
        )
        inputs = [source_mode, selected_library, upload, fs_hz]
        replay_outputs = [
            result_md,
            state_box,
            model_box,
            mode_box,
            score_box,
            interpretation,
            temporal_plot,
            fft_plot,
            welch_plot,
            band_plot,
            evolution_plot,
            feature_table,
            analysis_done,
            analyzed_path,
            reveal_button,
        ]
        decision_outputs = [
            result_md,
            state_box,
            model_box,
            mode_box,
            score_box,
            interpretation,
            analysis_done,
            analyzed_path,
            reveal_button,
            signal_state,
            features_state,
            result_state,
        ]
        plot_outputs = [
            temporal_plot,
            fft_plot,
            welch_plot,
            band_plot,
            evolution_plot,
            feature_table,
        ]
        analyze_event = analyze_button.click(
            analyze_decision,
            inputs,
            decision_outputs,
            show_progress="minimal",
        )
        analyze_event.then(
            analyze_plots,
            [signal_state, features_state, result_state],
            plot_outputs,
            show_progress="minimal",
        )
        replay_event = replay_button.click(
            replay,
            [*inputs, speed],
            replay_outputs,
            show_progress="minimal",
        )
        stop_button.click(None, cancels=[replay_event])
        reset_button.click(
            reset,
            outputs=[
                *replay_outputs,
                signal_state,
                features_state,
                result_state,
                expected_label,
            ],
            show_progress="hidden",
        )
        reveal_button.click(
            reveal_label,
            [analysis_done, analyzed_path],
            expected_label,
            show_progress="hidden",
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    models_dir = PROJECT_ROOT / "resultados" / "runs" / "actual" / "modelos"
    conservative_bundle = models_dir / "bundle_demo_conservador.joblib"
    default_bundle = conservative_bundle
    if not conservative_bundle.is_file():
        default_bundle = models_dir / "bundle.joblib"
    parser.add_argument("--bundle", type=Path, default=default_bundle)
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path(os.environ.get("MUESTRAS_DIR", PROJECT_ROOT / "muestras_demo")),
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_app(args.bundle, args.samples_dir.expanduser().resolve())
    app.queue(default_concurrency_limit=2).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=False,
        show_error=True,
        allowed_paths=[str(args.samples_dir.expanduser().resolve())],
        footer_links=[],
        css=CSS,
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
