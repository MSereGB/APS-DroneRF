"""Estudia sensibilidad sintetica ante SNR, frecuencia de muestreo y bits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))

    from aps_drone_rf.config import FIGURES_DIR, TABLES_DIR, ensure_project_dirs
    from aps_drone_rf.estilo import COLORES, aplicar_estilo_matplotlib
    from aps_drone_rf.plots import save_figure
    from aps_drone_rf.spectral import fft_magnitude
    from aps_drone_rf.synthetic import add_awgn, pure_tone, quantize_uniform

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--target-freq", type=float, default=450.3)
    args = parser.parse_args()

    ensure_project_dirs()
    aplicar_estilo_matplotlib()

    rows = []
    rows.extend(
        _snr_sweep(
            args.target_freq,
            args.duration,
            args.seed,
            fft_magnitude,
            add_awgn,
            pure_tone,
        )
    )
    rows.extend(_sampling_sweep(args.target_freq, args.duration, fft_magnitude, pure_tone))
    rows.extend(
        _bits_sweep(
            args.target_freq,
            args.duration,
            fft_magnitude,
            pure_tone,
            quantize_uniform,
        )
    )

    table = pd.DataFrame(rows)
    output = TABLES_DIR / "sensitivity_synthetic.csv"
    table.to_csv(output, index=False)

    _plot_snr(table, COLORES, FIGURES_DIR, save_figure)
    _plot_sampling(table, COLORES, FIGURES_DIR, save_figure)
    _plot_bits(table, COLORES, FIGURES_DIR, save_figure)
    _plot_error_summary(table, COLORES, FIGURES_DIR, save_figure)

    print(f"Tabla guardada en {output}")
    print(table.round(4).to_string(index=False))


def _alias_frequency(freq_hz: float, fs_hz: float) -> float:
    """Frecuencia aparente positiva despues del muestreo."""

    return float(abs(((freq_hz + fs_hz / 2.0) % fs_hz) - fs_hz / 2.0))


def _estimate_from_fft(x, fs_hz: float, expected_hz: float, fft_magnitude) -> dict[str, float]:
    freqs, mag, _ = fft_magnitude(x, fs_hz, window="hann")
    idx = int(np.argmax(mag))
    estimate = float(freqs[idx])
    df_hz = float(fs_hz / len(x))
    floor = float(np.median(mag))
    peak = float(mag[idx])
    contrast = 20.0 * np.log10((peak + 1e-12) / (floor + 1e-12))
    return {
        "frecuencia_estimada_hz": estimate,
        "error_hz": abs(estimate - expected_hz),
        "df_hz": df_hz,
        "contraste_pico_db": float(contrast),
    }


def _snr_sweep(target_freq, duration, seed, fft_magnitude, add_awgn, pure_tone):
    rows = []
    fs_hz = 2_000.0
    expected = _alias_frequency(target_freq, fs_hz)
    _, tone = pure_tone(target_freq, fs_hz, duration)

    for snr_db in [-5.0, 0.0, 5.0, 10.0, 20.0, 40.0]:
        noisy = add_awgn(tone, snr_db=snr_db, seed=seed)
        metrics = _estimate_from_fft(noisy, fs_hz, expected, fft_magnitude)
        rows.append(
            {
                "barrido": "SNR",
                "parametro": "snr_db",
                "valor": snr_db,
                "fs_hz": fs_hz,
                "frecuencia_objetivo_hz": target_freq,
                "frecuencia_esperada_hz": expected,
                "hay_aliasing": False,
                "mse_cuantizacion": np.nan,
                **metrics,
            }
        )
    return rows


def _sampling_sweep(target_freq, duration, fft_magnitude, pure_tone):
    rows = []

    for fs_hz in [700.0, 800.0, 1_000.0, 2_000.0, 4_000.0]:
        expected = _alias_frequency(target_freq, fs_hz)
        _, tone = pure_tone(target_freq, fs_hz, duration)
        metrics = _estimate_from_fft(tone, fs_hz, expected, fft_magnitude)
        rows.append(
            {
                "barrido": "fs",
                "parametro": "fs_hz",
                "valor": fs_hz,
                "fs_hz": fs_hz,
                "frecuencia_objetivo_hz": target_freq,
                "frecuencia_esperada_hz": expected,
                "hay_aliasing": bool(target_freq > fs_hz / 2.0),
                "mse_cuantizacion": np.nan,
                **metrics,
            }
        )
    return rows


def _bits_sweep(target_freq, duration, fft_magnitude, pure_tone, quantize_uniform):
    rows = []
    fs_hz = 2_000.0
    expected = _alias_frequency(target_freq, fs_hz)
    _, tone = pure_tone(target_freq, fs_hz, duration)

    for bits in [2, 3, 4, 6, 8, 12]:
        quantized = quantize_uniform(tone, bits=bits)
        metrics = _estimate_from_fft(quantized, fs_hz, expected, fft_magnitude)
        rows.append(
            {
                "barrido": "bits",
                "parametro": "bits",
                "valor": float(bits),
                "fs_hz": fs_hz,
                "frecuencia_objetivo_hz": target_freq,
                "frecuencia_esperada_hz": expected,
                "hay_aliasing": False,
                "mse_cuantizacion": float(np.mean((tone - quantized) ** 2)),
                **metrics,
            }
        )
    return rows


def _part(table: pd.DataFrame, name: str) -> pd.DataFrame:
    return table[table["barrido"] == name].copy()


def _plot_snr(table, colores, figures_dir, save_figure) -> None:
    part = _part(table, "SNR")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    axes[0].plot(part["valor"], part["error_hz"], marker="o", color=colores["rosa"])
    axes[0].set_title("Error de frecuencia")
    axes[0].set_xlabel("SNR [dB]")
    axes[0].set_ylabel("error [Hz]")

    axes[1].plot(part["valor"], part["contraste_pico_db"], marker="o", color=colores["violeta"])
    axes[1].set_title("Contraste del pico FFT")
    axes[1].set_xlabel("SNR [dB]")
    axes[1].set_ylabel("contraste [dB]")

    fig.suptitle("Sensibilidad ante ruido")
    fig.tight_layout()
    save_figure(fig, figures_dir / "sensitivity_snr.png")


def _plot_sampling(table, colores, figures_dir, save_figure) -> None:
    part = _part(table, "fs")
    fig, ax = plt.subplots(figsize=(7.5, 3.2))

    ax.plot(
        part["valor"],
        part["frecuencia_objetivo_hz"],
        "--",
        color=colores["gris_texto"],
        label="real",
    )
    ax.plot(
        part["valor"],
        part["frecuencia_esperada_hz"],
        marker="o",
        color=colores["lila"],
        label="esperada por muestreo",
    )
    ax.plot(
        part["valor"],
        part["frecuencia_estimada_hz"],
        marker="s",
        color=colores["rosa"],
        label="estimada por FFT",
    )
    ax.set_title("Sensibilidad ante frecuencia de muestreo")
    ax.set_xlabel("fs [Hz]")
    ax.set_ylabel("frecuencia [Hz]")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, figures_dir / "sensitivity_sampling_frequency.png")


def _plot_bits(table, colores, figures_dir, save_figure) -> None:
    part = _part(table, "bits")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    axes[0].plot(part["valor"], part["mse_cuantizacion"], marker="o", color=colores["azul_pastel"])
    axes[0].set_yscale("log")
    axes[0].set_title("Error de cuantización")
    axes[0].set_xlabel("bits")
    axes[0].set_ylabel("MSE")

    axes[1].plot(part["valor"], part["contraste_pico_db"], marker="o", color=colores["violeta"])
    axes[1].set_title("Contraste del pico FFT")
    axes[1].set_xlabel("bits")
    axes[1].set_ylabel("contraste [dB]")

    fig.suptitle("Sensibilidad ante cantidad de bits")
    fig.tight_layout()
    save_figure(fig, figures_dir / "sensitivity_quantization_bits.png")


def _plot_error_summary(table, colores, figures_dir, save_figure) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    color_map = {
        "SNR": colores["rosa"],
        "fs": colores["lila"],
        "bits": colores["azul_pastel"],
    }

    for sweep, part in table.groupby("barrido", sort=False):
        ax.plot(part["valor"], part["error_hz"], marker="o", label=sweep, color=color_map[sweep])

    ax.set_xlabel("valor de barrido")
    ax.set_ylabel("error absoluto [Hz]")
    ax.set_title("Resumen de error en frecuencia dominante")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, figures_dir / "sensitivity_frequency_error.png")


if __name__ == "__main__":
    main()
