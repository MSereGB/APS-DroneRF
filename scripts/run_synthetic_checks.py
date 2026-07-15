"""Genera verificaciones sinteticas basicas para FFT, PSD, ventanas y cuantizacion."""

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
    from aps_drone_rf.estilo import COLOR_SERIE, aplicar_estilo_matplotlib
    from aps_drone_rf.plots import plot_spectrum, plot_time_signal, save_figure
    from aps_drone_rf.spectral import fft_magnitude, periodogram_psd, welch_psd
    from aps_drone_rf.synthetic import (
        add_awgn,
        amplitude_modulated,
        pure_tone,
        quantize_uniform,
        sum_of_tones,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--fs", type=float, default=2_000.0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_project_dirs()
    aplicar_estilo_matplotlib()

    tone_freq = 180.0
    t, tone = pure_tone(tone_freq, args.fs, args.duration)
    _, mixture = sum_of_tones([180.0, 420.0], args.fs, args.duration, amplitudes=[1.0, 0.4])
    noisy = add_awgn(mixture, snr_db=10.0, seed=args.seed)
    _, am_signal = amplitude_modulated(
        carrier_hz=500.0,
        mod_hz=40.0,
        fs_hz=args.fs,
        duration_s=args.duration,
        modulation_index=0.6,
    )

    freqs, magnitude, _ = fft_magnitude(tone, args.fs, window="hann")
    peak_freq = float(freqs[magnitude.argmax()])

    f_periodogram, p_periodogram = periodogram_psd(noisy, args.fs, window="hann")
    f_welch, p_welch = welch_psd(noisy, args.fs, nperseg=256, window="hann")

    fig, _ = plot_time_signal(t[:400], noisy[:400], title="Suma de senoidales con ruido")
    save_figure(fig, FIGURES_DIR / "synthetic_time_signal.png")
    plt.close(fig)

    fig, _ = plot_spectrum(freqs, magnitude, title="FFT de senoidal pura")
    save_figure(fig, FIGURES_DIR / "synthetic_fft_tone.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(
        f_periodogram,
        10.0 * np.log10(np.maximum(p_periodogram, 1e-20)),
        label="Periodograma",
        color=COLOR_SERIE["periodograma"],
    )
    ax.plot(
        f_welch,
        10.0 * np.log10(np.maximum(p_welch, 1e-20)),
        label="Welch",
        color=COLOR_SERIE["welch"],
    )
    ax.set_title("Periodograma vs Welch - señal sintetica")
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("PSD [dB rel.]")
    ax.legend()
    save_figure(fig, FIGURES_DIR / "synthetic_periodogram_vs_welch.png")
    plt.close(fig)

    detuned_freq = tone_freq + 0.35 * args.fs / len(tone)
    _, detuned = pure_tone(detuned_freq, args.fs, args.duration)
    fig, ax = plt.subplots(figsize=(8, 3))
    for window_name in ["rectangular", "hann", "hamming", "blackman"]:
        f_aux, mag_aux, _ = fft_magnitude(detuned, args.fs, window=window_name)
        ax.plot(
            f_aux,
            20.0 * np.log10(np.maximum(mag_aux, 1e-12)),
            label=window_name,
            color=COLOR_SERIE[window_name],
        )
    ax.set_xlim(tone_freq - 40, tone_freq + 40)
    ax.set_ylim(-90, 5)
    ax.set_title("Efecto del ventaneo sobre la fuga espectral")
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Magnitud [dB rel.]")
    ax.legend()
    save_figure(fig, FIGURES_DIR / "synthetic_window_fuga_espectral.png")
    plt.close(fig)

    f_am, mag_am, _ = fft_magnitude(am_signal, args.fs, window="hann")
    fig, _ = plot_spectrum(f_am, mag_am, title="FFT de modulacion AM simple")
    save_figure(fig, FIGURES_DIR / "synthetic_am_fft.png")
    plt.close(fig)

    quant_rows = []
    fig, ax = plt.subplots(figsize=(8, 3))
    for bits in [3, 4, 8, 12]:
        quantized = quantize_uniform(tone, bits=bits)
        error = tone - quantized
        quant_rows.append(
            {
                "caso": "cuantizacion",
                "bits": bits,
                "mse": float(np.mean(error**2)),
            }
        )
        ax.plot(t[:80], quantized[:80], linewidth=1.0, label=f"{bits} bits")
    ax.set_title("Efecto de cuantizacion uniforme")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Amplitud")
    ax.legend()
    save_figure(fig, FIGURES_DIR / "synthetic_quantization_bits.png")
    plt.close(fig)

    table = pd.DataFrame(
        [
            {
                "caso": "fft_senoidal_pura",
                "frecuencia_esperada_hz": tone_freq,
                "frecuencia_estimada_hz": peak_freq,
                "error_absoluto_hz": abs(peak_freq - tone_freq),
            },
            {
                "caso": "welch_suma_ruidosa",
                "frecuencia_esperada_hz": 180.0,
                "frecuencia_estimada_hz": float(f_welch[p_welch.argmax()]),
                "error_absoluto_hz": abs(float(f_welch[p_welch.argmax()]) - 180.0),
            },
        ]
        + quant_rows
    )
    output = TABLES_DIR / "synthetic_checks.csv"
    table.to_csv(output, index=False)
    print(f"Figuras guardadas en {FIGURES_DIR}")
    print(f"Tabla guardada en {output}")


if __name__ == "__main__":
    main()
