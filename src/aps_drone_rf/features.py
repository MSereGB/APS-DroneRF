"""Extraccion de caracteristicas simples desde ventanas de señal."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from aps_drone_rf.spectral import band_power, fft_magnitude, integrate_psd_power, welch_psd


def equal_frequency_bands(fs_hz: float, count: int) -> list[tuple[float, float]]:
    """Divide el semiespectro real en bandas de igual ancho."""

    if fs_hz <= 0:
        raise ValueError("fs_hz debe ser positivo")
    if count <= 0:
        raise ValueError("count debe ser positivo")
    edges = np.linspace(0.0, fs_hz / 2.0, count + 1)
    return [(float(edges[index]), float(edges[index + 1])) for index in range(count)]


def features_from_window(
    x: ArrayLike,
    fs_hz: float,
    bands_hz: list[tuple[float, float]] | None = None,
    nperseg: int = 256,
    analysis_window: str = "hann",
) -> dict[str, float]:
    """Calcula features temporales y espectrales simples para una ventana."""

    signal_x = np.asarray(x).reshape(-1)
    power_samples = np.abs(signal_x) ** 2
    rms = float(np.sqrt(np.mean(power_samples)))
    power = float(np.mean(power_samples))
    energy = float(np.sum(power_samples))
    peak = float(np.max(np.abs(signal_x)))
    crest = float(peak / rms) if rms > 0 else 0.0

    fft_freqs, fft_values, _ = fft_magnitude(
        signal_x,
        fs_hz=fs_hz,
        window=analysis_window,
    )
    fft_peak_index = int(np.argmax(fft_values)) if len(fft_values) else 0
    dominant_fft = float(fft_freqs[fft_peak_index]) if len(fft_freqs) else 0.0

    freqs, psd = welch_psd(
        signal_x,
        fs_hz=fs_hz,
        nperseg=min(nperseg, len(signal_x)),
        window=analysis_window,
    )
    psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)
    abs_freqs = np.abs(freqs)
    total_psd_power = integrate_psd_power(freqs, psd)
    max_idx = int(np.argmax(psd)) if len(psd) else 0
    dominant_freq = float(abs_freqs[max_idx]) if len(freqs) else 0.0
    max_psd = float(psd[max_idx]) if len(psd) else 0.0
    weight_sum = float(np.sum(psd))
    if weight_sum > 0:
        centroid = float(np.sum(abs_freqs * psd) / weight_sum)
        bandwidth = float(np.sqrt(np.sum(((abs_freqs - centroid) ** 2) * psd) / weight_sum))
    else:
        centroid = 0.0
        bandwidth = 0.0

    features = {
        "rms": rms,
        "potencia_media": power,
        "energia": energy,
        "pico": peak,
        "factor_cresta": crest,
        "frecuencia_dominante_fft_hz": dominant_fft,
        "frecuencia_dominante_hz": dominant_freq,
        "psd_maxima": max_psd,
        "potencia_total_psd": total_psd_power,
        "centroide_espectral_hz": centroid,
        "ancho_banda_espectral_hz": bandwidth,
    }
    if bands_hz:
        powers = band_power(freqs, psd, bands_hz)
        features.update(powers)
        denominator = max(float(sum(powers.values())), np.finfo(float).eps)
        for index, value in enumerate(powers.values(), start=1):
            features[f"potencia_relativa_banda_{index}"] = float(value / denominator)
    return {
        key: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
        for key, value in features.items()
    }


def build_feature_table(
    windows: np.ndarray,
    metadata: pd.DataFrame,
    fs_hz: float,
    bands_hz: list[tuple[float, float]] | None = None,
    nperseg: int = 256,
    analysis_window: str = "hann",
) -> pd.DataFrame:
    """Construye una tabla de features y conserva metadatos de ventanas."""

    rows = []
    for idx, window in enumerate(windows):
        base = metadata.iloc[idx].to_dict() if len(metadata) > idx else {"window_id": str(idx)}
        base.update(
            features_from_window(
                window,
                fs_hz=fs_hz,
                bands_hz=bands_hz,
                nperseg=nperseg,
                analysis_window=analysis_window,
            )
        )
        rows.append(base)
    return pd.DataFrame(rows)
