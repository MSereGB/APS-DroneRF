"""Herramientas espectrales: FFT, PSD, Welch, ventaneo y Blackman-Tukey."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal


def make_window(name: str, n_samples: int) -> np.ndarray:
    """Devuelve una ventana rectangular, Hann, Hamming o Blackman."""

    if n_samples <= 0:
        raise ValueError("n_samples debe ser positivo")
    normalized = name.lower()
    if normalized in {"rect", "rectangular", "boxcar", "none"}:
        return np.ones(n_samples)
    if normalized in {"hann", "hanning"}:
        return signal.windows.hann(n_samples, sym=False)
    if normalized == "hamming":
        return signal.windows.hamming(n_samples, sym=False)
    if normalized == "blackman":
        return signal.windows.blackman(n_samples, sym=False)
    raise ValueError(f"Ventana no soportada: {name}")


def window_metrics(name: str, n_samples: int, fs_hz: float) -> dict[str, float]:
    """Calcula ganancia coherente y ancho de banda equivalente de una ventana."""

    if fs_hz <= 0:
        raise ValueError("fs_hz debe ser positivo")
    win = make_window(name, n_samples)
    window_sum = float(np.sum(win))
    if abs(window_sum) <= 1e-15:
        raise ValueError("La suma de la ventana no puede ser cero")
    coherent_gain = window_sum / n_samples
    enbw_bins = n_samples * float(np.sum(win**2)) / window_sum**2
    return {
        "ganancia_coherente": coherent_gain,
        "enbw_bins": enbw_bins,
        "enbw_hz": enbw_bins * fs_hz / n_samples,
    }


def relative_psd_diagnostics(psd: ArrayLike, eps: float = 1e-30) -> dict[str, float]:
    """Resume pico, piso mediano y contraste de una PSD en escala relativa."""

    values = np.asarray(psd, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("La PSD no puede estar vacía")
    if eps <= 0:
        raise ValueError("eps debe ser positivo")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    psd_db = 10.0 * np.log10(np.maximum(values, eps))
    floor_db = float(np.median(psd_db))
    peak_db = float(np.max(psd_db))
    return {
        "piso_psd_db_rel": floor_db,
        "pico_psd_db_rel": peak_db,
        "contraste_pico_piso_db": max(peak_db - floor_db, 0.0),
    }


def fft_magnitude(
    x: ArrayLike,
    fs_hz: float,
    n_fft: int | None = None,
    window: str | None = None,
    detrend: bool = False,
    one_sided: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula FFT y espectro de magnitud normalizado por coherencia de ventana."""

    signal_x = np.asarray(x)
    if signal_x.ndim != 1:
        signal_x = signal_x.reshape(-1)
    if detrend:
        signal_x = signal.detrend(signal_x)
    if n_fft is None:
        n_fft = len(signal_x)
    if fs_hz <= 0:
        raise ValueError("fs_hz debe ser positivo")
    if one_sided is None:
        one_sided = not np.iscomplexobj(signal_x)

    win = np.ones(len(signal_x)) if window is None else make_window(window, len(signal_x))
    windowed = signal_x * win
    coherent_gain = np.sum(win) / len(win)
    scale = max(len(signal_x) * coherent_gain, 1e-12)

    if one_sided:
        spectrum = np.fft.rfft(windowed, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs_hz)
        magnitude = np.abs(spectrum) / scale
        if len(magnitude) > 2:
            magnitude[1:-1] *= 2.0
    else:
        spectrum = np.fft.fft(windowed, n=n_fft)
        freqs = np.fft.fftfreq(n_fft, d=1.0 / fs_hz)
        order = np.argsort(freqs)
        freqs = freqs[order]
        spectrum = spectrum[order]
        magnitude = np.abs(spectrum) / scale
    return freqs, magnitude, spectrum


def periodogram_psd(
    x: ArrayLike,
    fs_hz: float,
    window: str = "boxcar",
    n_fft: int | None = None,
    detrend: str | bool = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """Estima PSD con periodograma."""

    return signal.periodogram(
        np.asarray(x).reshape(-1),
        fs=fs_hz,
        window=make_window(window, len(np.asarray(x).reshape(-1))),
        nfft=n_fft,
        detrend=detrend,
        scaling="density",
        return_onesided=not np.iscomplexobj(x),
    )


def welch_psd(
    x: ArrayLike,
    fs_hz: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """Estima PSD con el metodo de Welch."""

    signal_x = np.asarray(x).reshape(-1)
    nperseg = min(int(nperseg), len(signal_x))
    if nperseg <= 0:
        raise ValueError("nperseg debe ser positivo")
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = min(int(noverlap), nperseg - 1)
    return signal.welch(
        signal_x,
        fs=fs_hz,
        window=make_window(window, nperseg),
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling="density",
        return_onesided=not np.iscomplexobj(signal_x),
    )


def blackman_tukey_psd(
    x: ArrayLike,
    fs_hz: float,
    max_lag: int | None = None,
    window: str = "blackman",
    n_fft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimacion simple Blackman-Tukey a partir de autocorrelacion ventaneada."""

    signal_x = np.asarray(x).reshape(-1)
    signal_x = signal_x - np.mean(signal_x)
    n = len(signal_x)
    if max_lag is None:
        max_lag = min(n - 1, n // 4)
    if max_lag <= 0 or max_lag >= n:
        raise ValueError("max_lag debe estar entre 1 y len(x)-1")
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(2 * max_lag + 1)))

    autocorr_full = np.correlate(signal_x, signal_x, mode="full") / n
    center = n - 1
    autocorr = autocorr_full[center - max_lag : center + max_lag + 1]
    lag_window = make_window(window, len(autocorr))
    psd_twosided = np.real(np.fft.fftshift(np.fft.fft(autocorr * lag_window, n=n_fft))) / fs_hz
    freqs_twosided = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / fs_hz))
    mask = freqs_twosided >= 0
    return freqs_twosided[mask], np.maximum(psd_twosided[mask], 0.0)


def _trapezoid_sorted(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]))


def integrate_psd_power(freqs_hz: ArrayLike, psd: ArrayLike) -> float:
    """Integra una PSD one-sided o two-sided respetando el eje de frecuencias."""

    freqs = np.asarray(freqs_hz, dtype=float)
    values = np.nan_to_num(np.asarray(psd, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if freqs.shape != values.shape:
        raise ValueError("freqs_hz y psd deben tener la misma forma")
    if np.any(freqs < 0):
        power = _trapezoid_sorted(freqs, values)
    else:
        power = _trapezoid_sorted(np.abs(freqs), values)
    return max(float(power), 0.0)


def _integrate_frequency_interval(
    freqs_hz: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    """Integra un intervalo incluyendo sus bordes mediante interpolacion lineal."""

    if freqs_hz.size < 2:
        return 0.0
    order = np.argsort(freqs_hz)
    freqs = freqs_hz[order]
    values = psd[order]
    low = max(float(low_hz), float(freqs[0]))
    high = min(float(high_hz), float(freqs[-1]))
    if high <= low:
        return 0.0

    interior = (freqs > low) & (freqs < high)
    interval_freqs = np.concatenate(([low], freqs[interior], [high]))
    interval_values = np.concatenate(
        (
            [np.interp(low, freqs, values)],
            values[interior],
            [np.interp(high, freqs, values)],
        )
    )
    return max(float(np.trapezoid(interval_values, interval_freqs)), 0.0)


def band_power(
    freqs_hz: ArrayLike, psd: ArrayLike, bands_hz: list[tuple[float, float]]
) -> dict[str, float]:
    """Integra potencia de PSD en bandas sin dejar huecos entre limites contiguos."""

    freqs = np.asarray(freqs_hz, dtype=float)
    values = np.nan_to_num(np.asarray(psd, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if freqs.shape != values.shape:
        raise ValueError("freqs_hz y psd deben tener la misma forma")
    positive_mask = freqs >= 0
    negative_mask = freqs < 0
    positive_freqs = freqs[positive_mask]
    positive_values = values[positive_mask]
    negative_freqs = -freqs[negative_mask]
    negative_values = values[negative_mask]
    output: dict[str, float] = {}
    for idx, (low, high) in enumerate(bands_hz, start=1):
        if high <= low:
            raise ValueError("Cada banda debe cumplir high > low")
        key = f"potencia_banda_{idx}"
        positive_power = _integrate_frequency_interval(
            positive_freqs,
            positive_values,
            low,
            high,
        )
        negative_power = _integrate_frequency_interval(
            negative_freqs,
            negative_values,
            low,
            high,
        )
        output[key] = max(float(positive_power + negative_power), 0.0)
    return output
