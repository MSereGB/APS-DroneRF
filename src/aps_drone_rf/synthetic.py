"""Señales sinteticas para verificar metodos de APS con casos controlados."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def time_axis(duration_s: float, fs_hz: float) -> np.ndarray:
    """Devuelve un eje temporal discreto de duracion `duration_s`."""

    if duration_s <= 0:
        raise ValueError("duration_s debe ser positivo")
    if fs_hz <= 0:
        raise ValueError("fs_hz debe ser positivo")
    n_samples = int(round(duration_s * fs_hz))
    return np.arange(n_samples, dtype=float) / fs_hz


def pure_tone(
    freq_hz: float,
    fs_hz: float,
    duration_s: float,
    amplitude: float = 1.0,
    phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera una senoidal pura."""

    t = time_axis(duration_s, fs_hz)
    x = amplitude * np.sin(2.0 * np.pi * freq_hz * t + phase_rad)
    return t, x


def sum_of_tones(
    freqs_hz: ArrayLike,
    fs_hz: float,
    duration_s: float,
    amplitudes: ArrayLike | None = None,
    phases_rad: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera una suma de senoidales."""

    freqs = np.asarray(freqs_hz, dtype=float)
    if amplitudes is None:
        amps = np.ones_like(freqs)
    else:
        amps = np.asarray(amplitudes, dtype=float)
    if phases_rad is None:
        phases = np.zeros_like(freqs)
    else:
        phases = np.asarray(phases_rad, dtype=float)
    if not (freqs.shape == amps.shape == phases.shape):
        raise ValueError("freqs_hz, amplitudes y phases_rad deben tener la misma forma")

    t = time_axis(duration_s, fs_hz)
    x = np.zeros_like(t)
    for freq, amp, phase in zip(freqs, amps, phases, strict=True):
        x += amp * np.sin(2.0 * np.pi * freq * t + phase)
    return t, x


def amplitude_modulated(
    carrier_hz: float,
    mod_hz: float,
    fs_hz: float,
    duration_s: float,
    modulation_index: float = 0.5,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera una señal AM simple para conectar muestreo y contenido espectral."""

    if not 0 <= modulation_index <= 1:
        raise ValueError("modulation_index debe estar entre 0 y 1")
    t = time_axis(duration_s, fs_hz)
    envelope = 1.0 + modulation_index * np.sin(2.0 * np.pi * mod_hz * t)
    carrier = np.sin(2.0 * np.pi * carrier_hz * t)
    return t, amplitude * envelope * carrier


def white_noise(n_samples: int, sigma: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Genera ruido blanco gaussiano."""

    if n_samples <= 0:
        raise ValueError("n_samples debe ser positivo")
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=sigma, size=n_samples)


def add_awgn(x: ArrayLike, snr_db: float, seed: int | None = None) -> np.ndarray:
    """Agrega ruido blanco para obtener una SNR aproximada en dB."""

    signal = np.asarray(x)
    power = np.mean(np.abs(signal) ** 2)
    if power <= 0:
        raise ValueError("La potencia de la señal debe ser positiva")
    noise_power = power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def quantize_uniform(x: ArrayLike, bits: int, full_scale: float | None = None) -> np.ndarray:
    """Cuantiza una señal real con cuantizacion uniforme simetrica."""

    if bits < 2:
        raise ValueError("bits debe ser al menos 2")
    signal = np.asarray(x, dtype=float)
    scale = float(np.max(np.abs(signal)) if full_scale is None else full_scale)
    if scale <= 0:
        return np.zeros_like(signal)
    levels = 2**bits
    clipped = np.clip(signal / scale, -1.0, 1.0)
    quantized = np.round((clipped + 1.0) * (levels - 1) / 2.0)
    restored = (2.0 * quantized / (levels - 1)) - 1.0
    return restored * scale
