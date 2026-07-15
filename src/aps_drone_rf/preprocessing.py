"""Normalizacion y segmentacion de señales."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def normalize_signal(x: ArrayLike, method: str = "maxabs", eps: float = 1e-12) -> np.ndarray:
    """Normaliza una señal con un metodo simple y reproducible."""

    signal = np.asarray(x)
    if method == "none":
        return signal.copy()
    if method == "maxabs":
        scale = np.max(np.abs(signal))
        return signal / max(float(scale), eps)
    if method == "rms":
        scale = np.sqrt(np.mean(np.abs(signal) ** 2))
        return signal / max(float(scale), eps)
    if method == "zscore":
        centered = signal - np.mean(signal)
        scale = np.std(centered)
        return centered / max(float(scale), eps)
    raise ValueError(f"Metodo de normalizacion no soportado: {method}")


def preprocess_signal(
    x: ArrayLike,
    *,
    remove_mean: bool = True,
    normalization: str = "maxabs",
) -> np.ndarray:
    """Remueve la media y normaliza un registro completo de forma explícita."""

    signal = np.asarray(x).reshape(-1)
    if signal.size == 0:
        raise ValueError("La señal no puede estar vacía")
    if not np.all(np.isfinite(signal)):
        raise ValueError("La señal contiene valores NaN o infinitos")
    if remove_mean:
        signal = signal - np.mean(signal)
    return normalize_signal(signal, method=normalization)


def deterministic_window_starts(
    n_samples: int,
    window_size: int,
    hop_size: int,
    n_windows: int,
) -> np.ndarray:
    """Elige inicios repartidos a lo largo del registro sin usar azar."""

    if n_samples < window_size:
        raise ValueError("El registro es más corto que la ventana")
    if window_size <= 0 or hop_size <= 0 or n_windows <= 0:
        raise ValueError("window_size, hop_size y n_windows deben ser positivos")
    candidates = np.arange(0, n_samples - window_size + 1, hop_size, dtype=np.int64)
    if len(candidates) <= n_windows:
        return candidates
    selected_indices = np.linspace(0, len(candidates) - 1, n_windows, dtype=np.int64)
    return candidates[selected_indices]


def select_representative_window(
    x: ArrayLike,
    window_size: int,
    candidate_windows: int = 20,
) -> tuple[int, np.ndarray]:
    """Elige la ventana no solapada cuyo RMS queda más cerca de la mediana."""

    signal = np.asarray(x).reshape(-1)
    if window_size <= 0 or candidate_windows <= 0:
        raise ValueError("window_size y candidate_windows deben ser positivos")
    available = len(signal) // window_size
    if available == 0:
        raise ValueError("La señal es más corta que la ventana solicitada")
    count = min(candidate_windows, available)
    windows = signal[: count * window_size].reshape(count, window_size)
    rms = np.sqrt(np.mean(np.abs(windows) ** 2, axis=1))
    median_rms = float(np.median(rms))
    index = int(np.argmin(np.abs(rms - median_rms)))
    start = index * window_size
    return start, windows[index].copy()


def sample_signal_windows(
    x: ArrayLike,
    window_size: int,
    hop_size: int,
    n_windows: int,
    *,
    group_id: str,
    label: str,
    part: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Toma ventanas determinísticas sin materializar todas las candidatas."""

    signal = np.asarray(x).reshape(-1)
    starts = deterministic_window_starts(len(signal), window_size, hop_size, n_windows)
    windows = np.stack([signal[start : start + window_size] for start in starts])
    metadata = pd.DataFrame(
        [
            {
                "window_id": f"{group_id}_{part.lower()}:{index}",
                "group_id": group_id,
                "label": label,
                "part": part.upper(),
                "start": int(start),
                "stop": int(start + window_size),
            }
            for index, start in enumerate(starts)
        ]
    )
    return windows, metadata


def segment_signal(
    x: ArrayLike,
    window_size: int,
    hop_size: int | None = None,
    *,
    group_id: str,
    label: str,
    window_id_prefix: str | None = None,
    drop_last: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Segmenta una señal 1D y conserva `group_id` en todas las ventanas."""

    signal = np.asarray(x)
    if signal.ndim != 1:
        signal = signal.reshape(-1)
    if window_size <= 0:
        raise ValueError("window_size debe ser positivo")
    hop = window_size if hop_size is None else hop_size
    if hop <= 0:
        raise ValueError("hop_size debe ser positivo")
    prefix = group_id if window_id_prefix is None else window_id_prefix
    if len(signal) < window_size:
        if drop_last:
            return np.empty((0, window_size), dtype=signal.dtype), pd.DataFrame(
                columns=["window_id", "group_id", "label", "start", "stop"]
            )
        padded = np.zeros(window_size, dtype=signal.dtype)
        padded[: len(signal)] = signal
        metadata = pd.DataFrame(
            [
                {
                    "window_id": f"{prefix}:0",
                    "group_id": group_id,
                    "label": label,
                    "start": 0,
                    "stop": len(signal),
                }
            ]
        )
        return padded[None, :], metadata

    starts = list(range(0, len(signal) - window_size + 1, hop))
    if not drop_last and starts[-1] + window_size < len(signal):
        starts.append(len(signal) - window_size)

    windows = np.stack([signal[start : start + window_size] for start in starts])
    metadata = pd.DataFrame(
        [
            {
                "window_id": f"{prefix}:{idx}",
                "group_id": group_id,
                "label": label,
                "start": int(start),
                "stop": int(start + window_size),
            }
            for idx, start in enumerate(starts)
        ]
    )
    return windows, metadata
