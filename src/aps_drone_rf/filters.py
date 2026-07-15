"""Diseño y aplicacion de filtros digitales FIR/IIR."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal


def _pass_zero(kind: str) -> bool | str:
    normalized = kind.lower()
    if normalized in {"lowpass", "low"}:
        return True
    if normalized in {"highpass", "high"}:
        return False
    if normalized in {"bandpass", "passband"}:
        return False
    if normalized in {"bandstop", "stopband", "notch"}:
        return True
    raise ValueError(f"Tipo de filtro no soportado: {kind}")


def design_fir_filter(
    kind: str,
    fs_hz: float,
    cutoff_hz: float | tuple[float, float],
    numtaps: int = 101,
    window: str = "hamming",
) -> tuple[np.ndarray, np.ndarray]:
    """Disena un FIR con ventana usando `scipy.signal.firwin`."""

    b = signal.firwin(
        numtaps=numtaps,
        cutoff=cutoff_hz,
        fs=fs_hz,
        window=window,
        pass_zero=_pass_zero(kind),
    )
    a = np.array([1.0])
    return b, a


def design_iir_filter(
    kind: str,
    fs_hz: float,
    cutoff_hz: float | tuple[float, float],
    order: int = 4,
    ftype: str = "butter",
) -> tuple[np.ndarray, np.ndarray]:
    """Disena un IIR basico. Por defecto usa Butterworth."""

    kind_map = {
        "low": "lowpass",
        "lowpass": "lowpass",
        "high": "highpass",
        "highpass": "highpass",
        "bandpass": "bandpass",
        "passband": "bandpass",
        "bandstop": "bandstop",
        "stopband": "bandstop",
        "notch": "bandstop",
    }
    btype = kind_map.get(kind.lower())
    if btype is None:
        raise ValueError(f"Tipo de filtro no soportado: {kind}")
    b, a = signal.iirfilter(order, cutoff_hz, fs=fs_hz, btype=btype, ftype=ftype, output="ba")
    return b, a


def frequency_response(
    b: ArrayLike,
    a: ArrayLike | float = 1.0,
    fs_hz: float = 1.0,
    worN: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula respuesta en frecuencia compleja."""

    freqs, response = signal.freqz(b, a, worN=worN, fs=fs_hz)
    return freqs, response


def zeros_and_poles(b: ArrayLike, a: ArrayLike | float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Obtiene ceros y polos de H(z) a partir de sus coeficientes."""

    b_arr = np.asarray(b, dtype=float)
    a_arr = np.asarray([a] if np.isscalar(a) else a, dtype=float)
    zeros, poles, _ = signal.tf2zpk(b_arr, a_arr)
    return zeros, poles


def transfer_function_at_z(
    b: ArrayLike, a: ArrayLike | float, z: ArrayLike | complex
) -> np.ndarray:
    """Evalúa H(z)=sum(b_k z^-k) / sum(a_k z^-k)."""

    b_arr = np.asarray(b, dtype=float)
    a_arr = np.asarray([a] if np.isscalar(a) else a, dtype=float)
    z_arr = np.asarray(z, dtype=complex)
    if np.any(np.isclose(z_arr, 0.0)):
        raise ValueError("H(z) no se puede evaluar en z = 0 con esta forma")

    z_inv = 1.0 / z_arr
    numerator = np.polynomial.polynomial.polyval(z_inv, b_arr)
    denominator = np.polynomial.polynomial.polyval(z_inv, a_arr)
    return numerator / denominator


def response_on_unit_circle(
    b: ArrayLike, a: ArrayLike | float, omega_rad: ArrayLike | float
) -> np.ndarray:
    """Evalúa H(z) sobre z=e^(j omega), equivalente a H(e^(j omega))."""

    omega_arr = np.asarray(omega_rad, dtype=float)
    return transfer_function_at_z(b, a, np.exp(1j * omega_arr))


def transfer_function_expression(
    b: ArrayLike, a: ArrayLike | float = 1.0, symbol: str = "z"
) -> str:
    """Devuelve H(z) escrita con potencias negativas para mostrar en el informe."""

    b_arr = np.asarray(b, dtype=float)
    a_arr = np.asarray([a] if np.isscalar(a) else a, dtype=float)

    def polynomial_text(coefficients: np.ndarray) -> str:
        terms = []
        for index, coefficient in enumerate(coefficients):
            if np.isclose(coefficient, 0.0):
                continue
            power = "" if index == 0 else f" {symbol}^(-{index})"
            terms.append(f"({coefficient:.6g}){power}")
        return " + ".join(terms) if terms else "0"

    return f"H({symbol}) = [{polynomial_text(b_arr)}] / [{polynomial_text(a_arr)}]"


def apply_filter(
    x: ArrayLike, b: ArrayLike, a: ArrayLike | float = 1.0, zero_phase: bool = True
) -> np.ndarray:
    """Aplica un filtro con `filtfilt` cuando hay muestras suficientes."""

    signal_x = np.asarray(x)
    b_arr = np.asarray(b)
    a_arr = np.asarray([a] if np.isscalar(a) else a)
    min_len = 3 * max(len(a_arr), len(b_arr))
    if zero_phase and len(signal_x) > min_len:
        return signal.filtfilt(b_arr, a_arr, signal_x)
    return signal.lfilter(b_arr, a_arr, signal_x)


def is_iir_stable(a: ArrayLike, tol: float = 0.0) -> bool:
    """Verifica estabilidad BIBO de un IIR a partir de polos dentro del circulo unitario."""

    if tol < 0:
        raise ValueError("tol debe ser no negativo")
    a_arr = np.asarray(a, dtype=float)
    if a_arr.size <= 1:
        return True
    poles = np.roots(a_arr)
    return bool(np.all(np.abs(poles) < 1.0 - tol))
