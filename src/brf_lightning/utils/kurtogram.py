# kurtogram_utils.py
"""Utility functions for spectral-kurtosis analysis and kurtogram-based band selection.

Revision 2025-06-06
~~~~~~~~~~~~~~~~~~~
* ``max_sk_band`` now guarantees that the returned band sits **strictly inside
  (0, fs/2)** to avoid SciPy ``butter`` errors when SK selects a band that
  touches DC or Nyquist.  New keyword arguments:

  * ``min_freq`` - lower clamp (default 10 Hz)
  * ``max_ratio`` - upper clamp as a fraction of Nyquist (default 0.99)

This change fixes the *“filter critical frequencies must be greater than 0”*
exception observed when processing healthy traces whose SK peak straddles very
low frequencies.
"""
from __future__ import annotations

from typing import Tuple, Optional, Sequence

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

__all__: Sequence[str] = [
    "spectral_kurtosis",
    "max_sk_band",
    "plot_sk",
    "kurtogram",
]

# -----------------------------------------------------------------------------
# 1) Short-time spectral kurtosis (Antoni, 2006) via STFT
# -----------------------------------------------------------------------------

def spectral_kurtosis(
    x: np.ndarray,
    fs: float,
    *,
    nperseg: int = 1024,
    noverlap: Optional[int] = None,
    window: str | tuple | np.ndarray = "hann",
    detrend: str | None = "constant",
    eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Second-order spectral kurtosis curve.

    Returns
    -------
    f, sk : ndarray
        Frequency bins [Hz] and corresponding SK values.
    """
    if x.ndim != 1:
        raise ValueError("`x` must be 1-D array")

    f, _, Zxx = sig.stft(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        padded=False,
        return_onesided=True,
    )

    S2 = np.abs(Zxx) ** 2
    m2 = np.mean(S2, axis=1)
    m4 = np.mean(S2 ** 2, axis=1)
    sk = (m4 / (m2 ** 2 + eps)) - 2.0  # unbiased SK
    return f, sk

# -----------------------------------------------------------------------------
# 2) Highest-SK band selection with safety clamps
# -----------------------------------------------------------------------------

def max_sk_band(
    f: np.ndarray,
    sk: np.ndarray,
    *,
    bandwidth: float,
    min_freq: float = 10.0,
    max_ratio: float = 0.99,
) -> Tuple[float, float]:
    """Return [f_lo, f_hi] with maximal average SK, clamped inside Nyquist.

    The sliding-window average of SK is evaluated over a user-defined
    ``bandwidth``.  The resulting band is then clipped to
    ``[min_freq, max_ratio*Nyquist]`` to prevent zero or >Nyquist critical
    frequencies when designing IIR filters.
    """
    if bandwidth <= 0:
        raise ValueError("`bandwidth` must be positive")

    df = f[1] - f[0]
    win = int(round(bandwidth / df))
    if win < 1:
        raise ValueError("`bandwidth` too small relative to resolution")

    cumsum = np.concatenate(([0.0], np.cumsum(sk)))
    ma = (cumsum[win:] - cumsum[:-win]) / win
    idx_max = int(np.argmax(ma))
    f_lo = f[idx_max]
    f_hi = f[idx_max + win - 1]

    # Safety clamps
    f_lo = max(f_lo, min_freq)
    nyq = f[-1]
    f_hi = min(f_hi, max_ratio * nyq)
    if f_hi <= f_lo:
        # Fallback: centre band at middle of allowed range
        mid = 0.5 * (min_freq + max_ratio * nyq)
        half_bw = 0.5 * bandwidth
        f_lo, f_hi = mid - half_bw, mid + half_bw
    return f_lo, f_hi

# -----------------------------------------------------------------------------
# 3) (unchanged) dyadic kurtogram and plot helper
# -----------------------------------------------------------------------------

def _dyadic_bandpass(x: np.ndarray, order: int, level: int, fs: float) -> np.ndarray:
    nyq = 0.5 * fs
    bw = fs / (2 ** (level + 1))
    f0 = order * bw
    f1 = f0 + bw
    sos = sig.butter(4, [f0 / nyq, f1 / nyq], btype="bandpass", output="sos")
    return sig.sosfiltfilt(sos, x)


def kurtogram(
    x: np.ndarray,
    fs: float,
    *,
    max_level: int = 6,
    eps: float = 1e-10,
):
    levels = list(range(max_level + 1))
    K = []
    for L in levels:
        row = []
        for o in range(2 ** L):
            bp = _dyadic_bandpass(x, o, L, fs)
            row.append(sig.kurtosis(bp, fisher=False, bias=False) + eps)
        K.append(row)
    K = np.array(K)
    Ls, os = np.unravel_index(np.argmax(K), K.shape)
    return K, levels, Ls, os


def plot_sk(f: np.ndarray, sk: np.ndarray, ax: Optional[plt.Axes] = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    ax.plot(f, sk)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Spectral kurtosis")
    ax.set_title("Second-order spectral kurtosis")
    ax.grid(True, ls=":", lw=0.5)
    return ax
