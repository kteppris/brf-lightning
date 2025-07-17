"""
Balanced Resonate-and-Fire helpers
=================================
Implements:
  p_omega(delta, omega)        - Higuchi et al. divergence boundary (Eq. 20)
  choose_bc(delta, omega, safety) - pick a leak bc = p - safety
  bc_from_halfbw(half_bw)      - bandwidth → leak heuristic
  lambda_discrete(bc, omega, d) - exact eigen-value for simulation
  is_stable_exact(bc)          - stability test for the exact update
All functions are NumPy-vectorised.
Higuchi 2024 paper: https://arxiv.org/abs/2402.14603
"""
import numpy as np
import logging

logger = logging.getLogger("lightning.pytorch.core")

# ------------- divergence boundary (Euler criterion) -------------------
def p_omega(delta: float, omega):
    omega = np.asarray(omega, float)
    rad = 1.0 - (delta * omega) ** 2
    if np.any(rad < 0):
        raise ValueError("δ·ω must satisfy δω ≤ 1 in Euler theory")
    return (-1.0 + np.sqrt(rad)) / delta

# ------------- leak selection helpers ---------------------------------
def choose_bc(delta: float, omega, safety: float = 1500.):
    """Pick a leak that is safety [s⁻¹] below the boundary."""
    return p_omega(delta, omega) - safety

def bc_from_halfbw(half_bw: float):
    """Classical Q-mapping:  b ≈ -π·halfBW  Izhikevich 2001"""
    return -np.pi * half_bw

def is_stable_exact(bc):
    """Exact update is stable iff  Re(bc)<0 ."""
    return np.all(np.asarray(bc) < 0.0)

def bc_exact(half_bw=None, safety=1500.0):
    """Leak for exact exponential update (ω-independent)."""
    if half_bw is not None: # Q‑matching shortcut
        return -np.pi * half_bw # Izhikevich 2001 heuristic
    return -safety # just keep it negative

def lambda_discrete(bc, omega, delta):
    """Eigen-value for exact update (always stable if bc<0)."""
    return np.exp((bc + 1j*omega)*delta)

def make_brf_params(fs: float,
                    omega: np.ndarray,
                    half_bw: float | np.ndarray,
                    safety: float = 0.) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (bc, λ) arrays for balanced‑RF neurons.

    Parameters
    ----------
    fs        : sampling rate (Hz)
    omega     : angular carriers (rad s⁻¹)
    half_bw   : scalar or array of -3 dB half-bandwidths (Hz)
    safety    : optional extra damping (s⁻¹) subtracted from bc
    """
    half_bw = np.asarray(half_bw, float)
    if half_bw.size == 1:
        half_bw = np.full_like(omega, half_bw)

    bc  = bc_from_halfbw(half_bw) - safety         # always negative
    lam = lambda_discrete(bc, omega, 1/fs)

    # -------- sanity & log ---------------------------------------------
    assert is_stable_exact(bc).all(), "bc must be < 0 for exact map stability"

    for k, (w, b, l) in enumerate(zip(omega, bc, lam)):
        logger.info(f"[BRF {k:02d}]  ω={w/2/np.pi:7.2f} kHz  "
                 f"bc={b:8.1f} s⁻¹  |λ|={abs(l):.3f}")

    return bc, lam