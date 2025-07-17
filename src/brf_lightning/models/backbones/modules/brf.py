import torch
import torch.nn as nn
from brf_snn.modules.rf import BRFCell, brf_update, sustain_osc


# brf_snn/modules/balanced_rf_cell.py
# ---------------------------------------------------------------
# Dependencies: torch, numpy, LinearMask (unchanged),
#               StepDoubleGaussianGrad, bc_from_halfbw
# ---------------------------------------------------------------
import torch, numpy as np
from brf_snn.functional import StepDoubleGaussianGrad
from brf_snn.modules.linear_layer import LinearMask
from brf_lightning.utils.brf_utils import bc_from_halfbw


def sustain_osc(omega: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
    return (-1 + torch.sqrt(1 - torch.square(dt * omega))) / dt
class BalancedRFCell(torch.nn.Module):
    """
    Unified balanced RF layer with an optional Euler safeguard.

    Parameters
    ----------
    input_size      : int          full input dimension (external + recurrent)
    layer_size      : int          number of BRF neurons
    fs              : float        sampling rate in Hz  (dt = 1/fs)
    carriers_khz    : list[float]  resonance centres in kHz  (len == layer_size)
    half_bw_hz      : list[float]  −3 dB half‑band‑widths in Hz (same length)
    use_euler       : bool         True → forward‑Euler with p_ω correction
                                   False → exact exp‑map (recommended)
    adaptive_omega  : bool         if True, ω is trainable (`nn.Parameter`)
    adaptive_b_off  : bool         if True, leak magnitude is trainable
    pruning         : bool         replace dense Linear by LinearMask
    mask_prob       : float        probability of masking recurrent weights
    bias            : bool         include bias term in Linear/LinearMask
    """

    def __init__(self,
                 input_size      : int,
                 layer_size      : int,
                 fs              : float,
                 carriers_khz    : list[float],
                 half_bw_hz      : list[float],
                 *,
                 use_euler       : bool = False,
                 adaptive_omega  : bool = True,
                 adaptive_b_off  : bool = True,
                 pruning         : bool = False,
                 mask_prob       : float = 0.0,
                 bias            : bool = False):

        super().__init__()

        self.hidden_size = layer_size
        self.layer_size = layer_size

        # ─── weight layer (dense or masked) ────────────────────────────
        if pruning:
            self.linear = LinearMask(
                in_features = input_size,
                out_features= layer_size,
                bias        = bias,
                mask_prob   = mask_prob,
                lbd         = input_size - layer_size,
                ubd         = input_size)
        else:
            self.linear = torch.nn.Linear(input_size, layer_size, bias=bias)
            torch.nn.init.xavier_uniform_(self.linear.weight)

        # ─── neuron constants from kurtogram spec ─────────────────────
        assert layer_size == len(carriers_khz) == len(half_bw_hz), \
            "layer_size, carriers_khz, half_bw_hz must match"

        omega_init = 2 * np.pi * np.asarray(carriers_khz) * 1e3   # rad/s
        bc_init    = bc_from_halfbw(np.asarray(half_bw_hz))       # negative

        # register as parameters or buffers
        if adaptive_omega:
            self.omega = torch.nn.Parameter(torch.tensor(omega_init,
                                                         dtype=torch.float32))
        else:
            self.register_buffer('omega', torch.tensor(omega_init,
                                                       dtype=torch.float32))

        if adaptive_b_off:
            # store magnitude; sign flipped in forward
            self.b_offset = torch.nn.Parameter(torch.tensor(-bc_init,
                                                            dtype=torch.float32).abs())
        else:
            self.register_buffer('b_offset', torch.tensor(-bc_init,
                                                          dtype=torch.float32).abs())

        # ─── simulation parameters ────────────────────────────────────
        self.dt        = 1.0 / fs
        self.use_euler = use_euler
        self.theta     = 1.0         # spike threshold
        self.gamma_q   = 0.9         # refractory decay factor

    # ─── forward step ────────────────────────────────────────────────
    def forward(self, x: torch.Tensor,
                state: tuple[torch.Tensor, torch.Tensor,
                             torch.Tensor, torch.Tensor]):
        """
        state = (z_prev, u, v, q)
        Returns same 4-tuple with updated tensors.
        """
        in_sum = self.linear(x)
        z_prev, u, v, q = state

        omega = torch.abs(self.omega)                 # ensure ≥ 0
        b_mag = torch.abs(self.b_offset)              # positive magnitude

        if self.use_euler:
            # Requires dt*omega ≤ 1 for stability
            p_omega = sustain_osc(omega, self.dt)
            b       = p_omega - b_mag - q
        else:
            # Exact exp‑map: negative leak + refractory
            b       = -b_mag - q 

        # ─ membrane Euler step (same as original brf_update) ─────────
        u_new = u + b * u * self.dt - omega * v * self.dt + in_sum * self.dt
        v     = v + omega * u * self.dt + b * v * self.dt

        z = StepDoubleGaussianGrad.apply(u_new - self.theta - q)
        q = q * self.gamma_q + z                      # refractory accumulation

        return z, u_new, v, q

class SpeedAdaptiveBRFCell(nn.Module):
    """
    Speed-scaled BRFCell that modulates resonance frequencies by shaft speed (Hz),
    and computes time-varying BRF parameters per sample.

    This version is fully compatible with the proper BRF formulation (resonance + spiking)
    and mirrors the behavior of BRFCellWithInputScaling.
    """

    def __init__(
        self,
        input_size: int,
        layer_size: int,
        fault_orders: list[float],
        Q: float = 2.0,
        fs: float = 8192.0,
        bias: bool = False,
        adaptive_b_offset: bool = True,
        adaptive_b_offset_a: float = 0.005,
        adaptive_b_offset_b: float = 0.05,
    ):
        super().__init__()

        self.input_size = input_size
        self.layer_size = layer_size
        self.Q = Q
        self.fs = fs
        self.dt = 1.0 / fs
        self.bias = bias
        self.adaptive_b_offset = adaptive_b_offset
        self.adaptive_b_offset_a = adaptive_b_offset_a
        self.adaptive_b_offset_b = adaptive_b_offset_b

        self.linear = nn.Linear(input_size, layer_size, bias=bias)

        self.fault_orders = nn.Parameter(
            torch.tensor(fault_orders, dtype=torch.float32).view(1, -1),
            requires_grad=False,
        )
        self.omega = torch.tensor(1.0)  # dummy to satisfy is_brf()

    def compute_dynamic_params(self, speed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute time-varying omega and b from fault orders and speed.

        Parameters
        ----------
        speed : Tensor (B, 1) [Hz]

        Returns
        -------
        omega : Tensor (B, H)
        b     : Tensor (B, H)
        """
        B = speed.size(0)
        fault_orders = self.fault_orders.to(speed.device)  # (1, H)
        speed = speed.view(B, 1)                            # (B, 1)

        f0 = fault_orders * speed                           # (B, H)
        omega = 2 * torch.pi * f0 * self.dt                 # (B, H)

        p_omega = sustain_osc(omega, dt=self.dt)

        if self.adaptive_b_offset:
            b_offset = self.adaptive_b_offset_a + self.adaptive_b_offset_b * omega.abs()
        else:
            b_offset = torch.zeros_like(omega)

        b = p_omega - b_offset  # divergence boundary
        self.omega_dynamic = omega.mean(0).detach()  # shape: [H]
        return omega, b

    def forward(
        self,
        x: torch.Tensor,                   # (B, input_size)
        speed: torch.Tensor,              # (B, 1)
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        z, u, v, q = state

        in_sum = self.linear(x)
        omega, b = self.compute_dynamic_params(speed)       # per-sample parameters
        b = b - q                                            # apply refractory inhibition

        z, u, v, q = brf_update(x=in_sum, u=u, v=v, q=q, b=b, omega=omega, dt=self.dt)

        return z, u, v, q