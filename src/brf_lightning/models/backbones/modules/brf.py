import torch
import torch.nn as nn
from brf_snn.modules.rf import BRFCell, brf_update, sustain_osc

class BRFCellWithInputScaling(BRFCell):
    """
    BRFCell variant that boosts input by 1/dt to match NumPy-style signal injection.

    This compensates for the Euler scaling typically applied to the input signal,
    ensuring that a fixed-amplitude sinusoidal input produces consistent energy
    regardless of `dt`.
    """

    def __init__(
        self,
        *,
        input_size: int,
        layer_size: int,
        f0s: list[float],
        Q: float,
        fs: float,
        adaptive_omega: bool = False,
        adaptive_omega_a: float = 0.005,
        adaptive_omega_b: float = 0.05,
        adaptive_b_offset: bool = True,
        adaptive_b_offset_a: float = 0.005,
        adaptive_b_offset_b: float = 0.05,
    ):
        dt = 1 / fs
        omega = 2 * torch.pi * torch.tensor(f0s) * dt
        b = -omega / (2 * Q)
        from brf_snn.modules.rf import sustain_osc
        p_omega = sustain_osc(omega, dt=dt)
        b_offset = p_omega - b.abs()

        super().__init__(
            input_size=input_size,
            layer_size=layer_size,
            omega=omega,
            b_offset=b_offset,
            adaptive_omega=adaptive_omega,
            adaptive_omega_a=adaptive_omega_a,
            adaptive_omega_b=adaptive_omega_b,
            adaptive_b_offset=adaptive_b_offset,
            adaptive_b_offset_a=adaptive_b_offset_a,
            adaptive_b_offset_b=adaptive_b_offset_b,
            dt=dt,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, u, v, q = state
        in_sum = self.linear(x) / self.dt
        omega = torch.abs(self.omega)
        b_offset = torch.abs(self.b_offset)
        p_omega = sustain_osc(omega, dt=self.dt)
        b = p_omega - b_offset - q
        z, u, v, q = brf_update(x=in_sum, u=u, v=v, q=q, b=b, omega=omega, dt=self.dt)
        return z, u, v, q


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