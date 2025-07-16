# brf_lightning/models/backbones/minimal_brf.py
import torch
import torch.nn as nn
from brf_snn.modules.rf import RFCell, brf_update, sustain_osc
from brf_snn.modules import LICell       # plain leaky‑integrator read‑out


class FixedBRFCell(RFCell):
    """
    A BRFCell with *fixed* ω & b_offset computed from f0 / Q.
    Nothing gets multiplied by 1/dt - this is the “textbook” form.
    """

    def __init__(
        self,
        *,
        f0_hz: list[float],          # resonance centres in Hz
        Q: float,                    # quality factor (same for all neurons)
        fs: float = 8192.0,          # sampling rate
        input_size: int = 1,         # channels per sample
        bias: bool = False,
        adaptive_b_offset: bool = True,
        adaptive_omega: bool = False,
    ):
        dt = 1.0 / fs
        omega = 2 * torch.pi * torch.tensor(f0_hz) * dt        # (H,)
        b     = -omega / (2 * Q)                               # damping
        p_ω   = sustain_osc(omega, dt=dt)
        b_off = p_ω - b.abs()                                  # stable offset

        super().__init__(
            input_size=input_size,
            layer_size=len(f0_hz),
            omega=omega,
            b_offset=b_off,
            adaptive_omega=adaptive_omega,
            adaptive_b_offset=adaptive_b_offset,
            dt=dt,
            bias=bias,
        )

    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        in_sum = self.linear(x) / self.dt

        z, u, v, q = state

        omega = torch.abs(self.omega)

        p_omega = sustain_osc(omega, dt=self.dt)

        b_offset = torch.abs(self.b_offset)

        # divergence boundary
        b = p_omega - b_offset - q

        z, u, v, q = brf_update(
            x=in_sum,
            u=u,
            v=v,
            q=q,
            b=b,
            omega=omega,
            dt=self.dt,
        )

        return z, u, v, q


class MinimalBRFEncoder(nn.Module):
    """
    One RF layer  ➜  one LICell read-out.
    Suitable to show that resonance spiking alone separates classes.
    """

    def __init__(
        self,
        *,
        f0_hz: list[float],
        Q: float = 5.0,
        fs: float = 8192.0,
        input_size: int = 1,
        num_classes: int = 3,
        readout_tau_mean: float = 20.0,
        readout_tau_std: float = 0.0,
    ):
        super().__init__()

        self.brfc = FixedBRFCell(
            f0_hz=f0_hz,
            Q=Q,
            fs=fs,
            input_size=input_size,
        )

        self.readout = LICell(
            input_size=len(f0_hz),
            layer_size=num_classes,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=readout_tau_mean,
            adaptive_tau_mem_std=readout_tau_std,
            bias=False,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor (T, B, input_size)
        returns logits : Tensor (T, B, num_classes)
        """
        T, B, _ = x.shape
        H = len(self.brfc.omega)

        z = x.new_zeros(B, H)
        u = v = q = z.clone()
        u_ro = x.new_zeros(B, self.readout.layer_size)

        out = []
        for t in range(T):
            z, u, v, q = self.brfc(x[t], (z, u, v, q))
            u_ro = self.readout(z, u_ro)
            out.append(u_ro)

        return torch.stack(out)
