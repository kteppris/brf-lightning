import torch
import torch.nn as nn

from brf_lightning.models.backbones.modules.brf import (
    BRFCellWithInputScaling,
    SpeedAdaptiveBRFCell,
)
from brf_lightning.models.backbones.modules.normalize import OnlineNormalSuppression
from brf_snn.modules import LICell


class BRFStackedEncoder(nn.Module):
    """
    Stacked BRF Encoder with:
    - Optional normalization (per time step) based on normal data statistics
    - First BRF layer: fixed ω (Hz domain), input scaling
    - Second BRF layer: speed-scaled ω (order domain)
    - LICell readout
    """

    def __init__(
        self,
        *,
        input_size: int,
        num_classes: int,
        fs: float = 8192.0,
        q_carrier: float = 3.0,
        carrier_band_hz: list[float] = [1800.0, 2100.0, 2400.0],
        q_brfc2: float = 1.0, 
        fault_orders: list[float] = [4.5, 5.5, 6.5],
        # Readout config
        readout_tau_mean: float = 20.0,
        readout_tau_std: float = 0.0,
        # Optional normalization
        normalize_input: bool = False,
        norm_mean: float = 0.000034,
        norm_std: float = 0.041865,
        norm_amplify: bool = True,
    ):
        super().__init__()

        dt = 1 / fs
        self.fs = fs
        self.dt = dt

        self.normalize_input = normalize_input
        if normalize_input:
            self.normalizer = OnlineNormalSuppression(
                mean=torch.tensor([norm_mean]),
                std=torch.tensor([norm_std]),
                amplify_faults=norm_amplify,
            )
        else:
            self.normalizer = nn.Identity()

        # First BRF layer (resonance detection)
        self.hidden_size = len(carrier_band_hz)
        self.brfc = BRFCellWithInputScaling(
            input_size=input_size,
            layer_size=self.hidden_size,
            f0s=carrier_band_hz,
            Q=q_carrier,
            fs=fs,
            adaptive_omega=False,
            adaptive_b_offset=True
        )

        # Second BRF layer (fault-frequency demodulation, speed-scaled)
        self.fault_size = len(fault_orders)
        self.brfc2 = SpeedAdaptiveBRFCell(
            input_size=self.hidden_size,
            layer_size=self.fault_size,
            fault_orders=fault_orders,
            Q=q_brfc2,
            fs=fs,
            adaptive_b_offset=True,
            adaptive_b_offset_a=0.02,
            adaptive_b_offset_b=0.10
        )

        # Readout layer (LICell)
        self.readout = LICell(
            input_size=self.fault_size,
            layer_size=num_classes,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=readout_tau_mean,
            adaptive_tau_mem_std=readout_tau_std,
            bias=False,
        )

    def forward(self, x: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (T, B, input_size)
        speed : Tensor (B, 1) in Hz

        Returns
        -------
        Tensor (T, B, num_classes)
        """
        t_steps, batch, _ = x.shape

        z1 = x.new_zeros(batch, self.hidden_size)
        u1 = v1 = q1 = z1.clone()

        z2 = x.new_zeros(batch, self.fault_size)
        u2 = v2 = q2 = z2.clone()

        u_ro = x.new_zeros(batch, self.readout.layer_size)
        out = []

        for t in range(t_steps):
            x_t = self.normalizer(x[t])
            z1, u1, v1, q1 = self.brfc(x_t, (z1, u1, v1, q1))
            x_mod = x_t * z1  # amplitude-gated signal for fault detection
            z2, u2, v2, q2 = self.brfc2(x_mod, speed, (z2, u2, v2, q2))
            u_ro = self.readout(z2, u_ro)
            out.append(u_ro)

        return torch.stack(out)
