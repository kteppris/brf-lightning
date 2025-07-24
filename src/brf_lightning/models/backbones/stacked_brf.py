import torch
import torch.nn as nn

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
        carrier_brf: nn.Module,
        envelope_brf: nn.Module,
        num_classes,
        out_adaptive_tau_mem_mean: float = 20.,
        out_adaptive_tau_mem_std: float = 5.,
        amplitude_gated: bool = True
    ):
        super().__init__()

        # First BRF layer (resonance detection)
        self.brfc = carrier_brf

        # Second BRF layer (fault-frequency demodulation)
        self.brfe = envelope_brf

        # Readout layer (LICell)
        self.readout = LICell(
            input_size=envelope_brf.layer_size,
            layer_size=num_classes,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=out_adaptive_tau_mem_std,
            bias=False,
        )

        self.amplitude_gated = amplitude_gated

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

        z1 = x.new_zeros(batch, self.brfc.layer_size)
        u1 = v1 = q1 = z1.clone()

        z2 = x.new_zeros(batch, self.brfe.layer_size)
        u2 = v2 = q2 = z2.clone()

        u_ro = x.new_zeros(batch, self.readout.layer_size)
        out = []

        for t in range(t_steps):
            z1, u1, v1, q1 = self.brfc(x[t], (z1, u1, v1, q1))
            if self.amplitude_gated:
                x_mod = x[t] * z1  # spike gated signal
                z2, u2, v2, q2 = self.brfe(x_mod, (z2, u2, v2, q2))
            else:
                z2, u2, v2, q2 = self.brfe(z1, (z2, u2, v2, q2))
            u_ro = self.readout(z2, u_ro)
            out.append(u_ro)

        return torch.stack(out)
