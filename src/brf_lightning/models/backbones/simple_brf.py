from typing import Optional, Tuple
import torch
import torch.nn as nn
from brf_snn.modules import BRFCell, LICell

from .modules.readout_gain import ReadoutGain


class BRFBackbone(nn.Module):
    """
    BRF encoder followed by a leaky-integrator read-out and an optional
    post-gain scalar.

    All parameters can be set from Lightning-CLI/YAML.  Setting
    ``post_gain_auto = false`` recreates the exact behaviour of the
    previous backbone.
    """

    def __init__(
        self,
        *,
        input_size: int = 1,
        hidden_size: int = 64,
        num_classes: int = 3,
        dt: float = 1.0 / 8192,
        adaptive_omega_a: float = 0.8,
        adaptive_omega_b: float = 2.3,
        adaptive_b_offset_a: float = 0.2,
        adaptive_b_offset_b: float = 1.0,
        readout_tau_mean: float = 20.0,
        readout_tau_std: float = 5.0,
    ) -> None:
        super().__init__()

        self.brfc = BRFCell(
            input_size=input_size + hidden_size,
            layer_size=hidden_size,
            adaptive_omega=True,
            adaptive_omega_a=adaptive_omega_a,
            adaptive_omega_b=adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=adaptive_b_offset_a,
            adaptive_b_offset_b=adaptive_b_offset_b,
            bias=False,
            dt=dt,
        )

        self.readout = LICell(
            input_size=hidden_size,
            layer_size=num_classes,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=readout_tau_mean,
            adaptive_tau_mem_std=readout_tau_std,
            bias=True,
        )


    # --------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        speed: Optional[torch.Tensor] = None,   # placeholder
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (T, B, C_in)
            Time-major input tensor.

        Returns
        -------
        logits : (T, B, num_classes)
            Unnormalised class scores.
        """
        t_steps, batch, _ = x.shape

        z = x.new_zeros(batch, self.brfc.layer_size)
        u = v = q = z.clone()
        u_ro = x.new_zeros(batch, self.readout.layer_size)

        logits = []

        for t in range(t_steps):
            z, u, v, q = self.brfc(torch.cat((x[t], z), 1), (z, u, v, q))
            u_ro = self.readout(z, u_ro)
            logits.append(self.post_gain(u_ro))

        return torch.stack(logits)
