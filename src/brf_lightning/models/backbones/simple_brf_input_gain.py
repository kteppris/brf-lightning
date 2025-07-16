# file: brf_lightning/models/backbones/brf_backbone.py
from typing import Optional, Tuple

import torch
import torch.nn as nn

from brf_snn.modules import LICell
from .modules.scaled_brf import ScaledBRFCell
from .simple_brf import ReadoutGain


class BRFBackboneScaledInput(nn.Module):
    """
    Two–layer SNN backbone

    ┌─ ScaledBRFCell (encoder, T × B × C_in → T × B × H)  
    └─ LICell        (read-out, integrates spikes)  
       ↑ optional scalar ReadoutGain for automatic logit scale

    All hyper-parameters are exposed to Lightning-CLI.
    """

    def __init__(
        self,
        *,
        # input / hidden / output sizes
        input_size: int = 1,
        hidden_size: int = 64,
        num_classes: int = 3,
        # neuron dynamics
        dt: float = 1.0 / 8192,
        adaptive_omega_a: float = 0.8,
        adaptive_omega_b: float = 2.3,
        adaptive_b_offset_a: float = 0.2,
        adaptive_b_offset_b: float = 1.0,
        # encoder scaling
        weight_gain: float = 2.0,
        input_gain_init: float = 10.0,
        learn_input_gain: bool = True,
        # read-out dynamics
        readout_tau_mean: float = 20.0,
        readout_tau_std: float = 5.0,
        # post-gain
        post_gain_auto: bool = True,
        post_gain_learnable: bool = False,
        post_gain_target_std: float = 1.0,
    ) -> None:
        super().__init__()

        self.brfc = ScaledBRFCell(
            input_size=input_size + hidden_size,
            layer_size=hidden_size,
            adaptive_omega=True,
            adaptive_omega_a=adaptive_omega_a,
            adaptive_omega_b=adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=adaptive_b_offset_a,
            adaptive_b_offset_b=adaptive_b_offset_b,
            dt=dt,
            weight_gain=weight_gain,
            input_gain_init=input_gain_init,
            learn_input_gain=learn_input_gain,
            bias=False,
        )

        self.readout = LICell(
            input_size=hidden_size,
            layer_size=num_classes,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=readout_tau_mean,
            adaptive_tau_mem_std=readout_tau_std,
            bias=True,
        )

        self.post_gain = ReadoutGain(
            target_std=post_gain_target_std,
            auto_calibrate=post_gain_auto,
            learnable=post_gain_learnable,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                       # (T, B, C_in)
        speed: Optional[torch.Tensor] = None,  # not used yet
    ) -> torch.Tensor:
        t_steps, batch, _ = x.shape

        z = x.new_zeros(batch, self.brfc.layer_size)
        u = v = q = z.clone()
        u_ro = x.new_zeros(batch, self.readout.layer_size)

        logits: list[torch.Tensor] = []

        for t in range(t_steps):
            z, u, v, q = self.brfc(torch.cat((x[t], z), dim=1), (z, u, v, q))
            u_ro = self.readout(z, u_ro)
            logits.append(self.post_gain(u_ro))

        return torch.stack(logits)            # (T, B, num_classes)
