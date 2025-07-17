"""
BRF‑LIF Backbone (v2.1)
=======================
*   **Theta inside the BRF neuron** – no second hard‑gate.
*   Keeps baseline bias/rms scaler fixes from v2.

Implementation changes
----------------------
1.  Introduce `ThetaBRFCell`, a thin wrapper around `brf_snn.modules.BRFCell`
    that stores a *learnable* `theta` per neuron and passes it to
    `brf_update()`.
2.  `BRFBackboneV2` now instantiates `ThetaBRFCell` and **removes** the
    manual `(u‑theta>=0)` gate.  `z` straight from the cell feeds the LIF.
3.  Hyper‑parameter `theta_init` moved into the cell ctor; no standalone
    `self.theta` in the backbone.

The external API (Lightning config) is unchanged.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from brf_snn.modules import BRFCell, LICell
from brf_snn.functional import StepDoubleGaussianGrad  # for typing only
from brf_lightning.models.backbones.modules.readout_gain import ReadoutGain

# -------------------------------------------------------------
# Utility: rms scaling layer
# -------------------------------------------------------------
class RMSScaler(nn.Module):
    def __init__(self, target_rms: float = 0.3, momentum: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.target_rms = target_rms
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_scale", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (T,B,C)
        rms = x.pow(2).mean().sqrt()
        scale = self.target_rms / (rms + self.eps)
        if not self.training:
            self.running_scale.mul_(self.momentum).add_((1 - self.momentum) * scale)
            scale = self.running_scale
        return x * scale


# -------------------------------------------------------------
# BRF cell with learnable theta
# -------------------------------------------------------------
from brf_snn.modules.rf import brf_update, sustain_osc  # reuse low‑level ops

class ThetaBRFCell(BRFCell):
    """BRFCell that stores per‑neuron `theta` and passes it to `brf_update`."""

    def __init__(
        self,
        *,
        input_size: int,
        layer_size: int,
        theta_init: float = 0.05,
        trainable_theta: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(input_size=input_size, layer_size=layer_size, **kwargs)

        theta = theta_init * torch.ones(layer_size)
        if trainable_theta:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer("theta", theta)

    # ------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        in_sum = self.linear(x)
        z, u, v, q = state
        omega = torch.abs(self.omega)
        b_offset = torch.abs(self.b_offset)
        p_omega = sustain_osc(omega, dt=self.dt)
        b = p_omega - b_offset - q

                # keep `b` for debugging callbacks
        self.b = b.detach()

        z, u, v, q = brf_update(
            x=in_sum,
            u=u,
            v=v,
            q=q,
            b=b,
            omega=omega,
            dt=self.dt,
            theta=self.theta,
        )
        self.z = z
        return z, u, v, q


# -------------------------------------------------------------
# Backbone model
# -------------------------------------------------------------
class BRFBackboneV2(nn.Module):
    """Carrier-tuned BRF ➔ LIF backbone (theta inside BRF)."""

    def __init__(
        self,
        *,
        input_size: int = 1,
        hidden_size: int = 64,
        num_classes: int = 3,
        fs: float = 6_000.0,
        carrier_hz: Optional[List[float]] = None,
        carrier_base_hz: float = 600.0,
        carrier_step_hz: float = 120.0,
        carrier_count: int = 8,
        theta_init: float = 0.05,
        b_offset: float = 0.2,
        baseline_rate: float = 0.0,
        omega_jitter: float = 0.1,
        adaptive_b: bool = True,
        adaptive_omega: bool = True,
        tau_ms: float = 5.0,
        use_rms_scaler: bool = True,
        rms_target: float = 0.3,
        input_gain: float = 1.0,
        post_gain_auto: bool = True,
        post_gain_learnable: bool = False,
        post_gain_target_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.dt = 1.0 / fs
        self.input_gain = input_gain

        # ─── input normaliser ─────────────────────────────
        self.norm = RMSScaler(rms_target) if use_rms_scaler else nn.Identity()

        # ─── carrier list & ω init ───────────────────────
        if carrier_hz is None:
            carrier_hz = [carrier_base_hz + i * carrier_step_hz for i in range(carrier_count)]
        assert hidden_size % len(carrier_hz) == 0, "hidden_size must be multiple of carrier_count"

        base_omega = torch.tensor(carrier_hz) * 2 * torch.pi
        omega_init = base_omega.repeat_interleave(hidden_size // len(carrier_hz))
        omega_init *= torch.empty_like(omega_init).uniform_(1 - omega_jitter, 1 + omega_jitter)

        # ─── BRF layer ────────────────────────────────────
        self.brfc = ThetaBRFCell(
            input_size=input_size + hidden_size,
            layer_size=hidden_size,
            theta_init=theta_init,
            adaptive_omega=adaptive_omega,
            adaptive_omega_a=float(omega_init.min()),
            adaptive_omega_b=float(omega_init.max()),
            adaptive_b_offset=adaptive_b,
            adaptive_b_offset_a=b_offset * 0.5,
            adaptive_b_offset_b=b_offset * 1.5,
            bias=False,
            dt=self.dt,
        )

        self.baseline_rate = baseline_rate

        # ─── LIF read‑out ─────────────────────────────────
        self.readout = LICell(
            input_size=hidden_size,
            layer_size=num_classes,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=tau_ms,
            adaptive_tau_mem_std=0.1 * tau_ms,
            bias=True,
        )

        self.post_gain = ReadoutGain(
            target_std=post_gain_target_std,
            auto_calibrate=post_gain_auto,
            learnable=post_gain_learnable,
        )

    # ------------------------------------------------------
    def forward(self, x: torch.Tensor, speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (T,B,C)
        x = self.norm(x)
        T, B, C_in = x.shape
        device = x.device

        z = torch.zeros(B, self.brfc.layer_size, device=device)
        u = v = q = z.clone()
        u_ro = torch.zeros(B, self.readout.layer_size, device=device)

        baseline = x.new_full((B, C_in), self.baseline_rate)
        logits = []
        for t in range(T):
            inp = torch.cat(((x[t] + baseline) * self.input_gain, z), dim=1)
            z, u, v, q = self.brfc(inp, (z, u, v, q))
            u_ro = self.readout(z, u_ro)
            logits.append(self.post_gain(u_ro))

        return torch.stack(logits)
