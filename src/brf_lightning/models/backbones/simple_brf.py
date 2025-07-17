# my_brf_model.py
from __future__ import annotations
import logging
from typing import Sequence

import torch
import torch.nn as nn

from brf_snn.modules.linear_layer import LinearMask
from brf_snn.functional import StepDoubleGaussianGrad 
from brf_lightning.utils.brf_utils import bc_from_halfbw, lambda_discrete

logger = logging.getLogger("lightning.pytorch.core")


class SimpleBRFNet(nn.Module):
    """
    One-layer balanced RF encoder with fully exposed hyper-parameters
    and automatic stability sanity-checks.
    """

    def __init__(
        self,
        input_size: int,
        layer_size: int,
        fs: float = 8192.0, # sampling rate  Hz
        carriers_khz: Sequence[float] | None = None, # centre freqs kHz
        half_bw: Sequence[float] | None = None, # −3 dB half‑band (Hz)
        safety: float = 1500.0, # extra leak margin s⁻¹
        mask_prob: float = 0.0,
        pruning: bool = False,
        bias: bool = False,
        dt: float | None = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.layer_size = layer_size
        self.fs = fs
        self.dt = (1.0 / fs) if dt is None else dt

        if pruning:
            self.linear = LinearMask(
                in_features=input_size,
                out_features=layer_size,
                bias=bias,
                mask_prob=mask_prob,
                lbd=input_size - layer_size,
                ubd=input_size,
            )
        else:
            self.linear = nn.Linear(input_size, layer_size, bias=bias)
            nn.init.xavier_uniform_(self.linear.weight)

        if carriers_khz is None:
            raise ValueError("carriers_khz must be provided")

        if len(carriers_khz) != layer_size:
            raise ValueError("len(carriers_khz) must equal layer_size")

        self.register_buffer(
            "omega",
            torch.tensor([2 * torch.pi * 1e3 * f for f in carriers_khz],
                         dtype=torch.float32),
        )

        if half_bw is not None:
            if len(half_bw) == 1:
                bc = bc_from_halfbw(half_bw[0]) * torch.ones(layer_size)
            elif len(half_bw) == layer_size:
                bc = torch.tensor([bc_from_halfbw(b) for b in half_bw])
            else:
                raise ValueError("half_bw must have length 1 or layer_size")
        else:
            # negative constant leak using safety margin
            bc = -safety * torch.ones(layer_size)

        self.register_buffer("bc", bc)

        lam = lambda_discrete(self.bc, self.omega, self.dt)

        for idx, (w, b, l) in enumerate(zip(self.omega, self.bc, lam)):
            ok_mag = abs(l) < 1.0
            ok_sign = b < 0
            logger.info(
                f"[BRF-init] idx={idx:02d} "
                f"ω={w/2/torch.pi/1e3:.3f} kHz  "
                f"b_c={b:.1f} s⁻¹  |λ|={abs(l):.3f}  "
                f"{'OK' if (ok_mag and ok_sign) else 'UNSTABLE'}"
            )
            if not (ok_mag and ok_sign):
                raise ValueError(
                    f"Neuron {idx} unstable: |λ|={abs(l):.3f}, b_c={b:.1f}"
                )

        self.register_buffer("u", torch.zeros(layer_size))
        self.register_buffer("v", torch.zeros(layer_size))
        self.register_buffer("q", torch.zeros(layer_size))

    def brf_update(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # aliases for speed
        dt = self.dt
        u, v, q = self.u, self.v, self.q
        omega, bc = self.omega, self.bc

        # membrane update
        u_ = u + bc * u * dt - omega * v * dt + x * dt
        v = v + omega * u * dt + bc * v * dt

        # spike generation
        z = StepDoubleGaussianGrad.apply(u_ - 1.0 - q)  # θ=1

        # refractory
        q = 0.9 * q + z

        # cache state
        self.u.copy_(u_)
        self.v.copy_(v)
        self.q.copy_(q)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_size)
        returns spike tensor (batch, layer_size)
        """
        inj = self.linear(x)
        return self.brf_update(inj)
