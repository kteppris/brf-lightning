import torch
import torch.nn as nn
from brf_snn.modules import LICell
from brf_snn.modules.rf import sustain_osc, brf_update
from brf_snn.modules import BRFCell


class BRFCellBoostedInput(BRFCell):
    """
    Subclass of BRFCell with boosted input (x / dt) to match NumPy behavior.
    Only the forward() method is overridden.
    """
    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, u, v, q = state
        in_sum = self.linear(x) / self.dt

        omega = torch.abs(self.omega)
        p_omega = sustain_osc(omega)
        b_offset = torch.abs(self.b_offset)
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


class BRFStackedEncoderWithSpeed(nn.Module):
    """
    Fixed-parameter BRF encoder + LICell + MLP classifier that uses speed.
    """
    def __init__(
        self,
        *,
        f0s: list[float],
        Q: float,
        fs: float = 8192,
        input_size: int = 1,
        num_classes: int = 3,
        readout_tau_mean: float = 20.0,
        readout_tau_std: float = 0.0,
        mlp_hidden_size: int = 32,
    ):
        super().__init__()

        dt = 1 / fs
        self.fs = fs
        self.dt = dt
        self.hidden_size = len(f0s)

        # Compute omega and b_offset from desired resonance frequencies
        omega, b_offset = self.from_f0_Q(f0s, Q, fs)

        self.brfc = BRFCellBoostedInput(
            input_size=input_size,
            layer_size=self.hidden_size,
            adaptive_omega=False,
            omega=omega,
            adaptive_b_offset=False,
            b_offset=b_offset,
            bias=False,
            dt=dt,
        )

        self.readout = LICell(
            input_size=self.hidden_size,
            layer_size=mlp_hidden_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=readout_tau_mean,
            adaptive_tau_mem_std=readout_tau_std,
            bias=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(mlp_hidden_size + 1, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, num_classes),
        )

    @staticmethod
    def from_f0_Q(f0s: list[float], Q: float, fs: float):
        dt = 1 / fs
        omega = 2 * torch.pi * torch.tensor(f0s) * dt
        b = -omega / (2 * Q)
        p_omega = sustain_osc(omega)
        b_offset = p_omega - b.abs()
        return omega, b_offset

    def forward(self, x: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (T, B, input_size)
            Time-major vibration input.

        speed : (B, 1)
            Scalar speed per sample [Hz].

        Returns
        -------
        logits : (T, B, num_classes)
        """
        t_steps, batch, _ = x.shape

        z = x.new_zeros(batch, self.hidden_size)
        u = v = q = z.clone()
        u_ro = x.new_zeros(batch, self.readout.layer_size)

        logits = []

        for t in range(t_steps):
            z, u, v, q = self.brfc(x[t], (z, u, v, q))
            u_ro = self.readout(z, u_ro)

            # Concatenate speed → (B, features + 1)
            combined = torch.cat([u_ro, speed], dim=1)
            logit = self.classifier(combined)
            logits.append(logit)

        return torch.stack(logits)
