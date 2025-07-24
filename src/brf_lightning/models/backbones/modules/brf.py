import torch
from brf_snn.modules.rf import RFCell, sustain_osc, brf_update
class BRFCell(RFCell):
    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        in_sum = self.linear(x)

        z, u, v, q = state

        omega = torch.abs(self.omega)

        p_omega = sustain_osc(omega, dt=self.dt)

        b_offset = torch.abs(self.b_offset)

        # divergence boundary
        b = p_omega - b_offset - q
        self._current_b = b.detach().cpu()
        
        z, u, v, q = brf_update(
            x=in_sum,
            u=u,
            v=v,
            q=q,
            b=b,
            omega=omega,
            dt=self.dt,
            theta=self.theta
        )

        return z, u, v, q