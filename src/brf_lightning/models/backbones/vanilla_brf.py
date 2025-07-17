# simple_res_rnn_balanced.py
import torch
from brf_lightning.models.backbones.modules.brf import BalancedRFCell
from brf_snn.modules import LICell 

class SimpleResRNN(torch.nn.Module):
    """
    One hidden layer of Balanced‐RF neurons followed by an LICell read‑out.
    Carrier frequencies, bandwidths and sampling rate are fixed at init,
    but ω and b_offset remain trainable if you choose.
    """
    def __init__(self,
                 input_size       : int,
                 hidden_size      : int = 6,       # we only need six neurons
                 output_size      : int = 1,
                 *,
                 fs               : float = 8192.0,
                 carriers_khz     : list[float] = (1.91,2.05,2.20,2.86,3.08,3.29),
                 half_bw_hz       : list[float] = (512,512,512,683,683,683),
                 use_euler        : bool  = False,
                 pruning          : bool  = True,
                 mask_prob        : float = 0.5,
                 adaptive_omega   : bool  = True,
                 adaptive_b_off   : bool  = True,
                 hidden_bias      : bool  = False,
                 output_bias      : bool  = False):

        super().__init__()

        # ─── Hidden recurrent (Balanced RF) layer ─────────────────────
        self.hidden = BalancedRFCell(
            input_size      = input_size + hidden_size,   # recurrence
            layer_size      = hidden_size,
            fs              = fs,
            carriers_khz    = carriers_khz,
            half_bw_hz      = half_bw_hz,
            use_euler       = use_euler,
            pruning         = pruning,
            mask_prob       = mask_prob,
            adaptive_omega  = adaptive_omega,
            adaptive_b_off  = adaptive_b_off,
            bias            = hidden_bias)

        # ─── Output LICell (unchanged) ────────────────────────────────
        self.out = LICell(
            input_size      = hidden_size,
            layer_size      = output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=20.,
            adaptive_tau_mem_std =5.,
            bias            = output_bias)

    # ─── forward pass identical to the original SimpleResRNN ──────────
    def forward(self, x: torch.Tensor):
        seq_len, batch = x.shape[0], x.shape[1]
        z, u, v, q = (torch.zeros(batch, self.hidden.hidden_size,
                                  device=x.device) for _ in range(4))
        out_u      = torch.zeros(batch, self.out.layer_size, device=x.device)
        outputs    = []
        spike_cnt  = torch.tensor(0., device=x.device)

        for t in range(seq_len):
            in_t   = x[t]
            state  = (z, u, v, q)
            z, u, v, q = self.hidden(torch.cat((in_t, z), 1), state)
            spike_cnt += z.sum()
            out_u = self.out(z, out_u)
            outputs.append(out_u)

        return torch.stack(outputs)
