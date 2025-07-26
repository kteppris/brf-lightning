import math
import torch
import torch.nn as nn
from rich.console import Console

from brf_snn.modules import LICell, RFCell, LICellBP
from brf_snn.functional import spike_deletion, quantize_tensor

from brf_lightning.models.backbones.modules.brf  import BRFCell

console = Console()


# --------- MIL Pooling Head ----------
class TemporalMILPool(nn.Module):
    """
    Aggregates [T, B, C] logits to [B, C] bag-level logits.
    mode ∈ {"mean", "max", "lse", "topk"}.
    """
    def __init__(self, mode: str = "lse", r: float = 5.0, k: int | None = None):
        super().__init__()
        assert mode in {"mean", "max", "lse", "topk"}
        self.mode = mode
        self.r = r
        self.k = k

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [T, B, C]
        T = logits.size(0)

        if self.mode == "mean":
            return logits.mean(dim=0)                           # [B, C]

        if self.mode == "max":
            return logits.max(dim=0).values                     # [B, C]

        if self.mode == "lse":
            # smooth max (log-sum-exp) pooling
            s = torch.logsumexp(self.r * logits, dim=0) - math.log(T)
            return s / self.r                                   # [B, C]

        if self.mode == "topk":
            k = self.k if self.k is not None else max(1, T // 50)
            topk_vals, _ = logits.topk(k, dim=0)
            return topk_vals.mean(dim=0)                        # [B, C]


class SimpleResRNNWithPooling(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            pruning: bool = False,
            adaptive_omega_a: float = 5.,
            adaptive_omega_b: float = 10.,
            adaptive_b_offset_a: float = 0.,
            adaptive_b_offset_b: float = 1.,
            out_adaptive_tau_mem_mean: float = 20.,
            out_adaptive_tau_mem_std: float = 5.,
            n_last: int = 1,
            mask_prob: float = 0.,
            sub_seq_length: int = 0,
            hidden_bias: bool = False,
            output_bias: bool = False,
            label_last: bool = False,
            dt: float = 0.01,
            theta: float = 0.9,
            # ---- new: pooling config ----
            mil_pool_mode: str = "lse",
            mil_pool_r: float = 5.0,
            mil_pool_topk: int | None = None,
    ) -> None:
        super(SimpleResRNNWithPooling, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.sub_seq_length = sub_seq_length
        self.label_last = label_last
        self.n_last = n_last
        self.mask_prob = mask_prob

        self.adaptive_omega_a = adaptive_omega_a
        self.adaptive_omega_b = adaptive_omega_b
        self.adaptive_b_offset_a = adaptive_b_offset_a
        self.adaptive_b_offset_b = adaptive_b_offset_b

        self.out_adaptive_tau_mem_mean = out_adaptive_tau_mem_mean
        self.out_adaptive_tau_mem_std  = out_adaptive_tau_mem_std

        self.hidden = BRFCell(
            input_size=input_size + hidden_size,
            layer_size=hidden_size,
            bias=hidden_bias,
            mask_prob=mask_prob,
            adaptive_omega=True,
            adaptive_omega_a=adaptive_omega_a,
            adaptive_omega_b=adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=adaptive_b_offset_a,
            adaptive_b_offset_b=adaptive_b_offset_b,
            dt=dt,
            pruning=pruning,
            theta=theta
        )

        self.out = LICell(
            input_size=hidden_size,
            layer_size=output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=out_adaptive_tau_mem_std,
            bias=output_bias,
        )

        self.mil_pool = TemporalMILPool(
            mode=mil_pool_mode,
            r=mil_pool_r,
            k=mil_pool_topk
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor]:
        """
        Returns:
            - outputs_t: [T_eff, B, C] per-timestep logits
            - outputs_bag: [B, C] pooled bag-level logits (use this for CE/NLL)
            - states, num_spikes (unchanged)
        """
        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = []
        num_spikes = torch.tensor(0., device=x.device)

        hidden_z = torch.zeros((batch_size, self.hidden_size), device=x.device)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_v = torch.zeros_like(hidden_z)
        hidden_q = torch.zeros_like(hidden_z)

        out_u = torch.zeros((batch_size, self.output_size), device=x.device)

        for t in range(sequence_length):
            input_t = x[t]
            hidden = (hidden_z, hidden_u, hidden_v, hidden_q)

            hidden_z, hidden_u, hidden_v, hidden_q = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),
                hidden
            )

            num_spikes += hidden_z.sum()
            out_u = self.out(hidden_z, out_u)

            if t >= self.sub_seq_length:
                outputs.append(out_u)

        outputs_t = torch.stack(outputs)  # [T_eff, B, C]

        if self.label_last:
            outputs_t = outputs_t[-self.n_last:, :, :]

        outputs_bag = self.mil_pool(outputs_t)  # [B, C]

        return outputs_bag, outputs_t, ((hidden_z, hidden_u), out_u), num_spikes
