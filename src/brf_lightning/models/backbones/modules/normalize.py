import torch
import torch.nn as nn
import torch.nn.functional as F

class OnlineNormalSuppression(nn.Module):
    """
    Normalization layer that applies a time-step-wise suppression of signals based on precomputed normal statistics.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, floor: float = 1e-4, amplify_faults: bool = True):
        """
        Parameters:
        - mean: Tensor of shape (C,) representing the normal mean per channel (sensor).
        - std: Tensor of shape (C,) representing the normal std per channel.
        - floor: Minimum std to avoid division by zero.
        - amplify_faults: Whether to scale outputs back to maintain energy.
        """
        super().__init__()
        self.register_buffer("mean", mean.view(1, -1))  # shape (1, C)
        self.register_buffer("std", std.view(1, -1))    # shape (1, C)
        self.floor = floor
        self.amplify = amplify_faults

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize x using per-channel normal statistics.

        x: shape (B, C), where C = input_size
        Returns: normalized x, same shape
        """
        std = torch.clamp(self.std, min=self.floor)
        x_norm = (x - self.mean) / std

        if self.amplify:
            # Optional: rescale to original energy domain (acts like whitening)
            x_norm = x_norm * std

        return x_norm
