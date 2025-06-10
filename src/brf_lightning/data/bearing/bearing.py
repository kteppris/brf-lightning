from pathlib import Path
import logging
import math
import random
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
import lightning as L

from brf_lightning.data.bearing.utils import (
    load_vibration_data,
    _class_id_from_path,
    # _highpass,
)

logger = logging.getLogger("lightning.pytorch.core")

class VibraWindowDataset(Dataset):
    """Slice a vibration trace into fixed‑length windows.

    Parameters
    ----------
    df : pandas.DataFrame
    fs : float
        Sampling rate [Hz].
    label : int
    window_size : float, default 1.0
        Window duration [s].
    hop_size : float
        Step size (hop) between consecutive windows [s].
        - `hop_size < window_size`  → overlap
        - `hop_size = window_size`  → contiguous (no overlap, no gap)
        - `hop_size > window_size`  → gap (data is skipped between windows)
    amplitude_factor : float, default 5.0
        Scalar applied **after** z‑normalisation to restore RMS scale.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        fs: float,
        label: int,
        window_size: float = 1.0,
        hop_size: float = 0.5, # Changed from stride_ratio, default assumes window_size=1.0
        amplitude_factor: float = 5.0,
    ):
        assert window_size > 0, "window_size must be > 0"
        assert hop_size > 0, "hop_size must be > 0"

        raw = df["sensor_bearing"].astype("float32").values
        win_samples = int(window_size * fs)
        hop_samples = max(1, int(round(hop_size * fs)))

        # generate windows
        N = len(raw)
        if N < win_samples:
            self._slices = np.empty((0, win_samples), dtype=np.float32)
        else:
            num_frames = (N - win_samples) // hop_samples + 1
            self._slices = np.zeros((num_frames, win_samples), dtype=np.float32)
            for i in range(num_frames):
                start = i * hop_samples
                self._slices[i, :] = raw[start : start + win_samples]
        
        self._amp = amplitude_factor
        self._label = torch.tensor(label, dtype=torch.long)

    def __len__(self) -> int:
        return len(self._slices)

    def __getitem__(self, idx: int):
        x = self._slices[idx]
        x = (x - x.mean()) / (x.std() + 1e-6)
        x = x * self._amp
        return torch.from_numpy(x).unsqueeze(-1), self._label


class BearingDataModule(L.LightningDataModule):
    """Prepare windowed bearing data for train/val/test.

    Parameters
    ----------
    filepaths : list[str | Path]
        One CSV per class/condition.
    batch_size : int, default 32
    val_ratio  : float, default 0.1
        Fraction of each trace reserved for validation.
    test_ratio : float, default 0.1
        Fraction reserved for the end of the trace (test set).
    keep_fraction : float, default 1.0
        Randomly keep only this fraction of sequences within *each* split.
    window_size : float, default 1.0  [s]
    hop_size : float, optional
        Step size (hop) between consecutive windows [s].
        If None, defaults to `0.5 * window_size` (i.e., 50% overlap).
        See :class:`VibraWindowDataset`.
    amp_scale : float, default 5.0
    num_workers : int, default 4
    """

    def __init__(
        self,
        *,
        filepaths: List[str | Path],
        batch_size: int = 32,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        keep_fraction: float = 1.0,
        window_size: float = 1.0,
        hop_size: Optional[float] = None,
        amp_scale: float = 5.0,
        num_workers: int = 4,
    ):
        super().__init__()
        assert 0.0 < val_ratio < 1.0, "val_ratio must be in (0,1)"
        assert 0.0 < test_ratio < 1.0, "test_ratio must be in (0,1)"
        assert (val_ratio + test_ratio) < 1.0, "val+test must be < 1"
        assert 0.0 < keep_fraction <= 1.0, "keep_fraction must be in (0,1]"
        assert window_size > 0.0, "window_size must be > 0"

        self._fps = [Path(p) for p in filepaths]
        self._bs = batch_size
        self._valr = val_ratio
        self._testr = test_ratio
        self._keep = keep_fraction
        self._win_sec = window_size
        
        if hop_size is None:
            self._hop_sec = window_size * 0.5  # Default to 50% overlap of window_size
        else:
            self._hop_sec = hop_size
        assert self._hop_sec > 0, "hop_size must be > 0"
        
        self._amp = amp_scale
        self._nw = num_workers

    @staticmethod
    def _subsample(idxs: list[int], keep_fraction: float) -> list[int]:
        if keep_fraction >= 1.0 or len(idxs) <= 1:
            return idxs
        k = max(1, int(round(len(idxs) * keep_fraction)))
        return random.sample(idxs, k)

    def setup(self, stage: str | None = None):
        train_parts, val_parts, test_parts = [], [], []

        for fp in self._fps:
            df, fs = load_vibration_data(fp)
            label = _class_id_from_path(fp)

            full_ds = VibraWindowDataset(
                df,
                fs,
                label,
                window_size=self._win_sec,
                hop_size=self._hop_sec,
                amplitude_factor=self._amp,
            )

            n_total = len(full_ds)
            n_train = int(math.floor(n_total * (1.0 - self._valr - self._testr)))
            n_val   = int(math.floor(n_total * self._valr))
            n_test  = n_total - n_train - n_val

            idx_train = list(range(0, n_train))
            idx_val   = list(range(n_train, n_train + n_val))
            idx_test  = list(range(n_train + n_val, n_total))

            idx_train = self._subsample(idx_train, self._keep)
            idx_val   = self._subsample(idx_val,   self._keep)
            idx_test  = self._subsample(idx_test,  self._keep)

            train_parts.append(Subset(full_ds, idx_train))
            val_parts.append(Subset(full_ds, idx_val))
            test_parts.append(Subset(full_ds, idx_test))

            logger.info(
                f"{fp.name}: windows={n_total:,d} → "
                f"train={len(idx_train):,d}, val={len(idx_val):,d}, "
                f"test={len(idx_test):,d} (hop={self._hop_sec:.3f}s, keep={self._keep})" # Updated log message
            )

        self.train_ds = ConcatDataset(train_parts)
        self.val_ds   = ConcatDataset(val_parts)
        self.test_ds  = ConcatDataset(test_parts)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, self._bs, shuffle=True,
            num_workers=self._nw, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, self._bs, shuffle=False,
            num_workers=self._nw, pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, self._bs, shuffle=False,
            num_workers=self._nw, pin_memory=True,
        )
