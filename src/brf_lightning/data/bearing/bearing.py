# brf_lightning/data/bearing/datamodule.py
# ----------------------------------------
import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal as sig
import torch
import lightning as L
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from brf_lightning.data.bearing.utils import (
    _class_id_from_path,
    fdtw_safe,
    load_vibration_data,
)

logger = logging.getLogger("lightning.pytorch.core")

ROBUST_C = 1.4826

def _rpm_from_path(fp: str | Path) -> int:
    """Extract the first integer from a filename as nominal RPM."""
    return int("".join(c for c in Path(fp).stem if c.isdigit()))


class WindowBearingDataset(Dataset):
    """
    Single-trace window dataset.
    Returns (window[T,1], label, speed) where *speed* is scalar tensor [Hz or order].
    """

    def __init__(
        self,
        signal: np.ndarray,
        *,
        label: int,
        speed_hz: int,
        fs: float,
        win_s: float,
        hop_s: float,
        target_rms: Optional[float] = None,
    ):
        win_n = int(round(win_s * fs))
        hop_n = max(1, int(round(hop_s * fs)))
        n_win = max(0, (signal.size - win_n) // hop_n + 1)

        self._slices = (
            np.lib.stride_tricks.sliding_window_view(signal, win_n)[:: hop_n].copy()
            if n_win
            else np.empty((0, win_n), dtype=np.float32)
        )

        self.target_rms = float(target_rms) if target_rms else None
        self.label = torch.tensor(label, dtype=torch.long)
        self.speed = torch.tensor([float(speed_hz)], dtype=torch.float32)

    # ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._slices)

    def __getitem__(self, idx: int):
        x = self._slices[idx].astype(np.float32)
        x -= x.mean()
        if self.target_rms is not None:
            rms = math.sqrt(float((x * x).mean()))
            x *= self.target_rms / (rms + 1e-6)
        return torch.from_numpy(x).unsqueeze(-1), self.label, self.speed


# ────────────────────────────────────────────────────────────────────────
# main DataModule
# ────────────────────────────────────────────────────────────────────────
class BearingDataModule(L.LightningDataModule):
    """Bearing-fault DataModule with clean, override-ready internals."""

    SPLIT_MODES = {"chronological", "file_speed"}
    ORDER_MODES = {"none", "fixed", "trigger", "fdtw"}

    # ───────── init ─────────
    def __init__(
        self,
        *,
        filepaths: List[str | Path],
        batch_size: int = 32,
        # splitting
        split_strategy: str = "chronological",
        split_by_speed: Optional[Dict[str, List[int]]] = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        keep_fraction: float = 1.0,
        # windows
        window_size: float = 1.0,
        hop_size: Optional[float] = None,
        # preprocessing
        target_rms: Optional[float] = None,
        norm: Optional[str] = "global",  # global / file / None
        order_domain_mode: str = "none",  # none / fixed / trigger / fdtw
        samples_per_rev: int = 360,
        trig_column: str = "trig",
        sensor_column: str = "sensor_bearing",
        # misc
        balance_mode: str = "global_min",
        num_workers: int = 4,
    ):
        super().__init__()

        if split_strategy not in self.SPLIT_MODES:
            raise ValueError(f"split_strategy ∉ {self.SPLIT_MODES}")
        if order_domain_mode not in self.ORDER_MODES:
            raise ValueError(f"order_domain_mode ∉ {self.ORDER_MODES}")

        # immutable hyper-params
        self.paths = [Path(p) for p in filepaths]
        self.bs = batch_size
        self.split_strategy = split_strategy
        self.split_by_speed = split_by_speed or {}
        self.val_ratio, self.test_ratio = val_ratio, test_ratio
        self.keep_fraction = keep_fraction

        self.win_s = window_size
        self.hop_s = hop_size or window_size * 0.5

        self.target_rms = target_rms
        self.norm = norm
        self.order_mode = order_domain_mode
        self.spr = samples_per_rev
        self.trig_col, self.sensor_col = trig_column, sensor_column

        self.balance_mode = balance_mode
        self.nw = num_workers

        # filled during setup
        self.global_mean: Optional[float] = None
        self.global_std: Optional[float] = None
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None

    # ───────── order-domain conversions ─────────
    def _to_angle_fixed(self, raw: np.ndarray, fs: float, rpm: float) -> Tuple[np.ndarray, float]:
        factor = rpm * self.spr / fs
        return sig.resample(raw, int(round(raw.size * factor))).astype(np.float32), float(self.spr)

    def _to_angle_trigger(self, df: pd.DataFrame, fs: float) -> Tuple[np.ndarray, float]:
        trig = df[self.trig_col].to_numpy()
        edges = np.flatnonzero(np.diff(trig.astype(np.int8)) == 1) + 1
        if len(edges) < 2:
            raise RuntimeError("trigger channel missing edges")
        segs = np.split(df[self.sensor_col].to_numpy(np.float32), edges)
        cycles = [sig.resample(s, self.spr).astype(np.float32) for s in segs if len(s)]
        return np.concatenate(cycles), float(self.spr)

    # ───────── public Lightning hook ─────────
    def setup(self, stage: Optional[str] = None):
        meta, bucket_cnt, global_pool = self._load_all_files()
        if self.norm == "global":
            self._compute_global_norm(global_pool)

        split_sets, split_log = self._build_split_sets(meta, bucket_cnt)
        self.train_ds = ConcatDataset(split_sets["train"])
        self.val_ds = ConcatDataset(split_sets["val"]) if split_sets["val"] else None
        self.test_ds = ConcatDataset(split_sets["test"]) if split_sets["test"] else None
        self._log_split_summary(split_log)

    # ───────────────── helper blocks ─────────────────
    # 1. load and preprocess every file --------------------------------------------------
    def _load_all_files(self):
        meta, bucket_cnt, global_pool = [], defaultdict(int), []

        for fp in self.paths:
            df, fs = load_vibration_data(fp)
            label, rpm_nom = _class_id_from_path(fp), _rpm_from_path(fp)

            # order-domain transformation
            sig_arr, fs_eff, rpm_eff = self._prepare_signal(df, fs, rpm_nom)

            # per-file or global z-norm collection
            if self.norm == "file":
                sig_arr = ((sig_arr - sig_arr.mean()) / (sig_arr.std() + 1e-6)).astype(np.float32)
            elif self.norm == "global":
                global_pool.append(sig_arr.astype(np.float64))

            # window count estimate for bucket balancing
            win_est = max(
                0,
                (len(sig_arr) - int(fs_eff * self.win_s))
                // max(1, int(fs_eff * self.hop_s))
                + 1,
            )
            bucket_cnt[(label, rpm_nom)] += win_est

            meta.append(
                dict(
                    path=fp,
                    signal=sig_arr,
                    fs=fs_eff,
                    label=label,
                    rpm_nom=rpm_nom,
                    rpm_eff=rpm_eff,
                )
            )
        return meta, bucket_cnt, global_pool

    def _prepare_signal(
        self, df: pd.DataFrame, fs: float, rpm_nom: int
    ) -> Tuple[np.ndarray, float, int]:
        """Return (signal, effective_fs, effective_rpm)."""
        if self.order_mode == "none":
            return df[self.sensor_col].to_numpy(np.float32), fs, rpm_nom

        if self.order_mode == "fixed":
            sig_arr, fs_eff = self._to_angle_fixed(df[self.sensor_col].to_numpy(np.float32), fs, rpm_nom)
            return sig_arr, fs_eff, 1

        if self.order_mode == "trigger":
            sig_arr, fs_eff = self._to_angle_trigger(df, fs)
            return sig_arr, fs_eff, 1

        # fdtw
        sig_arr, fs_eff = fdtw_safe(
            df[self.sensor_col].to_numpy(np.float32),
            fs,
            spr=self.spr,
            max_len_sec=30.0,
        )
        return sig_arr, fs_eff, 1

    # 2. compute global mean/std ----------------------------------------------------------
    def _compute_global_norm(self, global_pool: List[np.ndarray]):
        """
        Robust global z-score:  μ = mean,  σ̂ = 1.4826 * MAD
        """
        concat = np.concatenate(global_pool, dtype=np.float64)

        mu = float(concat.mean())
        mad = float(np.median(np.abs(concat - mu)))
        sigma_robust = ROBUST_C * mad                       # ≈ σ for Gaussian data

        # 99.8‑th percentile clip (winsorisation)
        clip_val = float(np.percentile(np.abs(concat), 99.8))

        self.global_mean = mu
        self.global_std  = sigma_robust
        self.clip_val    = clip_val

        logger.info(
            f"[GlobalNorm] μ={mu:.5f}  σ̂(MAD)={sigma_robust:.5f}  "
            f"clip ±{clip_val:.2f}"
        )

    # 3. build train/val/test datasets ----------------------------------------------------
    def _build_split_sets(self, meta, bucket_cnt):
        # cap windows per (class,rpm_nom) bucket
        if self.balance_mode == "global_min":
            cap = min(bucket_cnt.values())
            bucket_cap = defaultdict(lambda: cap)
        else:
            bucket_cap = defaultdict(lambda: float("inf"))

        val_rpms = set(self.split_by_speed.get("val", []))
        test_rpms = set(self.split_by_speed.get("test", []))

        split_sets = dict(train=[], val=[], test=[])
        split_log = []
        rng = random.Random(0)

        for m in meta:
            sig_arr = m["signal"]
            if self.norm == "global":
                sig_arr = (sig_arr - self.global_mean) / (self.global_std + 1e-8)
                sig_arr = np.clip(sig_arr, -self.clip_val,  self.clip_val, sig_arr)

            ds_full = WindowBearingDataset(
                sig_arr,
                label=m["label"],
                speed_hz=m["rpm_eff"],
                fs=m["fs"],
                win_s=self.win_s,
                hop_s=self.hop_s,
                target_rms=self.target_rms,
            )

            idx_map = self._index_map_for_file(
                len(ds_full),
                m["rpm_nom"],
                val_rpms,
                test_rpms,
            )

            # iterate splits
            for split, idx_all in idx_map.items():
                if not idx_all:
                    continue

                # balance only training
                if split == "train":
                    idx_all = idx_all[: bucket_cap[(m['label'], m['rpm_nom'])]]

                # global keep_fraction
                if self.keep_fraction < 1.0:
                    k = max(1, int(len(idx_all) * self.keep_fraction))
                    idx_all = rng.sample(idx_all, k)

                split_sets[split].append(Subset(ds_full, idx_all))
                split_log.append(
                    dict(
                        file=Path(m["path"]).name,
                        label=m["label"],
                        rpm=m["rpm_nom"],
                        split=split,
                        n_win=len(idx_all),
                    )
                )

        # sanity: requested splits must not be empty
        for split in ("val", "test"):
            if self.split_by_speed.get(split) and not split_sets[split]:
                raise ValueError(
                    f"[BearingDataModule] split_strategy='file_speed' but '{split}' is empty."
                )
        return split_sets, split_log

    # helper for per-file index mapping
    def _index_map_for_file(
        self, n_win: int, rpm: int, val_rpms: set[int], test_rpms: set[int]
    ) -> Dict[str, List[int]]:
        """Return dict(split → list(indices)) for a single file."""
        if self.split_strategy == "chronological":
            n_test = int(n_win * self.test_ratio)
            n_val = int(n_win * self.val_ratio)
            return {
                "train": list(range(0, n_win - n_val - n_test)),
                "val": list(range(n_win - n_val - n_test, n_win - n_test)),
                "test": list(range(n_win - n_test, n_win)),
            }

        # file_speed strategy
        if rpm in val_rpms and rpm in test_rpms:
            # NEW: cut by *relative* val/test ratios
            denom = self.val_ratio + self.test_ratio
            n_val  = int(round(n_win * self.val_ratio  / denom))
            n_test = n_win - n_val
            return {
                "val":  list(range(0,           n_val)),
                "test": list(range(n_val,       n_val + n_test)),
            }
        if rpm in val_rpms:
            return {"val": list(range(n_win))}
        if rpm in test_rpms:
            return {"test": list(range(n_win))}
        return {"train": list(range(n_win))}

    # 4. pretty logging --------------------------------------------------
    @staticmethod
    def _log_split_summary(split_log):
        log_df = pd.DataFrame(split_log)
        if log_df.empty:
            logger.warning("No data was split into train/val/test sets!")
            return
        table = (
            log_df.pivot_table(
                index=["file", "label", "rpm"],
                columns="split",
                values="n_win",
                fill_value=0,
            )
            .sort_index()
        )
        logger.info("Data split summary (windows per split):\n" + table.to_string())

    # ───────── dataloaders ─────────
    @staticmethod
    def _make_dl(ds: Dataset, bs: int, nw: int, shuffle: bool):
        return DataLoader(ds, bs, shuffle=shuffle, num_workers=nw, pin_memory=True)

    def train_dataloader(self):
        return self._make_dl(self.train_ds, self.bs, self.nw, shuffle=True)

    def val_dataloader(self):
        if self.val_ds is None:
            raise RuntimeError("val split empty")
        return self._make_dl(self.val_ds, self.bs, self.nw, shuffle=True)

    def test_dataloader(self):
        if self.test_ds is None:
            raise RuntimeError("test split empty")
        return self._make_dl(self.test_ds, self.bs, self.nw, shuffle=False)
