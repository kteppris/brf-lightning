import collections
import io
import logging
import math
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
from PIL import Image

log = logging.getLogger("lightning.pytorch.core")

def is_brf(module: torch.nn.Module) -> bool:
    return hasattr(module, "omega") and "rf" in module.__class__.__name__.lower()

def fig_to_tensor(fig) -> torch.Tensor:
    buf = io.BytesIO()
    Canvas(fig).print_png(buf)
    buf.seek(0)
    im = np.asarray(Image.open(buf)).copy()
    plt.close(fig)
    return torch.from_numpy(im).permute(2, 0, 1)

class BRFDebug(Callback):
    def __init__(
        self,
        *,
        log_interval: int = 20,
        fft_len: int = 128,
        raster_len: int = 64,
        vmax_len: int = 128,
        max_classes: int = 10,
    ):
        super().__init__()
        self.log_int = log_interval
        self.fft_len = fft_len
        self.raster_len = raster_len
        self.vmax_len = vmax_len
        self.max_classes = max_classes

        self.step = 0
        self.stats: Dict[str, Dict] = {}
        self.hooks_done = False
        self.C: Optional[int] = None

    def _empty_stat(self, H: int):
        return {
            "spikes": torch.zeros(H),
            "n_win": 0,
            "raster": collections.deque(maxlen=self.raster_len),
            "trace": collections.deque(maxlen=self.fft_len),
            "per_class_trace": {},
            "per_class": torch.zeros(self.C, H),
            "q_max": collections.deque(maxlen=self.vmax_len),
            "b_med": collections.deque(maxlen=self.vmax_len),
            "omega": [],
            "input_fft": {},
        }

    def _hook(self, name: str):
        def fn(mod, inputs, out):
            x = inputs[0]
            z, _, _, q = out
            if z.ndim == 3:
                z = z.sum(0)
            B, H = z.shape
            st = self.stats[name]

            st["spikes"].add_(z.sum(0).cpu())
            st["n_win"] += 1
            st["trace"].append(float(z.sum()))
            st["raster"].append((z > 0).any(0).cpu())
            st["q_max"].append(float(q.max()))
            if hasattr(mod, "b"):
                st["b_med"].append(float(mod.b.median()))
            if hasattr(mod, "omega_dynamic"):
                st["omega"].append(mod.omega_dynamic.cpu())
            else:
                st["omega"].append(mod.omega.detach().abs().cpu())

            labels = getattr(mod, "_current_labels", None)
            if labels is not None and labels.numel() == B:
                for c in range(min(self.C, self.max_classes)):
                    idx = (labels == c).nonzero(as_tuple=True)[0]
                    if idx.numel():
                        st["per_class"][c] += z[idx].sum(0).cpu()
                        st["per_class_trace"].setdefault(c, collections.deque(maxlen=self.fft_len)).append(
                            float(z[idx].sum())
                        )
                        st["input_fft"][c] = x[idx].mean(0).detach().cpu().numpy()
            if hasattr(mod, "_current_labels"):
                delattr(mod, "_current_labels")
        return fn

    def _stash_labels(self, batch, pl_module):
        _, y, _ = batch
        for mod in pl_module.modules():
            if is_brf(mod):
                mod._current_labels = y.to(mod.omega.device)

    def on_fit_start(self, trainer, pl_module):
        self.C = pl_module.hparams["num_classes"]
        if self.hooks_done:
            return
        for name, mod in pl_module.named_modules():
            if is_brf(mod):
                mod.register_forward_hook(self._hook(name))
                self.stats[name] = self._empty_stat(mod.layer_size)
        self.hooks_done = True

    def on_train_batch_start(self, trainer, pl_module, batch, *_):
        self._stash_labels(batch, pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, *_):
        self._stash_labels(batch, pl_module)

    def on_train_batch_end(self, trainer, pl_module, *_):
        self.step += 1
        if self.step % self.log_int:
            return
        for name, mod in pl_module.named_modules():
            if not is_brf(mod):
                continue
            st = self.stats[name]

            pl_module.log(f"{name}|omega|", mod.omega.abs().mean(), prog_bar=True, logger=False)

            if st["n_win"]:
                rate = st["spikes"].sum() / st["n_win"]
                pl_module.log(f"{name}_spk_per_win", rate, prog_bar=True, logger=False)

            if st["q_max"]:
                pl_module.log(f"{name}/q_max", st["q_max"][-1], prog_bar=True, logger=False)
            if st["b_med"]:
                pl_module.log(f"{name}/b_median", st["b_med"][-1], prog_bar=True, logger=False)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        tb = next((l for l in trainer.loggers if l.__class__.__name__ == "TensorBoardLogger"), None)
        epoch = trainer.current_epoch

        for name, st in self.stats.items():
            mod = dict(pl_module.named_modules())[name]
            dt = float(getattr(mod, "dt", 1.0))
            time_domain = dt < 0.5

            omega = torch.stack(st["omega"]).mean(dim=0)
            centres = omega / (2 * math.pi) / dt if time_domain else omega * dt / (2 * math.pi)
            y_lab = "Hz" if time_domain else "order"
            pl_module.log(f"{name}/median_{y_lab}", float(torch.median(centres)), logger=True)

            if tb:
                tb.experiment.add_histogram(f"{name}/centres_{y_lab}", centres, epoch)
                if st["raster"]:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.imshow(torch.stack(list(st["raster"])).T, cmap="Greys", vmin=0, vmax=1, aspect="auto")
                    ax.set(title=f"{name} raster e{epoch}", xlabel="window", ylabel="neuron")
                    tb.experiment.add_image(f"{name}/raster", fig_to_tensor(fig), epoch)

                for c, trace in st["per_class_trace"].items():
                    if len(trace) >= 4:
                        tr = torch.tensor(list(trace)) - torch.tensor(list(trace)).mean()
                        spec = torch.fft.rfft(tr).abs()
                        freq = torch.fft.rfftfreq(len(tr), d=dt)
                        fig, ax = plt.subplots(figsize=(4, 2))
                        ax.plot(freq, spec)
                        ax.set(title=f"{name} FFT class {c}", xlabel="Hz")
                        tb.experiment.add_image(f"{name}/fft_class_{c}", fig_to_tensor(fig), epoch)

                for c, signal in st["input_fft"].items():
                    spec = np.abs(np.fft.rfft(signal - signal.mean()))
                    freq = np.fft.rfftfreq(len(signal), d=dt)
                    fig, ax = plt.subplots(figsize=(4, 2))
                    ax.plot(freq, spec)
                    ax.set(title=f"{name} input FFT class {c}", xlabel="Hz")
                    tb.experiment.add_image(f"{name}/input_fft_class_{c}", fig_to_tensor(fig), epoch)

                if st["spikes"].sum() > 0:
                    fig, ax = plt.subplots(figsize=(4, 2))
                    ax.bar(np.arange(len(st["spikes"])), st["spikes"].detach().numpy())
                    ax.set(title=f"{name} spike histogram", xlabel="neuron", ylabel="spike count")
                    tb.experiment.add_image(f"{name}/spike_hist", fig_to_tensor(fig), epoch)

            self.stats[name] = self._empty_stat(len(st["spikes"]))
