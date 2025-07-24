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

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def is_brf(module: torch.nn.Module) -> bool:
    """Heuristic: a module with an `omega` attribute and class name containing 'rf'."""
    return hasattr(module, "omega") and "rf" in module.__class__.__name__.lower()


def fig_to_tensor(fig) -> torch.Tensor:
    """Convert a Matplotlib figure to CHW uint8 tensor for TensorBoard."""
    buf = io.BytesIO()
    Canvas(fig).print_png(buf)
    buf.seek(0)
    im = np.asarray(Image.open(buf)).copy()
    plt.close(fig)
    return torch.from_numpy(im).permute(2, 0, 1)


# -----------------------------------------------------------------------------
# Callback
# -----------------------------------------------------------------------------

class BRFDebug(Callback):
    """Collect BRF‑layer diagnostics with minimal overhead."""

    def __init__(
        self,
        *,
        log_interval: int = 20,
        raster_len: int = 32,
        vmax_len: int = 128,
        max_classes: int = 10,
    ):
        super().__init__()
        self.log_int = log_interval
        self.raster_len = raster_len
        self.vmax_len = vmax_len
        self.max_classes = max_classes

        self.step = 0
        self.stats: Dict[str, Dict] = {}
        self.hooks_done = False
        self.C: Optional[int] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_stat(self, H: int):
        return {
            "spikes": torch.zeros(H),
            "n_win": 0,
            "raster": collections.deque(maxlen=self.raster_len),
            "per_class": torch.zeros(self.C, H),
            "q_max": collections.deque(maxlen=self.vmax_len),
            "b_current": [],
            "b_init": None,
            "omega": [],
        }

    def _hook(self, name: str):
        @torch.no_grad()
        def fn(mod, inputs, out):
            z, u, v, q = out  # unpack forward outputs
            if z.ndim == 3:
                z = z.sum(0)
            B, H = z.shape
            st = self.stats[name]

            # accumulate
            st["spikes"].add_(z.sum(0).cpu())
            st["n_win"] += 1
            st["raster"].append((z > 0).any(0).cpu())
            st["q_max"].append(float(q.max()))
            st["omega"].append((mod.omega_dynamic if hasattr(mod, "omega_dynamic") else mod.omega).detach().abs().cpu())

            labels = getattr(mod, "_current_labels", None)
            if labels is not None and labels.numel() == B:
                for c in range(min(self.C, self.max_classes)):
                    idx = (labels == c).nonzero(as_tuple=True)[0]
                    if idx.numel():
                        st["per_class"][c] += z[idx].sum(0).cpu()

            if hasattr(mod, "_current_b"):
                b_batch = mod._current_b.mean(0).cpu()
                st["b_current"].append(b_batch)
                if st["b_init"] is None:
                    st["b_init"] = b_batch.clone()

            if hasattr(mod, "_current_labels"):
                delattr(mod, "_current_labels")
            if hasattr(mod, "_current_b"):
                delattr(mod, "_current_b")

        return fn

    def _stash_labels(self, batch, pl_module):
        _, y, _ = batch
        for m in pl_module.modules():
            if is_brf(m):
                m._current_labels = y.to(m.omega.device)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

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

    # ---------------- per‑batch --------------------------------------

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

    # ---------------- per‑epoch --------------------------------------

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        tb = next((l for l in trainer.loggers if l.__class__.__name__ == "TensorBoardLogger"), None)
        epoch = trainer.current_epoch

        for name, st in self.stats.items():
            mod = dict(pl_module.named_modules())[name]

            # frequency histogram (Hz)
            omega = torch.stack(st["omega"]).mean(dim=0)  # rad/s
            centres = omega / (2 * math.pi)
            pl_module.log(f"{name}/median_Hz", float(torch.median(centres)), logger=True)
            if tb:
                tb.experiment.add_histogram(f"{name}/centres_Hz", centres, epoch)

            # raster
            if st["raster"] and tb:
                fig, ax = plt.subplots(figsize=(4, 2.5))
                ax.imshow(torch.stack(list(st["raster"])).T, cmap="Greys", vmin=0, vmax=1, aspect="auto")
                ax.set(title=f"{name} raster e{epoch}", xlabel="window", ylabel="neuron")
                tb.experiment.add_image(f"{name}/raster", fig_to_tensor(fig), epoch)

            # spike rate per neuron
            if st["spikes"].sum() > 0 and tb:
                rate_per_neuron = st["spikes"] / max(st["n_win"], 1)
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.bar(np.arange(len(rate_per_neuron)), rate_per_neuron.numpy())
                ax.set(title=f"{name} spike rate", xlabel="neuron", ylabel="spikes / window")
                tb.experiment.add_image(f"{name}/spike_rate", fig_to_tensor(fig), epoch)

                # per-class spike rate bar plot
                tot_per_cls_rate = st["per_class"].sum(dim=1) / max(st["n_win"], 1)
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.bar(np.arange(len(tot_per_cls_rate)), tot_per_cls_rate.numpy())
                ax.set(title=f"{name} spikes per class", xlabel="class", ylabel="spikes / window")
                tb.experiment.add_image(f"{name}/spk_rate_per_class", fig_to_tensor(fig), epoch)

                present = tot_per_cls_rate[tot_per_cls_rate > 0]
                if present.numel():
                    pl_module.log(f"{name}/mean_spk_rate_present_cls", float(present.mean()), logger=True)

            # divergence / stability plot
            if st["b_init"] is not None and st["b_current"] and tb:
                b_init = st["b_init"]
                b_opt = torch.stack(st["b_current"]).mean(0)

                from brf_lightning.utils.plot_defaults import new_figure, init_plot_style
                from brf_lightning.utils.debuggin_plots import plot_stability
                init_plot_style()
                fig, ax = new_figure(fraction=0.48)
                fig, ax = plot_stability(
                    fig,
                    ax,
                    delta=float(getattr(mod, "dt", 1.0)),
                    omegas=omega.detach().cpu().numpy(),
                    bc_init=b_init.numpy(),
                    bc_opt=b_opt.numpy(),
                )
                tb.experiment.add_image(f"{name}/divergence_plot", fig_to_tensor(fig), epoch)

            # reset state for next epoch
            self.stats[name] = self._empty_stat(len(st["spikes"]))
