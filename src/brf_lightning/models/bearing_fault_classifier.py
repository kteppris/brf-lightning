import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from torchmetrics.classification import MulticlassAccuracy

from brf_snn.models import SimpleResRNN

logger = logging.getLogger("lightning.pytorch.core")

class BearingFaultClassifier(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        num_classes: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.backbone = backbone

        # Attempt JIT scripting for the backbone (optional, for potential speedup)
        if hasattr(self.backbone, 'forward'):
            try:
                # Check if backbone is already a ScriptModule
                if not isinstance(self.backbone, torch.jit.ScriptModule):
                    self.backbone = torch.jit.script(self.backbone)
                    logger.info(f"Successfully JIT scripted the backbone: {self.backbone.__class__.__name__}")
            except Exception as e:
                logger.warning(f"Could not JIT script the backbone {self.backbone.__class__.__name__}. Error: {e}")
                logger.info("Proceeding without JIT scripting for the backbone.")
        else:
            logger.warning(f"Backbone {self.backbone.__class__.__name__} does not have a forward method. Skipping JIT.")

        self.criterion = torch.nn.NLLLoss()
        self.acc_metric = MulticlassAccuracy(num_classes=self.hparams.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be (Batch, SeqLen, Features)
        # Permute to (SeqLen, Batch, Features) as backbone expects time-first
        x = x.permute(1, 0, 2)
        return self.backbone(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        seq, tgt = batch  # seq: (B, T_orig, C_in), tgt: (B)

        # Permute seq to (T_orig, B, C_in) for RNNs expecting time-first input
        seq = seq.permute(1, 0, 2)

        # outputs from the backbone
        # The backbone (SimpleResRNN) might return a tuple (e.g., logits, states, spikes)
        # or just logits.
        backbone_output = self.backbone(seq)

        if isinstance(backbone_output, tuple):
            # Assuming the first element is the primary output (logits sequence)
            # logits_seq shape: (T_model_output, B, NumClasses)
            logits_seq = backbone_output[0]
            # Try to get spike count if returned
            num_spikes = backbone_output[2] if len(backbone_output) > 2 and backbone_output[2] is not None else None
        else:
            logits_seq = backbone_output # Shape: (T_model_output, B, NumClasses)
            num_spikes = None

        # For window-level classification, take the output of the last time step
        # from the backbone's output sequence.
        # logits shape: (B, NumClasses)
        logits = logits_seq[-1, :, :]
        # Alternative: Global Average Pooling over time if T_model_output > 1
        # if logits_seq.size(0) > 1:
        #     logits = torch.mean(logits_seq, dim=0) # (B, NumClasses)
        # else:
        #     logits = logits_seq.squeeze(0) # if T_model_output is 1

        logp = F.log_softmax(logits, dim=-1)  # Shape: (B, NumClasses)

        # tgt is already the correct shape (B) with class indices

        mean_loss = self.criterion(logp, tgt)
        loss_for_backward = mean_loss # For window-level classification

        # Calculate accuracy
        # logp.exp() converts log-probabilities to probabilities
        accuracy = self.acc_metric(logp.exp(), tgt)

        # Logging
        # seq.size(1) is Batch size because seq was (T_orig, B, C_in)
        batch_size = seq.size(1)
        on_step_log = True if stage == "train" else False
        self.log(f"{stage}/loss", mean_loss, prog_bar=True, on_step=on_step_log, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/accuracy", accuracy, prog_bar=True, on_step=on_step_log, on_epoch=True, batch_size=batch_size)

        if num_spikes is not None:
            # Assuming num_spikes could be a tensor (e.g., spikes per layer or per output time step)
            # or a single scalar value representing total spikes for the window processing.
            # If it's a sequence of spike counts, sum them.
            if torch.is_tensor(num_spikes):
                total_window_spikes = num_spikes.sum().item()
            else: # Assuming it's already a summed float/int
                try:
                    total_window_spikes = float(num_spikes)
                except (ValueError, TypeError):
                    total_window_spikes = 0.0
            
            avg_spikes_per_window_in_batch = total_window_spikes / batch_size if batch_size > 0 else 0.0
            self.log(f"{stage}/sop_window", avg_spikes_per_window_in_batch, prog_bar=True, on_step=on_step_log, on_epoch=True, batch_size=batch_size)
   
        return loss_for_backward

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        # Ensure self.scheduler is a callable that takes an optimizer
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler) or \
           isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
             scheduler_instance = self.scheduler
        else: # Assumed to be a callable (class type)
             scheduler_instance = self.scheduler(optimizer)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler_instance}