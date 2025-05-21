import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from torchmetrics.classification import MulticlassAccuracy

from brf_snn.models import SimpleResRNN

logger = logging.getLogger("lightning.pytorch.core")

class ECGClassifier(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        num_classes: int = 6,
        sub_seq_length: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.backbone = backbone

        if hasattr(self.backbone, 'forward'):
            try:
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
        return self.backbone(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        seq, tgt = batch
        seq = seq.permute(1, 0, 2)  # (T_orig, B, C_in)
        tgt = tgt.permute(1, 0, 2)  # (T_orig, B, C_out_onehot)

        outputs = self.backbone(seq)

        if isinstance(outputs, tuple):
            logits = outputs[0]
            num_spikes = outputs[2] if len(outputs) > 2 and outputs[2] is not None else None
        else:
            logits = outputs
            num_spikes = None

        # logp will have the same time dimension as logits from the backbone.
        logp = F.log_softmax(logits, dim=-1) # (T_model_output, B, NumClasses)

        # Targets are initially full length from the dataloader.
        tgt_idx_full = tgt.argmax(dim=-1) # (T_orig, B)

        # Get the sub_seq_length specified for ECGClassifier.
        # This is used to slice the original targets
        s_len_for_target_slicing = self.hparams.sub_seq_length

        # Slice ONLY the target tensor based on ECGClassifier's sub_seq_length.
        # logp is not sliced here, this should have been done in the backbone.
        if s_len_for_target_slicing > 0:
            if tgt_idx_full.size(0) > s_len_for_target_slicing:
                tgt_idx_for_loss = tgt_idx_full[s_len_for_target_slicing:] # (T_orig - s_len_for_target_slicing, B)
            else:
                # This case means the original target sequence is too short to be sliced.
                logger.warning(
                    f"Target sequence length {tgt_idx_full.size(0)} is not greater than "
                    f"ECGClassifier.hparams.sub_seq_length {s_len_for_target_slicing} for stage '{stage}'. "
                    f"Using an empty or very short target slice for loss."
                )
                tgt_idx_for_loss = tgt_idx_full[tgt_idx_full.size(0):]
        else:
            tgt_idx_for_loss = tgt_idx_full

        # At this point, logp (from model) and tgt_idx_for_loss (sliced original target)
        # MUST have the same time dimension for the loss function.
        # This implies: logp.size(0) == (tgt_idx_full.size(0) - s_len_for_target_slicing)
        # which means the backbone must output a sequence of the appropriately reduced length.
        if logp.size(0) != tgt_idx_for_loss.size(0):
            raise RuntimeError(
                f"CRITICAL MISMATCH in sequence lengths for loss calculation in stage {stage}:\n"
                f"  - logp (from model output) has time dimension: {logp.size(0)}\n"
                f"  - tgt_idx_for_loss (sliced original targets) has time dimension: {tgt_idx_for_loss.size(0)}\n"
                f"  - Original target time dimension (T_orig): {tgt_idx_full.size(0)}\n"
                f"  - ECGClassifier.hparams.sub_seq_length (for target slicing): {s_len_for_target_slicing}\n"
                f"This means the backbone ('{self.backbone.__class__.__name__}') is not outputting a sequence "
                f"of the expected length (T_orig - ECGClassifier.hparams.sub_seq_length). "
                f"Ensure the backbone's own 'sub_seq_length' (or equivalent mechanism in its init_args) "
                f"is set and functions to produce this length."
            )

        loss = self.criterion(
            logp.reshape(-1, logp.size(-1)), # (EffectiveTime * B, NumClasses)
            tgt_idx_for_loss.reshape(-1)     # (EffectiveTime * B)
        )

        accuracy = self.acc_metric(
            logp.reshape(-1, logp.size(-1)).exp(),
            tgt_idx_for_loss.reshape(-1)
        )


        on_step_log = True if stage == "train" else False
        on_epoch_log = True
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=on_step_log, on_epoch=on_epoch_log, batch_size=seq.size(1))
        self.log(f"{stage}_accuracy", accuracy, prog_bar=True, on_step=on_step_log, on_epoch=on_epoch_log, batch_size=seq.size(1))

        if stage == "test" and num_spikes is not None:
            if torch.is_tensor(num_spikes):
                num_spikes_item = num_spikes.item() if num_spikes.numel() == 1 else num_spikes.sum().item()
            else:
                try: num_spikes_item = float(num_spikes)
                except (ValueError, TypeError): num_spikes_item = 0.0
            avg_batch_spikes = num_spikes_item / seq.size(1)
            self.log(f"{stage}/sop", avg_batch_spikes, on_step=False, on_epoch=True, batch_size=seq.size(1))
        
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}