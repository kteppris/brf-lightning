import inspect
import logging
from typing import Literal, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

logger = logging.getLogger("lightning.pytorch.core")


class BearingFaultClassifier(L.LightningModule):
    """
    Generic classifier wrapper for time windows or sequences.

    Parameters
    ----------
    backbone
        Feature extractor / sequence model.
        If ``implicit_normal=True`` it must output *num_classes-1* logits
        (one per defect type).
    num_classes
        Total number of labels in the dataset (including *normal*).
    reduction
        ``'last'`` - use final step of a sequence  
        ``'mean'`` - temporal mean  
        ``'none'`` - keep sequence (many-to-many)
    criterion
        Loss function. Defaults to NLLLoss().
    implicit_normal
        *False* - backbone outputs *all* `num_classes` logits (legacy).  
        *True*  - backbone outputs only defect logits;
        a zero-logit column is prepended so “no spike ⇒ class 0”.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_classes: int = 3,
        reduction: Literal["last", "mean", "none"] = "last",
        criterion: Optional[nn.Module] = None,
        implicit_normal: bool = False,
    ) -> None:
        super().__init__()
        # ignore large objects to keep checkpoints light
        self.save_hyperparameters(ignore=["backbone", "criterion"])

        # actual PyTorch model
        self.backbone = backbone

        self.reduction = reduction

        # Loss function, can be overwritten via config submodule injection
        self.criterion = criterion or nn.NLLLoss()
        self.implicit_normal = implicit_normal
        self.defect_classes = num_classes - 1 if implicit_normal else num_classes

        # torchmetrics setup TODO: Implement via config
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # check if model expects speed or not
        sig = inspect.signature(backbone.forward)
        self._pass_speed = "speed" in sig.parameters

    def _reduce_time(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 3:
            if self.reduction == "last":
                return logits[-1]
            if self.reduction == "mean":
                return logits.mean(0)
        return logits

    @staticmethod
    def _to_time_first(x: torch.Tensor) -> torch.Tensor:
        # DataLoader returns (B,T,1); backbone expects (T,B,1)
        return x.permute(1, 0, 2)

    def _expand_labels(self, y: torch.Tensor, seq_len: int) -> torch.Tensor:
        if y.ndim == 1:  # (B,)
            return y.unsqueeze(0).expand(seq_len, -1)  # (T,B)
        if y.ndim == 2 and y.size(0) == seq_len:  # already (T,B)
            return y
        raise ValueError("label shape incompatible with sequence logits")

    def forward(self, x: torch.Tensor, speed: Optional[torch.Tensor] = None):
        x = self._to_time_first(x)
        outputs = self.backbone(x, speed) if self._pass_speed else self.backbone(x)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        # prepend zero-logit for 'normal' if requested
        if self.implicit_normal:
            z = torch.zeros_like(logits[..., :1])
            logits = torch.cat([z, logits], dim=-1)  # (…, C)

        return self._reduce_time(logits)

    def _shared_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        metrics: MetricCollection,
    ) -> torch.Tensor:
        x, y, speed = batch
        logits = self(x, speed)

        if logits.ndim == 3:  # (T,B,C)
            if self.reduction == "none":
                y_seq = self._expand_labels(y, logits.size(0))
                log_p = F.log_softmax(logits, dim=-1)
                loss = self.criterion(log_p.flatten(0, 1), y_seq.flatten())
                metrics(log_p.exp().flatten(0, 1), y_seq.flatten())
                return loss

        log_p = F.log_softmax(logits, dim=-1)
        loss = self.criterion(log_p, y)
        metrics(log_p.exp(), y)
        return loss

    def training_step(self, batch, _):
        loss = self._shared_step(batch, self.train_metrics)
        self.log_dict(self.train_metrics, on_step=True, prog_bar=True)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._shared_step(batch, self.val_metrics)
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, _):
        loss = self._shared_step(batch, self.test_metrics)
        self.log_dict(self.test_metrics, on_epoch=True)
        self.log("test/loss", loss, on_epoch=True)
        return loss
