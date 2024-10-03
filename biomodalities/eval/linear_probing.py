import logging
from typing import Any, Dict, List, Tuple
from torch.utils.data import DataLoader, TensorDataset

import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
import random
import numpy as np

from biomodalities.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from biomodalities.utils.metrics import accuracy_at_k, weighted_mean


class LinearModel(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        optimizer_name: str,
        lr: float,
        weight_decay: float,
        batch_size: int,
        scheduler_name: str,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: int,
        lr_decay_steps: list,
        scheduler_interval: str,
        seed: int,
        label_key: str,  
        unique_labels: List[str],  
    ):
        """Implements linear probing for gene expression data.

        Args:
            input_dim (int): dimension of the input vector z.
            num_classes (int): number of classes for classification.
            optimizer_name (str): name of the optimizer.
            lr (float): learning rate.
            weight_decay (float): weight decay for optimizer.
            batch_size (int): number of samples in the batch.
            scheduler_name (str): name of the scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (int): number of warmup epochs.
            lr_decay_steps (list): steps to decay the learning rate if scheduler is step.
            scheduler_interval (str): interval to update the lr scheduler.
            seed (int): seed for initializing the linear layer.
            label_key (str): key to the labels to be used in obs of anndata.
            unique_labels (List[str]): list of unique labels in the dataset.
        """
        super().__init__()

        self.save_hyperparameters()

        # Seed for reproducibility
        self.seed = seed
        self._set_seed(seed)

        # Linear layer for classification
        self.classifier = nn.Linear(input_dim, num_classes)

        # Loss function
        self.loss_func = nn.CrossEntropyLoss()

        # Data related
        self.label_key = label_key
        self.unique_labels = unique_labels
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        

        # Optimizer related
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        # Scheduler related
        self.scheduler_name = scheduler_name
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.scheduler_interval = scheduler_interval
        assert self.scheduler_interval in ["step", "epoch"]

        if self.scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={self.scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

        # Keep track of validation metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []


    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """
        assert self.optimizer_name in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer_name](
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_name == "none":
            return optimizer

        if self.scheduler_name == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.trainer.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.trainer.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler_name == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler_name == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
        elif self.scheduler_name == "exponential":
            scheduler = ExponentialLR(optimizer, gamma=0.95)
        else:
            raise ValueError(
                f"{self.scheduler_name} not in (warmup_cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the linear layer for evaluation.

        Args:
            X (torch.Tensor): a batch of input vectors.

        Returns:
            Dict[str, Any]: a dict containing logits.
        """
        logits = self.classifier(X)
        return {"logits": logits}

    def _extract_labels(self, batch):
        """Extract and convert labels from the batch using label_key and unique_labels."""
        label_indices = np.array([self.label_to_idx[label] for label in batch[self.label_key]], dtype=np.int64)
        return torch.from_numpy(label_indices).to(self.device)

    def shared_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Any]:
        """Performs operations that are shared between the training and validation steps.

        Args:
            batch (Tuple): a batch of input vectors and targets.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]: a dict containing the batch size, loss, accuracy @1 and accuracy @5.
        """
        X, batch_data = batch
        target = self._extract_labels(batch_data)
        logits = self(X)["logits"]
        loss = self.loss_func(logits, target)
        acc1, acc5 = accuracy_at_k(logits, target, top_k=(1, 5))
        metrics = {"batch_size": X.size(0), "loss": loss, "acc1": acc1, "acc5": acc5}
        return metrics

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Performs the training step.

        Args:
            batch (Tuple): a batch of input vectors and targets.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: the loss.
        """
        out = self.shared_step(batch, batch_idx)
        log = {"train_loss": out["loss"], "train_acc1": out["acc1"], "train_acc5": out["acc5"]}
        self.log_dict(log, on_epoch=True, sync_dist=True)
        return out["loss"]

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step.

        Args:
            batch (Tuple): a batch of input vectors and targets.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]: a dict containing the batch size, validation loss, accuracy @1 and accuracy @5.
        """
        out = self.shared_step(batch, batch_idx)
        metrics = {
            "batch_size": out["batch_size"],
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
        }
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        """Averages the losses and accuracies of all the validation batches."""
        val_loss = weighted_mean(self.validation_step_outputs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(self.validation_step_outputs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(self.validation_step_outputs, "val_acc5", "batch_size")
        self.validation_step_outputs.clear()
        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        print(f"\n\nValidation loss: {val_loss}, Validation acc1: {val_acc1}, Validation acc5: {val_acc5}")
        self.log_dict(log, sync_dist=True)
    
    def test_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Any]:
        out = self.shared_step(batch, batch_idx)
        metrics = {
            "batch_size": out["batch_size"],
            "test_loss": out["loss"],
            "test_acc1": out["acc1"],
            "test_acc5": out["acc5"],
        }
        self.test_step_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self):
        test_loss = weighted_mean(self.test_step_outputs, "test_loss", "batch_size")
        test_acc1 = weighted_mean(self.test_step_outputs, "test_acc1", "batch_size")
        test_acc5 = weighted_mean(self.test_step_outputs, "test_acc5", "batch_size")
        self.test_step_outputs.clear()
        log = {"test_loss": test_loss, "test_acc1": test_acc1, "test_acc5": test_acc5}
        print(f"\n\nTest loss: {test_loss}, Test acc1: {test_acc1}, Test acc5: {test_acc5}")
        self.log_dict(log, sync_dist=True)
    


