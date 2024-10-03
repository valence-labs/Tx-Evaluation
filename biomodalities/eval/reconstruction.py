import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from torchmetrics.functional import mean_squared_error, mean_absolute_error
import numpy as np

import torch.nn.functional as F

from typing import List, Tuple


from biomodalities.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from biomodalities.utils.preprocessing import normalize_data
from biomodalities.utils.metrics import evaluate_reconstruction_metrics, StructuralTranscriptomeDistance






class DecoderModel(pl.LightningModule):
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

    def __init__(self, control_label, embedding_dim: int, output_dim: int, hidden_dims: List[int],
                 optimizer_name: str, scheduler_name: str, lr: float, weight_decay: float,
                 loss_type: str = 'mse', norm_type: str = 'log', batch_key: str = 'gem_group',
                 control_key: str = 'gene', scheduler_interval: str = "step",
                 warmup_start_lr: float = 0.01, min_lr: float = 0.001, warmup_epochs: int = 10,
                 lr_decay_steps: List[int] = [60, 80]):
        """Decoder model to reconstruct transcriptome vectors from embeddings.

        Args:
            control_label (str): Label for the control group.
            embedding_dim (int): Dimension of the input embeddings.
            output_dim (int): Dimension of the output transcriptome vector (number of genes).
            hidden_dims (List[int]): List of hidden layer dimensions.
            optimizer_name (str): Name of the optimizer.
            scheduler_name (str): Name of the learning rate scheduler.
            lr (float): Learning rate.
            weight_decay (float): Weight decay for the optimizer.
            loss_type (str): Type of loss function, 'mse' or 'mae'.
            norm_type (str): Normalization type for the output data.
            batch_key (str): Key to access batch information in the input data.
            control_key (str): Key to access control information in the input data.
            scheduler_interval (str): Interval for scheduler updates ('step', 'epoch').
            warmup_start_lr (float): Starting learning rate for warmup.
            min_lr (float): Minimum learning rate after decay.
            warmup_epochs (int): Number of warmup epochs.
            lr_decay_steps (List[int]): Epochs at which to decay the learning rate.
        """
        super().__init__()
        self.save_hyperparameters()

        self.loss_type = loss_type
        self.norm_type = norm_type
        self.batch_key = batch_key
        self.control_key = control_key
        self.layers = self._build_layers(embedding_dim, output_dim, hidden_dims)
        
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.scheduler_interval = scheduler_interval
        self.lr = lr
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.lr_decay_steps = lr_decay_steps

        self.control_label = control_label
        

        # Custom metric for batch and control analysis
        self.structural_distance = StructuralTranscriptomeDistance()

    
    def _build_layers(self, embedding_dim: int, output_dim: int, hidden_dims: List[int]) -> nn.Sequential:
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    
    
    def configure_optimizers(self) -> Tuple[List, List]:
        """Configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """
        assert self.optimizer_name in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer_name](
            self.parameters(),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.layers(x)


    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'mse':
            return mean_squared_error(y_hat, y)
        elif self.loss_type == 'mae':
            return mean_absolute_error(y_hat, y)
        else:
            raise ValueError("Unsupported loss type. Use 'mse' or 'mae'.")

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y_obs = batch
        obs = y_obs[1]
        y = y_obs[0]
        y_hat = self(x)
        y_norm = y #normalize_data(y, method=self.norm_type)
        loss = self._compute_loss(y_hat, y_norm)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self._shared_step(batch)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def on_test_epoch_start(self):
        self.all_test_data = []


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y_obs = batch
        obs = y_obs[1]
        y = y_obs[0]
        y_hat = self(x)
        y_norm = y #normalize_data(y, method=self.norm_type)

       
        


        self.log('test_loss', self._compute_loss(y_hat, y_norm))
        if batch_idx % 100 == 0:
             # Extract batch and control info
            batch_info = obs[self.batch_key].to_numpy()  # Extract batch ids as numpy array
            control_info = obs[self.control_key].to_list()  # Extract control info as list
            self.all_test_data.append({'preds': y_hat, 'targets': y_norm, 'batch_ids': batch_info, 'control': control_info})
        return {'preds': y_hat, 'targets': y_norm}

    def on_test_epoch_end(self):
        # Concatenate all batch-wise data
        all_preds = torch.cat([x['preds'] for x in self.all_test_data])
        all_targets = torch.cat([x['targets'] for x in self.all_test_data])
        all_batch_ids = np.concatenate([x['batch_ids'] for x in self.all_test_data])
        all_controls = sum([x['control'] for x in self.all_test_data], [])



        # Update and compute metric at the end of testing
        self.structural_distance.update(all_preds, all_targets, all_batch_ids, all_controls, control_label=self.control_label)
        structural_distance_value = self.structural_distance.compute()
        # Evaluate metrics
        metrics = evaluate_reconstruction_metrics(all_preds, all_targets)
        metrics['structural_distance'] = structural_distance_value
        self.structural_distance.reset()
        
        for key, value in metrics.items():
            self.log(key, value)
        
        self.all_test_data = []
        return metrics

if __name__ == "__main__":
    import torch
    from pytorch_lightning import Trainer
    from biomodalities.data.custom_datasets import AnnDataset
    from biomodalities.data.custom_dataloaders import AnnLoader

    filename = "/mnt/ps/home/CORP/ihab.bendidi/sandbox/biomodal_codebase/datasets/test1/rpe1_raw_singlecell_01.h5ad"
    batch_size = 2048
    model_depth = 3
    optimizer_name = "adam"
    scheduler_name = "warmup_cosine"  # Example: change as needed
    lr = 0.001
    weight_decay = 0.0001
    loss_type = "mse"
    norm_type = 'log'
    batch_key = 'gem_group'
    control_key = 'gene'
    scheduler_interval = "epoch"
    warmup_start_lr = 0.0001
    min_lr = 0.00001
    warmup_epochs = 5
    lr_decay_steps = [30, 60]

    # Initialize the dataset and loader in reconstruct mode
    anndataset = AnnDataset(filename, mode='r', chunk_size=128)
    # TODO : Adapt decoder as dataloader returns obs data as two values
    data_loader = AnnLoader(anndataset, batch_size=batch_size, shuffle=True, data_source='obsm', obsm_key='X_uce_4_layers',num_workers=4, task="reconstruct")

    embedding_dim = data_loader[0][0].shape[1]
    output_dim = data_loader[0][1][0].shape[1]

    # Scale the hidden dimensions between embedding_dim and output_dim
    hidden_dims = [int(embedding_dim + (output_dim - embedding_dim) * i / (model_depth + 1)) for i in range(1, model_depth + 1)]
    # Initialize the model
    model = DecoderModel(
        embedding_dim=embedding_dim, output_dim=output_dim, hidden_dims=hidden_dims,
        optimizer_name=optimizer_name, scheduler_name=scheduler_name, lr=lr, weight_decay=weight_decay,
        loss_type=loss_type, norm_type=norm_type, batch_key=batch_key, control_key=control_key,
        scheduler_interval=scheduler_interval, warmup_start_lr=warmup_start_lr, min_lr=min_lr,
        warmup_epochs=warmup_epochs, lr_decay_steps=lr_decay_steps
    )

    # Create a PyTorch Lightning trainer with a limited number of epochs for testing
    trainer = Trainer(max_epochs=1)

    # Run a training loop to test the model training
    trainer.fit(model, data_loader)

    metrics = trainer.test(model, data_loader)
    print(metrics)

    print("Testing completed successfully.")