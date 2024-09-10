__all__ = [
    'FineTunedTrainee',
    'SupervisedTrainee'
]

import pytorch_lightning as pl
import torch
from torch import nn
from typing import Union, Sequence, Optional
from .utils import ensure_list
from torch.optim import lr_scheduler


class FineTunedTrainee(pl.LightningModule):
    """
    A PyTorch Lightning Module that allows fine-tuning of specific attributes
    of the model (trainee) after a specified number of epochs.

    Parameters
    ----------
    trainee : pl.LightningModule, optional
        The base trainee model to be fine-tuned. If None, an empty model is
        created.
    **kwargs : dict
        Additional keyword arguments specifying the attributes to be
        fine-tuned.
        Each keyword should map to a dictionary where the keys are epoch
        numbers and the values are the new attribute values.

    Attributes
    ----------
    trainee : pl.LightningModule
        The base trainee model to be fine-tuned.
    keys : list of str
        List of attribute names that will be fine-tuned.
    """

    def __init__(self, trainee=None, **kwargs) -> None:
        super().__init__()
        self.trainee = trainee

        # If a model (trainee) is provided, use the same logging mechanism
        if self.trainee:
            self.trainee.log = self.log

        # Store the attribute names to be fine-tuned
        self.keys = list(kwargs.keys())

        # Validate fine-tuning parameters
        for key, value in kwargs.items():
            if not isinstance(value, dict):
                raise TypeError('FineTunedTrainee parameters should be '
                                'dictionary that map epochs to values')
            setattr(self, key, value)

    def configure_optimizers(self):
        """
        Configures the optimizers of the trainee model.

        Returns
        -------
        Any
            The optimizers configured in the trainee model.
        """
        return self.trainee.configure_optimizers()

    def training_step(self, *args, **kwargs):
        """
        Performs a single training step, fine-tuning the attributes as needed.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the training step.
        **kwargs : dict
            Keyword arguments passed to the training step.

        Returns
        -------
        Any
            The output of the trainee's training step.
        """
        n = self.current_epoch  # Current epoch

        # Check if any attributes need to be fine-tuned at this epoch
        for key in self.keys:
            if n in getattr(self, key):

                # Update the attribute of the model (trainee)
                setattr(self.trainee, key, getattr(self, key).pop(n))

        # Execute training step
        return self.trainee.training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        """
        Performs a single validation step.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the validation step.
        **kwargs : dict
            Keyword arguments passed to the validation step.

        Returns
        -------
        Any
            The output of the trainee's validation step.
        """
        return self.trainee.validation_step(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the forward method.
        **kwargs : dict
            Keyword arguments passed to the forward method.

        Returns
        -------
        Any
            The output of the trainee's forward method.
        """
        return self.trainee.forward(*args, **kwargs)


class SupervisedTrainee(pl.LightningModule):
    """
    A PyTorch Lightning Module for supervised learning tasks.

    This module expects the inputs to the network to be in the form of
    (input, reference) pairs. Loss is computed on these pairs.

    Parameters
    ----------
    network : nn.Module, optional
        The feedforward network model.
    loss : nn.Module or nn.ModuleList or nn.ModuleDict, optional
        The loss function(s) to optimize. If a ModuleList or ModuleDict is
        provided, the losses are weighted and summed.
    weights : float or list[float], optional
        The weight(s) to use for each loss if a ModuleList or ModuleDict is
        provided (default is 1).
    metrics : nn.Module or nn.ModuleList or nn.ModuleDict, optional
        The metrics to compute for evaluation.
    augmentation : nn.Module, optional
        The transformation to apply to the input data during training.
    lr : float, optional
        The learning rate for the optimizer (default is 1e-3).
    """

    def __init__(self, network: nn.Module = None, loss: nn.Module = None,
                 weights: Union[float, Sequence[float]] = 1,
                 metrics: Optional[nn.Module] = None,
                 augmentation: Optional[nn.Module] = None,
                 lr=1e-3):
        """
        Parameters
        ----------
        network : nn.Module, optional
            The feedforward network model.
        loss : nn.Module or nn.ModuleList or nn.ModuleDict, optional
            The loss function(s) to optimize. If a ModuleList or ModuleDict is
            provided, the losses are weighted and summed.
        weights : float or list[float], optional
            The weight(s) to use for each loss if a ModuleList or ModuleDict is
            provided (default is 1).
        metrics : nn.Module or nn.ModuleList or nn.ModuleDict, optional
            The metrics to compute for evaluation.
        augmentation : nn.Module, optional
            The transformation to apply to the input data during training.
        lr : float, optional
            The learning rate for the optimizer (default is 1e-3).
        """
        super().__init__()
        self.augmentation = augmentation
        self.network = network
        self.loss = loss
        self.weights = weights
        self.metrics = metrics
        self.lr = lr

    def forward(self, x: torch.Tensor) -> pl.LightningModule:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the network.
        """
        return self.network(x)

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizers and learning rate schedulers.

        Returns
        -------
        dict
            Dictionary containing optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Warm-up scheduler for gradual increase in learning rate
        scheduler1 = lr_scheduler.LinearLR(optimizer, start_factor=1e-8,
                                           end_factor=1, total_iters=2000,
                                           verbose=False)

        # Cool-down scheduler for gradual decrease in learning rate
        scheduler2 = lr_scheduler.LinearLR(optimizer, start_factor=1,
                                           end_factor=1e-6, total_iters=10000,
                                           verbose=False)

        # Combination of schedulers
        scheduler = lr_scheduler.SequentialLR(
            optimizer, [scheduler1, scheduler2], milestones=[90000]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val_loss",
                "frequency": 1
            }
        }

    def training_step(self, train_batch: torch.Tensor) -> torch.Tensor:
        """
        Performs a single training step.

        Parameters
        ----------
        train_batch : tuple
            Batch of training data in the form (input, reference).

        Returns
        -------
        torch.Tensor
            Computed loss for the batch.
        """
        loss = self.step('train', train_batch)
        return loss

    def validation_step(self, val_batch: torch.Tensor) -> None:
        """
        Performs a single validation step.

        Parameters
        ----------
        val_batch : tuple
            Batch of validation data in the form (input, reference).
        """
        self.step('val', val_batch)

    def step(self, stepname: str, batch: tuple[torch.Tensor, torch.Tensor]
             ) -> torch.Tensor:
        """
        Generic step for both training and validation.

        Parameters
        ----------
        stepname : str
            The name of the step ('train' or 'val').
        batch : tuple
            Batch of data in the form (input, reference).

        Returns
        -------
        torch.Tensor
            Computed loss for the batch.
        """
        # Apply augmentation if available
        if self.augmentation:
            batch = self.augmentation(batch)
            if hasattr(batch, 'values'):
                batch = batch.values()

        img, ref = batch  # Unpack the batch
        pred = self.network(img)  # Forward pass
        loss, losses = self.compute_loss(pred, ref)  # Compute loss
        self.log_losses(stepname, loss, losses)  # Log losses
        metrics = self.compute_metric(pred, ref)  # Compute metrics
        self.log_metrics(stepname, metrics)  # Log metrics
        return loss

    def log_losses(self, step: str, sumloss: torch.Tensor, losses: dict
                   ) -> None:
        """
        Logs losses to the logger.

        Parameters
        ----------
        step : str
            The name of the step ('train' or 'val').
        sumloss : torch.Tensor
            The total loss.
        losses : dict
            Dictionary of individual losses.
        """
        self.log(f'{step}_loss', sumloss.detach(), sync_dist=True)
        for name, value in losses.items():
            self.log(f'{step}_loss_{name}', value, sync_dist=True)

    def log_metrics(self, step: str, metrics: dict):
        """
        Logs metrics to the logger.

        Parameters
        ----------
        step : str
            The name of the step ('train' or 'val').
        metrics : dict
            Dictionary of computed metrics.
        """
        for name, value in metrics.items():
            self.log(f'{step}_metric_{name}', value, sync_dist=True)

    def compute_loss(self, pred: torch.Tensor, ref: torch.Tensor
                     ) -> tuple[float, dict]:
        """
        Computes the loss based on predictions and reference targets.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted output from the network.
        ref : torch.Tensor
            Reference target.

        Returns
        -------
        tuple
            Total loss and a dictionary of individual losses.
        """
        losses = {}
        sumloss = 0

        # Compute losses for ModuleDict (multiple losses)
        if type(self.loss) is nn.ModuleDict:
            weights = ensure_list(self.weights, len(self.loss))
            for (name, loss), weight in zip(self.loss.items(), weights):
                value = loss(pred, ref)
                losses[name] = value.detach()
                sumloss += weight * value

        # Compute losses for ModuleList (multiple losses)
        elif type(self.loss) is nn.ModuleList:
            weights = ensure_list(self.weights, len(self.loss))
            counter = {}
            for loss, weight in zip(self.loss, weights):
                value = loss(pred, ref)
                name = type(loss).__name__
                if name in counter:
                    if counter[name] == 0:
                        losses[f'{name}0'] = losses[name]
                        del losses[name]
                    counter[name] += 1
                    name = f'{name}{counter[name]}'
                else:
                    counter[name] = 0
                losses[name] = value.detach()
                sumloss += weight * value

        # Compute loss for a single loss function
        else:
            value = self.loss(pred, ref)
            name = type(self.loss).__name__
            losses[name] = value.detach()
            sumloss = value * self.weights

        return sumloss, losses

    def compute_metric(self, pred: torch.Tensor, ref: torch.Tensor) -> dict:
        """
        Computes metrics based on prediction and reference targets.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted output from the network.
        ref : torch.Tensor
            Reference target.

        Returns
        -------
        dict
            Dictionary of computed metrics.
        """
        with torch.no_grad():
            if not self.metrics:
                return {}

            metrics = {}

            # Compute metrics for ModuleDict (multiple metrics)
            if type(self.metrics) is nn.ModuleDict:
                for name, metric in self.metrics.items():
                    value = metric(pred, ref)
                    metrics[name] = value

            # Compute metrics for ModuleList (multiple metrics)
            elif type(self.metrics) is nn.ModuleList:
                counter = {}
                for metric in self.metrics:
                    value = metric(pred, ref)
                    name = type(metric).__name__
                    if name in counter:
                        if counter[name] == 0:
                            metrics[f'{name}0'] = metrics[name]
                            del metrics[name]
                        counter[name] += 1
                        name = f'{name}{counter[name]}'
                    else:
                        counter[name] = 0
                    metrics[name] = value

            # Compute a single metric
            else:
                value = self.metrics(pred, ref)
                name = type(self.metrics).__name__
                metrics[name] = value

        return metrics
