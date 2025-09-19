from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torchmetrics.classification.f_beta
import torchmetrics.classification.precision_recall
import torchmetrics.classification.specificity
import numpy as np
import torch
from trainers.common_trainer import CommonTrainerDownstream
from utils.loss_utils import focal_loss
from torchmetrics import Metric
from torch import Tensor
from typing import Any


class RegressionTrainer(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, target_key='age', weights=None, map_idx_dataloader=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.target_key = target_key
        self.map_idx_dataloader = map_idx_dataloader
        self.use_log = config.use_log

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        self.train_rsmape_0 = RobustMeanAbsoluteError(epsilon=0)
        self.valid_rsmape_0 = RobustMeanAbsoluteError(epsilon=0)
        self.test_rsmape_0 = RobustMeanAbsoluteError(epsilon=0)

        self.train_mse = torchmetrics.MeanSquaredError()
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

    def training_step(self, batch, _):
        loss, preds, targets = self.predict_batch(batch)

        self.train_mae(preds, targets)
        self.log('train_mae', self.train_mae, prog_bar=True)
        self.train_mse(preds, targets)
        self.log('train_mse', self.train_mse, prog_bar=True)

        self.train_rsmape_0(preds, targets)
        self.log('train_rsmape_0', self.train_rsmape_0, prog_bar=False)

        self.log('train_loss', loss.detach().item(), prog_bar=True)

        return loss
    
    def validation_step(self, batch, _):
        loss, preds, targets = self.predict_batch(batch)

        self.valid_mae(preds, targets)
        self.log('valid_mae', self.valid_mae, prog_bar=True)
        self.valid_mse(preds, targets)
        self.log('valid_mse', self.valid_mse, prog_bar=True)

        self.valid_rsmape_0(preds, targets)
        self.log('valid_rsmape_0', self.valid_rsmape_0, prog_bar=False)

        self.log('val_loss', loss.detach().item(), prog_bar=True)
        return loss
            
    def test_step(self, batch, batch_idx=0, dataloader_idx=0):

            # Detect start of a new dataset
        if batch_idx == 0:
            print(f"Resetting metrics for dataloader {dataloader_idx}")
            self.test_mae.reset()
            self.test_mse.reset()
            self.test_rsmape_0.reset()

        loss, preds, targets = self.predict_batch(batch)
        
        self.test_mae(preds, targets)
        self.test_mse(preds, targets)
        self.test_rsmape_0(preds, targets)

        # if the number of dataloader is bigger than 1
        if len(self.trainer.test_dataloaders) > 1:
            dataloader_idx = self.map_idx_dataloader[dataloader_idx] if self.map_idx_dataloader else dataloader_idx

            self.log(f'test_mae_{dataloader_idx}', self.test_mae, prog_bar=True)
            self.log(f'test_mse_{dataloader_idx}', self.test_mse, prog_bar=True)
            self.log(f'test_rsmape_0_{dataloader_idx}', self.test_rsmape_0, prog_bar=False)
            self.log(f'test_loss_{dataloader_idx}', loss.detach().item(), prog_bar=True)
            
        # if the number of dataloader is 1
        else:
            self.log('test_mae', self.valid_mae, prog_bar=True)
            self.log('test_mse', self.valid_mse, prog_bar=True)

            self.log('test_rsmape_0', self.test_rsmape_0, prog_bar=False)

            self.log('test_loss', loss.detach().item(), prog_bar=True)
            
        return loss  
   
    
    def predict_batch(self, batch):
        x = batch["signals"]
        targets = batch[self.target_key]
        # get one hot encoding

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        results = self.model(x).squeeze()
        if self.use_log:
            loss = nn.functional.mse_loss(results, torch.log(targets))
            results = torch.exp(results)
        else:
            loss = nn.functional.mse_loss(results, targets)

        return loss, results, targets
    


class RobustMeanAbsoluteError(Metric):
    r"""`Computes e Robust Symmetric Mean Absolute Percentage Error`_ (RSMAPE):

    .. math:: \text{RSMAPE} = \frac{1}{N}\sum_i^N | y_i - \hat{y_i} | / (|y_i| + |\hat{y_i}| + \epsilon)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    sum_abs_error: Tensor
    total: Tensor

    def __init__(
        self,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        abs_err = torch.abs(preds - target)
        abs_pred = torch.abs(preds)
        abs_target = torch.abs(target)

        err = abs_err / (abs_pred + abs_target + self.epsilon)

        self.sum_abs_error += err.sum()
        self.total += abs_pred.numel()

    def compute(self) -> Tensor:
        """Computes mean absolute error over state."""
        return self.sum_abs_error / self.total  
