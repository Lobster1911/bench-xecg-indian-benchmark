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
from utils.train_utils import focal_loss
from torchmetrics import Metric
from torch import Tensor
from typing import Any


class RegressionTrainer(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, target_key='age', weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.target_key = target_key

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        self.train_nmae = torchmetrics.()

        self.train_mse = torchmetrics.MeanSquaredError()
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

    def training_step(self, batch, _):
        loss, preds, targets = self.predict_batch(batch)

        self.train_mae(preds, targets)
        self.log('train_mae', self.train_mae, prog_bar=True)
        self.train_mse(preds, targets)
        self.log('train_mse', self.train_mse, prog_bar=True)

        self.log('train_loss', loss.detach().item(), prog_bar=True)

        return loss
    
    def validation_step(self, batch, _):
        loss, preds, targets = self.predict_batch(batch)

        self.valid_mae(preds, targets)
        self.log('valid_mae', self.valid_mae, prog_bar=True)
        self.valid_mse(preds, targets)
        self.log('valid_mse', self.valid_mse, prog_bar=True)

        self.log('val_loss', loss.detach().item(), prog_bar=True)
        return loss
            
    def test_step(self, batch, _):
        loss, preds, targets = self.predict_batch(batch)

        self.valid_mae(preds, targets)
        self.log('test_mae', self.valid_mae, prog_bar=True)
        self.valid_mse(preds, targets)
        self.log('test_mse', self.valid_mse, prog_bar=True)

        self.log('test_loss', loss.detach().item(), prog_bar=True)
        return loss     
    
    def predict_batch(self, batch):
        x = batch["signals"]
        targets = batch[self.target_key]
        # get one hot encoding

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        results = self.model(x).squeeze()
        loss = nn.functional.mse_loss(results, targets)

        return loss, results, targets
    


class MeanAbsoluteError(Metric):
    r"""`Computes Mean Absolute Error`_ (MAE):

    .. math:: \text{MAE} = \frac{1}{N}\sum_i^N | y_i - \hat{y_i} |

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import MeanAbsoluteError
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> mean_absolute_error = MeanAbsoluteError()
        >>> mean_absolute_error(preds, target)
        tensor(0.5000)
    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    sum_abs_error: Tensor
    total: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_abs_error, n_obs = 0,0,0

        self.sum_abs_error += sum_abs_error
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes mean absolute error over state."""
        return None
