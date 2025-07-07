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


class RegressionTrainer(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, target_key='age', weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.target_key = target_key

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

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