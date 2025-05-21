from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torchmetrics.classification.precision_recall
import torchmetrics.classification.specificity
import numpy as np
import torch
from schedulers import get_cosine_with_hard_restarts_schedule_with_warmup_and_decay
import trainers.common as common
from trainers.common_trainer import CommonTrainerDownstream
from torchmetrics import Metric
from torch import Tensor

 

class TrainingSleepApnea(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.train_feature_acc = torchmetrics.classification.accuracy.BinaryAccuracy()
        self.valid_feature_acc = torchmetrics.classification.accuracy.BinaryAccuracy()
        self.test_feature_acc = torchmetrics.classification.accuracy.BinaryAccuracy()

        loss = nn.BCEWithLogitsLoss(pos_weight=self.weights)

        # self.train_feature_f1 = torchmetrics.F1Score(num_classes=config.num_classes, threshold=0.5)
        # self.valid_feature_f1 = torchmetrics.F1Score(num_classes=config.num_classes, threshold=0.5)
        # self.test_feature_f1 = torchmetrics.F1Score(num_classes=config.num_classes, threshold=0.5)
        

    def training_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        train_feature_acc = self.train_feature_acc.to(preds.device)
        train_feature_acc(preds, targets)
        self.log('train_feature_acc', train_feature_acc, prog_bar=True)
        self.log('train_loss', loss.detach().item(), prog_bar=True)

        return loss
    
    def validation_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        valid_feature_acc = self.valid_feature_acc.to(preds.device)
        valid_feature_acc(preds, targets)
        self.log('val_feature_acc', valid_feature_acc, prog_bar=True)
        self.log('val_loss', loss.detach().item(), prog_bar=True)

        return loss
            
    def test_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        test_feature_acc = self.test_feature_acc.to(preds.device)
        test_feature_acc(preds, targets)
        self.log('test_feature_acc', test_feature_acc, prog_bar=True)
        self.log('test_loss', loss.detach().item(), prog_bar=True)

        return loss
            
    
    def predict_batch(self, batch):
        x = batch["signals"]
        targets = batch['labels']
        # get one hot encoding

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        logits = self.model(x)
        preds = (torch.sigmoid(logits) > 0.5).float()

        mask = (targets != -1)
        targets = targets[mask]
        logits = logits[mask]
        preds = preds[mask]
        
        loss_cls = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=self.weights)
        
        return loss_cls, logits, preds, targets



class SegmentAccuracy(Metric):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size    
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target) -> None:
        # preds will be [bs, seq_len, num_classes]
        # target will be [bs, minutes, num_classes]
        
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total