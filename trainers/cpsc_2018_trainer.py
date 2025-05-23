from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torchmetrics.classification.precision_recall
import torchmetrics.classification.specificity
import numpy as np
import torch
import trainers.common as common
from trainers.common_trainer import CommonTrainerDownstream

class TrainingCPSC_2018(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.train_acc = torchmetrics.classification.accuracy.MultilabelAccuracy(num_labels=self.num_classes, average='micro', ignore_index=-1)
        self.valid_acc = torchmetrics.classification.accuracy.MultilabelAccuracy(num_labels=self.num_classes, average='micro', ignore_index=-1)
        self.test_acc = torchmetrics.classification.accuracy.MultilabelAccuracy(num_labels=self.num_classes, average='micro', ignore_index=-1)

        self.train_f1 = torchmetrics.classification.MultilabelF1Score(num_labels=self.num_classes, average='macro', ignore_index=-1)
        self.valid_f1 = torchmetrics.classification.MultilabelF1Score(num_labels=self.num_classes, average='macro', ignore_index=-1)
        self.test_f1 = torchmetrics.classification.MultilabelF1Score(num_labels=self.num_classes, average='macro', ignore_index=-1)

        self.train_auroc = torchmetrics.classification.MultilabelAUROC(num_labels=self.num_classes, compute_on_step='macro', ignore_index=-1)  
        self.valid_auroc = torchmetrics.classification.MultilabelAUROC(num_labels=self.num_classes, compute_on_step='macro', ignore_index=-1)
        self.test_auroc = torchmetrics.classification.MultilabelAUROC(num_labels=self.num_classes, compute_on_step='macro', ignore_index=-1)


    def training_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        self.train_acc = self.train_acc.to(preds.device)
        self.train_acc(preds, targets)

        self.train_f1 = self.train_f1.to(preds.device)
        self.train_f1(preds, targets)

        self.log('train_loss', loss.detach().item(), prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)

        # auroc
        # self.train_auroc = self.train_auroc.cpu()
        self.train_auroc = self.train_auroc.to(logits.device)
        self.train_auroc(logits, targets)
        self.log("train_auroc", self.train_auroc)

        return loss
    
    def validation_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        self.valid_acc = self.valid_acc.to(preds.device)
        self.valid_acc(preds, targets)

        self.valid_f1 = self.valid_f1.to(preds.device)
        self.valid_f1(preds, targets)

        self.log('val_loss', loss.detach().item(), prog_bar=True)
        self.log('val_acc', self.valid_acc, prog_bar=True)
        self.log('val_f1', self.valid_f1, prog_bar=True)

        # auroc
        self.valid_auroc = self.valid_auroc.to(logits.device)
        self.valid_auroc(logits, targets)
        self.log('val_auroc', self.valid_auroc, prog_bar=True)

        return loss
            
    def test_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        self.test_acc = self.test_acc.to(preds.device)
        self.test_acc(preds, targets)


        self.test_f1 = self.test_f1.to(preds.device)
        self.test_f1(preds, targets)

        self.log("test_loss", loss.detach().item())
        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1)

        # auroc  
        self.test_auroc = self.test_auroc.to(logits.device)
        self.test_auroc(logits, targets)
        self.log("test_auroc", self.test_auroc)

        return loss
            
    
    def predict_batch(self, batch):
        x = batch["signals"]
        targets = batch['class_labels']
        # get one hot encoding

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        logits = self.model(x)
        preds = (torch.sigmoid(logits) > 0.5).float()

        loss_cls = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=self.weights)

        if self.use_focal_loss:
            pt = torch.exp(-loss_cls)
            alpha = 2.
            gamma = .25
            loss_cls = (alpha * (1-pt)**gamma * loss_cls)
        
        return loss_cls, logits, preds, targets
