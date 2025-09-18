from torch import optim, nn
import lightning as pl
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torchmetrics.classification.precision_recall
import torchmetrics.classification.specificity
import numpy as np
import torch
import trainers.common as common
from trainers.common_trainer import CommonTrainerDownstream
from utils.loss_utils import focal_loss


class TrainingDeepBeat(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        top_k = 1

        self.train_acc = torchmetrics.Accuracy(num_labels=self.num_classes, num_classes=self.num_classes, average='micro', ignore_index=-1, task=self.task, top_k=top_k)
        self.valid_acc = torchmetrics.Accuracy(num_labels=self.num_classes, num_classes=self.num_classes, average='micro', ignore_index=-1, task=self.task, top_k=top_k)
        self.test_acc = torchmetrics.Accuracy(num_labels=self.num_classes, num_classes=self.num_classes,average='micro', ignore_index=-1, task=self.task, top_k=top_k)

        self.train_f1 = torchmetrics.F1Score(num_labels=self.num_classes, num_classes=self.num_classes, average='macro', ignore_index=-1, task=self.task, top_k=top_k)
        self.valid_f1 = torchmetrics.F1Score(num_labels=self.num_classes, num_classes=self.num_classes, average='macro', ignore_index=-1, task=self.task, top_k=top_k)
        self.test_f1 = torchmetrics.F1Score(num_labels=self.num_classes, num_classes=self.num_classes, average='macro', ignore_index=-1, task=self.task, top_k=top_k)

        self.train_auprc = torchmetrics.AveragePrecision(num_classes=self.num_classes, num_labels=self.num_classes, ignore_index=-1, task=config.task)
        self.valid_auprc = torchmetrics.AveragePrecision(num_classes=self.num_classes, num_labels=self.num_classes, ignore_index=-1, task=config.task)
        self.test_auprc = torchmetrics.AveragePrecision(num_classes=self.num_classes, num_labels=self.num_classes, ignore_index=-1, task=config.task)

        self.train_auroc = torchmetrics.AUROC(num_labels=self.num_classes, num_classes=self.num_classes, average='macro', ignore_index=-1, task=self.task)
        self.valid_auroc = torchmetrics.AUROC(num_labels=self.num_classes, num_classes=self.num_classes, average='macro', ignore_index=-1, task=self.task)
        self.test_auroc = torchmetrics.AUROC(num_labels=self.num_classes, num_classes=self.num_classes, average='macro', ignore_index=-1, task=self.task)

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

        # auprc
        self.train_auprc = self.train_auprc.to(logits.device)
        self.train_auprc(logits, targets)
        self.log("train_auprc", self.train_auprc, prog_bar=True)

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

        # auprc
        self.valid_auprc = self.valid_auprc.to(logits.device)
        self.valid_auprc(logits, targets)
        self.log('val_auprc', self.valid_auprc, prog_bar=True)

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

        # auprc
        self.test_auprc = self.test_auprc.to(logits.device)
        self.test_auprc(logits, targets)
        self.log("test_auprc", self.test_auprc)


        return loss
    
    def predict_batch(self, batch):
        x = batch["signal"]
        
        targets = batch['label']

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        logits = self.model(x)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        targets = torch.argmax(targets, dim=1)
        if self.use_focal_loss:
            loss = nn.functional.cross_entropy(logits, targets, weight=self.weights, reduction='none')
            loss = focal_loss(loss)
        else:
            loss = nn.functional.cross_entropy(logits, targets)
        preds = torch.argmax(logits, dim=1)

        return loss, logits, preds, targets
