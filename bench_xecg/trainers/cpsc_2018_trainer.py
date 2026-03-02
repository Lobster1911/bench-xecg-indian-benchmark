from torch import nn
import torchmetrics
import torch

from .common_trainer import CommonTrainerDownstream
from ..utils.loss_utils import focal_loss


class TrainingCPSC_2018(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        top_k = 1 if self.task == 'multiclass' else None

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

        # add a metric to log auc for all the classes:
        self.train_auroc_per_class = torchmetrics.AUROC(num_labels=self.num_classes, num_classes=self.num_classes, average=None, ignore_index=-1, task=self.task)
        self.valid_auroc_per_class = torchmetrics.AUROC(num_labels=self.num_classes, num_classes=self.num_classes, average=None, ignore_index=-1, task=self.task)
        self.test_auroc_per_class = torchmetrics.AUROC(num_labels=self.num_classes, num_classes=self.num_classes, average=None, ignore_index=-1, task=self.task)

    def training_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)
        self.log('train_loss', loss.detach().item(), prog_bar=True)

        self.train_acc(preds, targets)
        self.log('train_acc', self.train_acc, prog_bar=True)

        self.train_f1(preds, targets)
        self.log('train_f1', self.train_f1, prog_bar=True)

        self.train_auroc(logits, targets)
        self.log("train_auroc", self.train_auroc)

        self.train_auprc(logits, targets)
        self.log("train_auprc", self.train_auprc, prog_bar=True)

        auroc_scores = self.train_auroc_per_class(logits, targets)
        self.log_dict({f"AUC/train_class_{i}": score for i, score in enumerate(auroc_scores)})

        return loss
    
    def validation_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)
        self.log('val_loss', loss.detach().item(), prog_bar=True)

        self.valid_acc(preds, targets)
        self.log('val_acc', self.valid_acc, prog_bar=True)

        self.valid_f1(preds, targets)
        self.log('val_f1', self.valid_f1, prog_bar=True)

        self.valid_auroc(logits, targets)
        self.log('val_auroc', self.valid_auroc, prog_bar=True)

        self.valid_auprc(logits, targets)
        self.log('val_auprc', self.valid_auprc, prog_bar=True)

        auroc_scores = self.valid_auroc_per_class(logits, targets)
        self.log_dict({f"AUC/val_class_{i}": score for i, score in enumerate(auroc_scores)})

        return loss
            
    def test_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)
        self.log("test_loss", loss.detach().item())

        self.test_acc(preds, targets)
        self.log("test_acc", self.test_acc)

        self.test_f1(preds, targets)
        self.log("test_f1", self.test_f1)

        self.test_auroc(logits, targets)
        self.log("test_auroc", self.test_auroc)

        self.test_auprc(logits, targets)
        self.log("test_auprc", self.test_auprc)

        auroc_scores = self.test_auroc_per_class(logits, targets)
        self.log_dict({f"AUC/test_class_{i}": score for i, score in enumerate(auroc_scores)})

        return loss
            
    
    def predict_batch(self, batch):
        x = batch["signals"]
        targets = batch['class_labels']
        # get one hot encoding

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        logits = self.model(x)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        if self.task == 'multiclass':
            targets = torch.argmax(targets, dim=1)
            if self.use_focal_loss:
                loss = nn.functional.cross_entropy(logits, targets, weight=self.weights, reduction='none')
                loss = focal_loss(loss)
            else:
                loss = nn.functional.cross_entropy(logits, targets)
            preds = torch.argmax(logits, dim=1)

        elif self.task == 'multilabel':
            if self.use_focal_loss:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=self.weights, reduction='none')
                loss = focal_loss(loss)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=self.weights)

            preds = (torch.sigmoid(logits) > 0.5).float()
        
        return loss, logits, preds, targets
