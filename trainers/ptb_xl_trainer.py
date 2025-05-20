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

class TrainingPTB_XL(L.LightningModule):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__()
        self.lr_head = config.lr_head
        self.lr_xlstm = config.lr_xlstm
        self.wd = config.wd
        self.model = model
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.weights = weights
        self.use_scheduler = config.use_scheduler
        self.len_train_dataset = len_train_dataset
        self.num_epochs_warmup = config.num_epochs_warmup
        self.sched_decay_factor = config.sched_decay_factor
        self.num_epochs_warm_restart = config.num_epochs_warm_restart
        self.label_smoothing = config.label_smoothing
        self.epochs = config.epochs
        self.use_focal_loss = config.use_focal_loss
        self.linear_probing = config.linear_probing

        self.classification_taksk = config.classification_task

        self.num_classes = 23 if self.classification_taksk == 'diagnosis_subclass' else 5

        self.train_acc = torchmetrics.classification.accuracy.MultilabelAccuracy(num_labels=self.num_classes, average='micro', ignore_index=-1)
        self.valid_acc = torchmetrics.classification.accuracy.MultilabelAccuracy(num_labels=self.num_classes, average='micro', ignore_index=-1)
        self.test_acc = torchmetrics.classification.accuracy.MultilabelAccuracy(num_labels=self.num_classes, average='micro', ignore_index=-1)
        self.test_acc_no_avg = torchmetrics.classification.accuracy.MultilabelAccuracy(num_labels=self.num_classes,  average=None, ignore_index=-1)
        self.train_f1 = torchmetrics.classification.MultilabelF1Score(num_labels=self.num_classes, average='macro', ignore_index=-1)
        self.valid_f1 = torchmetrics.classification.MultilabelF1Score(num_labels=self.num_classes, average='macro', ignore_index=-1)
        self.test_f1 = torchmetrics.classification.MultilabelF1Score(num_labels=self.num_classes, average=None, ignore_index=-1)
        self.test_f1_macro = torchmetrics.classification.MultilabelF1Score(num_labels=self.num_classes, average='macro', ignore_index=-1)
        self.train_auroc = torchmetrics.classification.MultilabelAUROC(num_labels=self.num_classes, compute_on_step=False, ignore_index=-1)  
        self.valid_auroc = torchmetrics.classification.MultilabelAUROC(num_labels=self.num_classes, compute_on_step=False, ignore_index=-1)
        self.test_auroc = torchmetrics.classification.MultilabelAUROC(num_labels=self.num_classes, compute_on_step=False, ignore_index=-1)

        # add sensitivity and specificity for the first class
        self.val_spec = torchmetrics.classification.specificity.MultilabelSpecificity(num_labels=self.num_classes, average=None, ignore_index=-1)
        self.test_spec = torchmetrics.classification.specificity.MultilabelSpecificity(num_labels=self.num_classes, average=None, ignore_index=-1)
        self.val_recall = torchmetrics.classification.precision_recall.MultilabelRecall(num_labels=self.num_classes, average=None, ignore_index=-1)
        self.test_recall = torchmetrics.classification.precision_recall.MultilabelRecall(num_labels=self.num_classes, average=None, ignore_index=-1)
        self.val_precision = torchmetrics.classification.precision_recall.MultilabelPrecision(num_labels=self.num_classes, average=None, ignore_index=-1)
        self.test_precision = torchmetrics.classification.precision_recall.MultilabelPrecision(num_labels=self.num_classes, average=None, ignore_index=-1)

        if not config.is_sweep:
            self.save_hyperparameters()

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

        # specificity
        self.val_spec = self.val_spec.to(preds.device)
        self.val_spec(preds, targets)
        self.log('val_specificity/STTC', self.val_spec[0], prog_bar=False, metric_attribute='val_spec')
        self.log('val_specificity/NORM', self.val_spec[1], prog_bar=False, metric_attribute='val_spec')
        self.log('val_specificity/MI', self.val_spec[2], prog_bar=False, metric_attribute='val_spec')
        self.log('val_specificity/HYP', self.val_spec[3], prog_bar=False, metric_attribute='val_spec')
        self.log('val_specificity/CD', self.val_spec[4], prog_bar=False, metric_attribute='val_spec')

        # sensitivity
        self.val_recall = self.val_recall.to(preds.device)
        self.val_recall(preds, targets)
        self.log('val_sensitivity/STTC', self.val_recall[0], prog_bar=False, metric_attribute='val_recall')
        self.log('val_sensitivity/NORM', self.val_recall[1], prog_bar=False, metric_attribute='val_recall')
        self.log('val_sensitivity/MI', self.val_recall[2], prog_bar=False, metric_attribute='val_recall')
        self.log('val_sensitivity/HYP', self.val_recall[3], prog_bar=False, metric_attribute='val_recall')
        self.log('val_sensitivity/CD', self.val_recall[4], prog_bar=False, metric_attribute='val_recall')

        # ppv
        self.val_precision = self.val_precision.to(preds.device)
        self.val_precision(preds, targets)
        self.log('val_ppv/STTC', self.val_precision[0], prog_bar=False, metric_attribute='val_precision')
        self.log('val_ppv/NORM', self.val_precision[1], prog_bar=False, metric_attribute='val_precision')
        self.log('val_ppv/MI', self.val_precision[2], prog_bar=False, metric_attribute='val_precision')
        self.log('val_ppv/HYP', self.val_precision[3], prog_bar=False, metric_attribute='val_precision')
        self.log('val_ppv/CD', self.val_precision[4], prog_bar=False, metric_attribute='val_precision')

        # auroc
        self.valid_auroc = self.valid_auroc.to(logits.device)
        self.valid_auroc(logits, targets)
        self.log('val_auroc', self.valid_auroc, prog_bar=True)

        return loss
            
    def test_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        self.test_acc = self.test_acc.to(preds.device)
        self.test_acc(preds, targets)

        self.test_acc_no_avg = self.test_acc_no_avg.to(preds.device)
        self.test_acc_no_avg(preds, targets)


        self.test_f1 = self.test_f1.to(preds.device)
        self.test_f1(preds, targets)
        self.test_f1_macro = self.test_f1_macro.to(preds.device)
        self.test_f1_macro(preds, targets)

        self.log("test_loss", loss.detach().item())

        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1_macro)

        # accuracy
        self.log("test_acc/STTC", self.test_acc_no_avg[0], metric_attribute='test_acc')
        self.log("test_acc/NORM", self.test_acc_no_avg[1], metric_attribute='test_acc')
        self.log("test_acc/MI", self.test_acc_no_avg[2], metric_attribute='test_acc')
        self.log("test_acc/HYP", self.test_acc_no_avg[3], metric_attribute='test_acc')
        self.log("test_acc/CD", self.test_acc_no_avg[4], metric_attribute='test_acc')

        # f1
        self.log("test_f1/STTC", self.test_f1[0], metric_attribute='test_f1')
        self.log("test_f1/NORM", self.test_f1[1], metric_attribute='test_f1')
        self.log("test_f1/MI", self.test_f1[2], metric_attribute='test_f1')
        self.log("test_f1/HYP", self.test_f1[3], metric_attribute='test_f1')
        self.log("test_f1/CD", self.test_f1[4], metric_attribute='test_f1')
        self.log("test_f1/mean", (self.test_f1[0] + self.test_f1[1] + self.test_f1[2] + self.test_f1[3] + self.test_f1[4]) / 5, metric_attribute='test_f1')

        # specificity
        self.test_spec = self.test_spec.to(preds.device)
        self.test_spec(preds, targets)
        self.log("test_specificity/STTC", self.test_spec[0], metric_attribute='test_spec')
        self.log("test_specificity/NORM", self.test_spec[1], metric_attribute='test_spec')
        self.log("test_specificity/MI", self.test_spec[2], metric_attribute='test_spec')
        self.log("test_specificity/HYP", self.test_spec[3], metric_attribute='test_spec')
        self.log("test_specificity/CD", self.test_spec[4], metric_attribute='test_spec')

        # sensitivity
        self.test_recall = self.test_recall.to(preds.device)
        self.test_recall(preds, targets)
        self.log("test_sensitivity/STTC", self.test_recall[0], metric_attribute='test_recall')
        self.log("test_sensitivity/NORM", self.test_recall[1], metric_attribute='test_recall')
        self.log("test_sensitivity/MI", self.test_recall[2], metric_attribute='test_recall')
        self.log("test_sensitivity/HYP", self.test_recall[3], metric_attribute='test_recall')
        self.log("test_sensitivity/CD", self.test_recall[4], metric_attribute='test_recall')

        # ppv
        self.test_precision = self.test_precision.to(preds.device)
        self.test_precision(preds, targets)
        self.log("test_ppv/STTC", self.test_precision[0], metric_attribute='test_precision')
        self.log("test_ppv/NORM", self.test_precision[1], metric_attribute='test_precision')
        self.log("test_ppv/MI", self.test_precision[2], metric_attribute='test_precision')
        self.log("test_ppv/HYP", self.test_precision[3], metric_attribute='test_precision')
        self.log("test_ppv/CD", self.test_precision[4], metric_attribute='test_precision')

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

    def get_params(self):
        if self.linear_probing:
            return [
                {'params': self.model.training_params(), 'lr': self.lr_head, 'weight_decay': self.wd},
            ]

        return [
            {'params': self.model.training_params(), 'lr': self.lr_head, 'weight_decay': self.wd},
            {'params': self.model.finetuning_params(), 'lr': self.lr_xlstm, 'weight_decay': self.wd}
        ]
    
    def get_lr(self):
        return self.lr_head
        
    def configure_optimizers(self):
        return common.configure_optimizers(self)
