from torch import nn
import torchmetrics
import torch

from .common_trainer import CommonTrainerDownstream
from ..utils.loss_utils import focal_loss



class TrainingPTB_XL(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.classification_task = config.classification_task
        self.num_classes = config.num_classes

        top_k = 1 if self.task == 'multiclass' else None

        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, num_labels=self.num_classes, average='micro', ignore_index=-1, task=config.task, top_k=top_k)
        self.valid_acc = torchmetrics.Accuracy(num_classes=self.num_classes, num_labels=self.num_classes, average='micro', ignore_index=-1,task=config.task, top_k=top_k)
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, num_labels=self.num_classes, average='micro', ignore_index=-1,task=config.task, top_k=top_k)
        self.train_f1 = torchmetrics.F1Score(num_classes=self.num_classes, num_labels=self.num_classes, average='macro', ignore_index=-1, task=config.task, top_k=top_k)
        self.valid_f1 = torchmetrics.F1Score(num_classes=self.num_classes, num_labels=self.num_classes, average='macro', ignore_index=-1,task=config.task, top_k=top_k)
        self.test_f1_macro = torchmetrics.F1Score(num_classes=self.num_classes, num_labels=self.num_classes, average='macro', ignore_index=-1,task=config.task, top_k=top_k)
        
        self.train_auroc = torchmetrics.AUROC(num_classes=self.num_classes, num_labels=self.num_classes, average='macro', ignore_index=-1, task=config.task)
        self.valid_auroc = torchmetrics.AUROC(num_classes=self.num_classes, num_labels=self.num_classes, average='macro', ignore_index=-1, task=config.task)
        self.test_auroc = torchmetrics.AUROC(num_classes=self.num_classes, num_labels=self.num_classes, average='macro', ignore_index=-1, task=config.task)

        # auprc
        self.train_auprc = torchmetrics.AveragePrecision(num_classes=self.num_classes, num_labels=self.num_classes, ignore_index=-1, task=config.task)
        self.valid_auprc = torchmetrics.AveragePrecision(num_classes=self.num_classes, num_labels=self.num_classes, ignore_index=-1, task=config.task)
        self.test_auprc = torchmetrics.AveragePrecision(num_classes=self.num_classes, num_labels=self.num_classes, ignore_index=-1, task=config.task)

        # add sensitivity and specificity for the first class
        # self.val_spec = torchmetrics.Specificity(num_classes=self.num_classes, num_labels=self.num_classes, average=None, ignore_index=-1,task=config.task, top_k=top_k)
        # self.test_spec = torchmetrics.Specificity(num_classes=self.num_classes, num_labels=self.num_classes, average=None, ignore_index=-1,task=config.task, top_k=top_k)
        # self.val_recall = torchmetrics.Recall(num_classes=self.num_classes, num_labels=self.num_classes, average=None, ignore_index=-1,task=config.task, top_k=top_k)
        # self.test_recall = torchmetrics.Recall(num_classes=self.num_classes, num_labels=self.num_classes, average=None, ignore_index=-1,task=config.task, top_k=top_k)
        # self.val_precision = torchmetrics.Precision(num_classes=self.num_classes, num_labels=self.num_classes, average=None, ignore_index=-1,task=config.task, top_k=top_k)
        # self.test_precision = torchmetrics.Precision(num_classes=self.num_classes, num_labels=self.num_classes, average=None, ignore_index=-1,task=config.task, top_k=top_k)

        # self.test_acc_no_avg = torchmetrics.Accuracy(num_classes=self.num_classes, num_labels=self.num_classes,  average=None, ignore_index=-1, task=config.task, top_k=top_k)
        # self.test_f1 = torchmetrics.F1Score(num_classes=self.num_classes, num_labels=self.num_classes, average=None, ignore_index=-1,task=config.task, top_k=top_k)


    def training_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        self.train_acc = self.train_acc.to(preds.device)
        self.train_acc(preds, targets)

        self.train_f1 = self.train_f1.to(preds.device)
        self.train_f1(preds, targets)

        self.log('train_loss', loss.detach().item(), prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=False)
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
        self.log('val_acc', self.valid_acc, prog_bar=False)
        self.log('val_f1', self.valid_f1, prog_bar=True)

        # specificity
        # self.val_spec = self.val_spec.to(preds.device)
        # self.val_spec(preds, targets)
        # self.log('val_specificity/STTC', self.val_spec[0], prog_bar=False, metric_attribute='val_spec')
        # self.log('val_specificity/NORM', self.val_spec[1], prog_bar=False, metric_attribute='val_spec')
        # self.log('val_specificity/MI', self.val_spec[2], prog_bar=False, metric_attribute='val_spec')
        # self.log('val_specificity/HYP', self.val_spec[3], prog_bar=False, metric_attribute='val_spec')
        # self.log('val_specificity/CD', self.val_spec[4], prog_bar=False, metric_attribute='val_spec')

        # # sensitivity
        # self.val_recall = self.val_recall.to(preds.device)
        # self.val_recall(preds, targets)
        # self.log('val_sensitivity/STTC', self.val_recall[0], prog_bar=False, metric_attribute='val_recall')
        # self.log('val_sensitivity/NORM', self.val_recall[1], prog_bar=False, metric_attribute='val_recall')
        # self.log('val_sensitivity/MI', self.val_recall[2], prog_bar=False, metric_attribute='val_recall')
        # self.log('val_sensitivity/HYP', self.val_recall[3], prog_bar=False, metric_attribute='val_recall')
        # self.log('val_sensitivity/CD', self.val_recall[4], prog_bar=False, metric_attribute='val_recall')

        # # ppv
        # self.val_precision = self.val_precision.to(preds.device)
        # self.val_precision(preds, targets)
        # self.log('val_ppv/STTC', self.val_precision[0], prog_bar=False, metric_attribute='val_precision')
        # self.log('val_ppv/NORM', self.val_precision[1], prog_bar=False, metric_attribute='val_precision')
        # self.log('val_ppv/MI', self.val_precision[2], prog_bar=False, metric_attribute='val_precision')
        # self.log('val_ppv/HYP', self.val_precision[3], prog_bar=False, metric_attribute='val_precision')
        # self.log('val_ppv/CD', self.val_precision[4], prog_bar=False, metric_attribute='val_precision')

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

        # self.test_acc_no_avg = self.test_acc_no_avg.to(preds.device)
        # self.test_acc_no_avg(preds, targets)

        # self.test_f1 = self.test_f1.to(preds.device)
        # self.test_f1(preds, targets)
        self.test_f1_macro = self.test_f1_macro.to(preds.device)
        self.test_f1_macro(preds, targets)

        self.log("test_loss", loss.detach().item())

        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1_macro)

        # accuracy
        # self.log("test_acc/STTC", self.test_acc_no_avg[0], metric_attribute='test_acc')
        # self.log("test_acc/NORM", self.test_acc_no_avg[1], metric_attribute='test_acc')
        # self.log("test_acc/MI", self.test_acc_no_avg[2], metric_attribute='test_acc')
        # self.log("test_acc/HYP", self.test_acc_no_avg[3], metric_attribute='test_acc')
        # self.log("test_acc/CD", self.test_acc_no_avg[4], metric_attribute='test_acc')

        # f1
        # self.log("test_f1/STTC", self.test_f1[0], metric_attribute='test_f1')
        # self.log("test_f1/NORM", self.test_f1[1], metric_attribute='test_f1')
        # self.log("test_f1/MI", self.test_f1[2], metric_attribute='test_f1')
        # self.log("test_f1/HYP", self.test_f1[3], metric_attribute='test_f1')
        # self.log("test_f1/CD", self.test_f1[4], metric_attribute='test_f1')
        # self.log("test_f1/mean", (self.test_f1[0] + self.test_f1[1] + self.test_f1[2] + self.test_f1[3] + self.test_f1[4]) / 5, metric_attribute='test_f1')

        # specificity
        # self.test_spec = self.test_spec.to(preds.device)
        # self.test_spec(preds, targets)
        # self.log("test_specificity/STTC", self.test_spec[0], metric_attribute='test_spec')
        # self.log("test_specificity/NORM", self.test_spec[1], metric_attribute='test_spec')
        # self.log("test_specificity/MI", self.test_spec[2], metric_attribute='test_spec')
        # self.log("test_specificity/HYP", self.test_spec[3], metric_attribute='test_spec')
        # self.log("test_specificity/CD", self.test_spec[4], metric_attribute='test_spec')

        # sensitivity
        # self.test_recall = self.test_recall.to(preds.device)
        # self.test_recall(preds, targets)
        # self.log("test_sensitivity/STTC", self.test_recall[0], metric_attribute='test_recall')
        # self.log("test_sensitivity/NORM", self.test_recall[1], metric_attribute='test_recall')
        # self.log("test_sensitivity/MI", self.test_recall[2], metric_attribute='test_recall')
        # self.log("test_sensitivity/HYP", self.test_recall[3], metric_attribute='test_recall')
        # self.log("test_sensitivity/CD", self.test_recall[4], metric_attribute='test_recall')

        # ppv
        # self.test_precision = self.test_precision.to(preds.device)
        # self.test_precision(preds, targets)
        # self.log("test_ppv/STTC", self.test_precision[0], metric_attribute='test_precision')
        # self.log("test_ppv/NORM", self.test_precision[1], metric_attribute='test_precision')
        # self.log("test_ppv/MI", self.test_precision[2], metric_attribute='test_precision')
        # self.log("test_ppv/HYP", self.test_precision[3], metric_attribute='test_precision')
        # self.log("test_ppv/CD", self.test_precision[4], metric_attribute='test_precision')

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
        x = batch["signals"]
        targets = batch['class_labels']
        # get one hot encoding

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        logits = self.model(x)

        if self.task == 'multiclass':
            targets = torch.argmax(targets, dim=1)
            if self.use_focal_loss:
                loss = nn.functional.cross_entropy(logits, targets, weight=self.weights, reduction='none')
                loss = focal_loss(loss)
            else:
                loss = nn.functional.cross_entropy(logits, targets, weight=self.weights)
            preds = torch.argmax(logits, dim=1)

        elif self.task == 'multilabel':
            if self.use_focal_loss:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=self.weights, reduction='none')
                loss = focal_loss(loss)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=self.weights)

            preds = (torch.sigmoid(logits) > 0.5).float()


        return loss, logits, preds, targets
