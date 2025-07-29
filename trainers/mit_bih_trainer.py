from torch import optim, nn
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torchmetrics.classification.precision_recall
import torchmetrics.classification.specificity
import numpy as np
import torch
import trainers.common as common
from trainers.common_trainer import CommonTrainerDownstream


class TrainingMIT_BIH(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)
            

        self.train_acc = torchmetrics.classification.accuracy.MulticlassAccuracy(num_classes=self.num_classes, average='micro', ignore_index=-1)
        self.valid_acc = torchmetrics.classification.accuracy.MulticlassAccuracy(num_classes=self.num_classes, average='micro', ignore_index=-1)
        self.test_acc = torchmetrics.classification.accuracy.MulticlassAccuracy(num_classes=self.num_classes, average='micro', ignore_index=-1)
        self.test_acc_no_avg = torchmetrics.classification.accuracy.MulticlassAccuracy(num_classes=self.num_classes, average=None, ignore_index=-1)
        self.train_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average='macro', ignore_index=-1)
        self.valid_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average='macro', ignore_index=-1)
        self.test_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average=None, ignore_index=-1)
        self.train_auroc = torchmetrics.classification.AUROC(num_classes=self.num_classes, compute_on_step=False, ignore_index=-1)  
        self.valid_auroc = torchmetrics.classification.AUROC(num_classes=self.num_classes, compute_on_step=False, ignore_index=-1)
        self.test_auroc = torchmetrics.classification.AUROC(num_classes=self.num_classes, compute_on_step=False, ignore_index=-1)

        self.single_hb = config.use_ecg_founder

        # add sensitivity and specificity for the first class
        self.val_spec = torchmetrics.classification.specificity.MulticlassSpecificity(num_classes=self.num_classes, average=None, ignore_index=-1)
        self.test_spec = torchmetrics.classification.specificity.MulticlassSpecificity(num_classes=self.num_classes, average=None, ignore_index=-1)
        self.val_recall = torchmetrics.classification.precision_recall.MulticlassRecall(num_classes=self.num_classes, average=None, ignore_index=-1)
        self.test_recall = torchmetrics.classification.precision_recall.MulticlassRecall(num_classes=self.num_classes, average=None, ignore_index=-1)
        self.val_precision = torchmetrics.classification.precision_recall.MulticlassPrecision(num_classes=self.num_classes, average=None, ignore_index=-1)
        self.test_precision = torchmetrics.classification.precision_recall.MulticlassPrecision(num_classes=self.num_classes, average=None, ignore_index=-1)

    def training_step(self, batch, _):
        loss_cls, preds, targets, logits = self.predict_batch(batch)

        self.train_acc = self.train_acc.to(preds.device)
        self.train_acc(preds, targets)

        self.train_f1 = self.train_f1.to(preds.device)
        self.train_f1(preds, targets)

        self.log('train_loss', loss_cls.detach().item(), prog_bar=True)


        self.log('train_acc', self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)

        # auroc
        self.train_auroc = self.train_auroc.to(logits.device)
        self.train_auroc(logits, targets)
        self.log("train_auroc", self.train_auroc)

        return loss_cls 
    
    def validation_step(self, batch, _):
        loss_cls, preds, targets, logits = self.predict_batch(batch)

        self.valid_acc = self.valid_acc.to(preds.device)
        self.valid_acc(preds, targets)

        self.valid_f1 = self.valid_f1.to(preds.device)
        self.valid_f1(preds, targets)

        self.log('val_loss', loss_cls.detach().item(), prog_bar=True)

        self.log('val_acc', self.valid_acc, prog_bar=True)
        self.log('val_f1', self.valid_f1, prog_bar=True)

        # specificity
        self.val_spec = self.val_spec.to(preds.device)
        self.val_spec(preds, targets)
        self.log('val_specificity/N', self.val_spec[0], prog_bar=False, metric_attribute='val_spec')
        self.log('val_specificity/S', self.val_spec[1], prog_bar=False, metric_attribute='val_spec')
        self.log('val_specificity/V', self.val_spec[2], prog_bar=False, metric_attribute='val_spec')
        if self.num_classes == 5:
            self.log('val_specificity/F', self.val_spec[3], prog_bar=False, metric_attribute='val_spec')
            self.log('val_specificity/Q', self.val_spec[4], prog_bar=False, metric_attribute='val_spec')

        # sensitivity
        self.val_recall = self.val_recall.to(preds.device)
        self.val_recall(preds, targets)
        self.log('val_sensitivity/N', self.val_recall[0], prog_bar=False, metric_attribute='val_recall')
        self.log('val_sensitivity/S', self.val_recall[1], prog_bar=False, metric_attribute='val_recall')
        self.log('val_sensitivity/V', self.val_recall[2], prog_bar=False, metric_attribute='val_recall')
        if self.num_classes == 5:
            self.log('val_sensitivity/F', self.val_recall[3], prog_bar=False, metric_attribute='val_recall')
            self.log('val_sensitivity/Q', self.val_recall[4], prog_bar=False, metric_attribute='val_recall')

        # ppv
        self.val_precision = self.val_precision.to(preds.device)
        self.val_precision(preds, targets)
        self.log('val_ppv/N', self.val_precision[0], prog_bar=False, metric_attribute='val_precision')
        self.log('val_ppv/S', self.val_precision[1], prog_bar=False, metric_attribute='val_precision')
        self.log('val_ppv/V', self.val_precision[2], prog_bar=False, metric_attribute='val_precision')
        if self.num_classes == 5:
            self.log('val_ppv/F', self.val_precision[3], prog_bar=False, metric_attribute='val_precision')
            self.log('val_ppv/Q', self.val_precision[4], prog_bar=False, metric_attribute='val_precision')

        # auroc
        self.valid_auroc = self.valid_auroc.to(logits.device)
        self.valid_auroc(logits, targets)
        self.log('val_auroc', self.valid_auroc, prog_bar=True)

        return loss_cls
            
    def test_step(self, batch, _):
        loss_cls, preds, targets, logits = self.predict_batch(batch)

        self.test_acc = self.test_acc.to(preds.device)
        self.test_acc(preds, targets)

        self.test_acc_no_avg = self.test_acc_no_avg.to(preds.device)
        self.test_acc_no_avg(preds, targets)

        self.test_f1 = self.test_f1.to(preds.device)
        self.test_f1(preds, targets)

        self.log("test_loss", loss_cls.detach().item())


        self.log("test_acc", self.test_acc)

        # accuracy
        self.log("test_acc/N", self.test_acc_no_avg[0], metric_attribute='test_acc')
        self.log("test_acc/S", self.test_acc_no_avg[1], metric_attribute='test_acc')
        self.log("test_acc/V", self.test_acc_no_avg[2], metric_attribute='test_acc')
        if self.num_classes == 5:
            self.log("test_acc/F", self.test_acc_no_avg[3], metric_attribute='test_acc')
            self.log("test_acc/Q", self.test_acc_no_avg[4], metric_attribute='test_acc')

        # f1
        self.log("test_f1/N", self.test_f1[0], metric_attribute='test_f1')
        self.log("test_f1/S", self.test_f1[1], metric_attribute='test_f1')
        self.log("test_f1/V", self.test_f1[2], metric_attribute='test_f1')
        if self.num_classes == 5:
            self.log("test_f1/F", self.test_f1[3], metric_attribute='test_f1')
            self.log("test_f1/Q", self.test_f1[4], metric_attribute='test_f1')
            self.log("test_f1/mean", (self.test_f1[0] + self.test_f1[1] + self.test_f1[2] + self.test_f1[3] + self.test_f1[4]) / 5, metric_attribute='test_f1')
        else:
            self.log("test_f1/mean", (self.test_f1[0] + self.test_f1[1] + self.test_f1[2]) / 3, metric_attribute='test_f1')

        # specificity
        self.test_spec = self.test_spec.to(preds.device)
        self.test_spec(preds, targets)
        self.log("test_specificity/N", self.test_spec[0], metric_attribute='test_spec')
        self.log("test_specificity/S", self.test_spec[1], metric_attribute='test_spec')
        self.log("test_specificity/V", self.test_spec[2], metric_attribute='test_spec')
        if self.num_classes == 5:
            self.log("test_specificity/F", self.test_spec[3], metric_attribute='test_spec')
            self.log("test_specificity/Q", self.test_spec[4], metric_attribute='test_spec')
            self.log("test_specificity/mean", (self.test_spec[0] + self.test_spec[1] + self.test_spec[2] + self.test_spec[3] + self.test_spec[4]) / 5, metric_attribute='test_spec')
        else:
            self.log("test_specificity/mean", (self.test_spec[0] + self.test_spec[1] + self.test_spec[2]) / 3, metric_attribute='test_spec')

        # sensitivity
        self.test_recall = self.test_recall.to(preds.device)
        self.test_recall(preds, targets)
        self.log("test_sensitivity/N", self.test_recall[0], metric_attribute='test_recall')
        self.log("test_sensitivity/S", self.test_recall[1], metric_attribute='test_recall')
        self.log("test_sensitivity/V", self.test_recall[2], metric_attribute='test_recall')
        if self.num_classes == 5:
            self.log("test_sensitivity/F", self.test_recall[3], metric_attribute='test_recall')
            self.log("test_sensitivity/Q", self.test_recall[4], metric_attribute='test_recall')
            self.log("test_sensitivity/mean", (self.test_recall[0] + self.test_recall[1] + self.test_recall[2] + self.test_recall[3] + self.test_recall[4]) / 5, metric_attribute='test_recall')
        else:
            self.log("test_sensitivity/mean", (self.test_recall[0] + self.test_recall[1] + self.test_recall[2]) / 3, metric_attribute='test_recall')

        # ppv
        self.test_precision = self.test_precision.to(preds.device)
        self.test_precision(preds, targets)
        self.log("test_ppv/N", self.test_precision[0], metric_attribute='test_precision')
        self.log("test_ppv/S", self.test_precision[1], metric_attribute='test_precision')
        self.log("test_ppv/V", self.test_precision[2], metric_attribute='test_precision')
        if self.num_classes == 5:
            self.log("test_ppv/F", self.test_precision[3], metric_attribute='test_precision')
            self.log("test_ppv/Q", self.test_precision[4], metric_attribute='test_precision')
            self.log("test_ppv/mean", (self.test_precision[0] + self.test_precision[1] + self.test_precision[2] + self.test_precision[3] + self.test_precision[4]) / 5, metric_attribute='test_precision')
        else:
            self.log("test_ppv/mean", (self.test_precision[0] + self.test_precision[1] + self.test_precision[2]) / 3, metric_attribute='test_precision')

        # auroc  
        self.test_auroc = self.test_auroc.to(logits.device)
        self.test_auroc(logits, targets)
        self.log("test_auroc", self.test_auroc)

        return loss_cls 
    
    def predict_batch(self, batch):
        x = batch["signal"]

        if self.linear_probing:
            self.model.set_eval_linear_probing()

        if not self.single_hb:
            targets = batch['label'].unfold(1, self.model.patch_size, self.model.patch_size).max(dim=-1)[0].long()
            # get one hot encoding

            cls = self.model(x)

            if cls.shape[1] > targets.shape[1]:
                print(f"cls shape: {cls.shape}, targets shape: {targets.shape}")
                cls = cls[:, :targets.shape[1], :]

            loss_cls = nn.functional.cross_entropy(cls.permute(0, 2, 1), targets, weight=self.weights, ignore_index=-1)
        else:
            targets = batch['label'].long()
            cls = self.model(x).unsqueeze(1)  # [bs, 1, num_classes]
            loss_cls = nn.functional.cross_entropy(cls.permute(0, 2, 1), targets, weight=self.weights,  ignore_index=-1)


        # need to transform the targets to [batch_size, num_patches] where if all the values are -1, then the value is -1 if not is the only value non -1
        preds = torch.argmax(cls, dim=-1)
        
        if self.use_focal_loss:
            pt = torch.exp(-loss_cls)
            alpha = 2.
            gamma = .25
            loss_cls = (alpha * (1-pt)**gamma * loss_cls)
        

        # return the masked target and cls
        mask = targets != -1
        targets = targets[mask]
        cls = cls[mask]
        preds = preds[mask]
        
        return loss_cls, preds, targets, cls 