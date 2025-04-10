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

class TrainingMIT_BIH(L.LightningModule):
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
        self.r_peaks_lambda = config.r_peaks_lambda
        self.use_focal_loss = config.use_focal_loss


        self.num_classes = config.num_classes
        self.train_acc = torchmetrics.classification.accuracy.MulticlassAccuracy(num_classes=self.num_classes, top_k=1, average='micro', ignore_index=-1)
        self.valid_acc = torchmetrics.classification.accuracy.MulticlassAccuracy(num_classes=self.num_classes, top_k=1, average='micro', ignore_index=-1)
        self.test_acc = torchmetrics.classification.accuracy.MulticlassAccuracy(num_classes=self.num_classes, top_k=1, average='micro', ignore_index=-1)
        self.test_acc_no_avg = torchmetrics.classification.accuracy.MulticlassAccuracy(num_classes=self.num_classes, top_k=1, average=None, ignore_index=-1)
        self.train_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, top_k=1, average='macro', ignore_index=-1)
        self.valid_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, top_k=1, average='macro', ignore_index=-1)
        self.test_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, top_k=1, average=None, ignore_index=-1)
        self.train_auroc = torchmetrics.classification.AUROC(num_classes=self.num_classes, compute_on_step=False, ignore_index=-1)  
        self.valid_auroc = torchmetrics.classification.AUROC(num_classes=self.num_classes, compute_on_step=False, ignore_index=-1)
        self.test_auroc = torchmetrics.classification.AUROC(num_classes=self.num_classes, compute_on_step=False, ignore_index=-1)
        self.train_acc_r_peak = torchmetrics.classification.precision_recall.BinaryRecall()
        self.valid_acc_r_peak = torchmetrics.classification.precision_recall.BinaryRecall()
        self.test_acc_r_peak = torchmetrics.classification.precision_recall.BinaryRecall()
        self.train_f1_r_peak = torchmetrics.classification.BinaryF1Score()
        self.valid_f1_r_peak = torchmetrics.classification.BinaryF1Score()
        self.test_f1_r_peak = torchmetrics.classification.BinaryF1Score()

        # add sensitivity and specificity for the first class
        self.val_spec = torchmetrics.classification.specificity.MulticlassSpecificity(num_classes=self.num_classes, average=None, ignore_index=-1)
        self.test_spec = torchmetrics.classification.specificity.MulticlassSpecificity(num_classes=self.num_classes, average=None, ignore_index=-1)
        self.val_recall = torchmetrics.classification.precision_recall.MulticlassRecall(num_classes=self.num_classes, average=None, ignore_index=-1)
        self.test_recall = torchmetrics.classification.precision_recall.MulticlassRecall(num_classes=self.num_classes, average=None, ignore_index=-1)
        self.val_precision = torchmetrics.classification.precision_recall.MulticlassPrecision(num_classes=self.num_classes, average=None, ignore_index=-1)
        self.test_precision = torchmetrics.classification.precision_recall.MulticlassPrecision(num_classes=self.num_classes, average=None, ignore_index=-1)

        if not config.is_sweep:
            self.save_hyperparameters()

    def training_step(self, batch, _):
        loss_cls, loss_r_peak_pos, preds, targets, logits, r_peak_pos, r_peaks = self.predict_batch(batch)

        self.train_acc = self.train_acc.to(preds.device)
        self.train_acc(preds, targets)

        self.train_acc_r_peak = self.train_acc_r_peak.to(r_peak_pos.device)
        self.train_acc_r_peak(r_peak_pos, r_peaks)

        self.train_f1_r_peak = self.train_f1_r_peak.to(r_peak_pos.device)
        self.train_f1_r_peak(r_peak_pos, r_peaks)

        self.train_f1 = self.train_f1.to(preds.device)
        self.train_f1(preds, targets)

        self.log('train_loss', loss_cls.detach().item(), prog_bar=True, batch_size=self.batch_size)

        self.log('train_r_peak_loss', loss_r_peak_pos.detach().item(), prog_bar=True, batch_size=self.batch_size)
        self.log('train_rec_r_peak', self.train_acc_r_peak, prog_bar=True, batch_size=self.batch_size)
        self.log('train_f1_r_peak', self.train_f1_r_peak, prog_bar=True, batch_size=self.batch_size)

        self.log('train_acc', self.train_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('train_f1', self.train_f1, prog_bar=True, batch_size=self.batch_size)

        # auroc
        # self.train_auroc = self.train_auroc.cpu()
        self.train_auroc = self.train_auroc.to(logits.device)
        self.train_auroc(logits, targets)
        self.log("train_auroc", self.train_auroc, batch_size=self.batch_size)

        return loss_cls + loss_r_peak_pos * self.r_peaks_lambda
    
    def validation_step(self, batch, _):
        loss_cls, loss_r_peak_pos, preds, targets, logits, r_peak_pos, r_peaks = self.predict_batch(batch)

        self.valid_acc = self.valid_acc.to(preds.device)
        self.valid_acc(preds, targets)

        self.valid_acc_r_peak = self.valid_acc_r_peak.to(r_peak_pos.device)
        self.valid_acc_r_peak(r_peak_pos, r_peaks)

        self.valid_f1_r_peak = self.valid_f1_r_peak.to(r_peak_pos.device)
        self.valid_f1_r_peak(r_peak_pos, r_peaks)

        self.valid_f1 = self.valid_f1.to(preds.device)
        self.valid_f1(preds, targets)

        self.log('val_loss', loss_cls.detach().item(), prog_bar=True, batch_size=self.batch_size)

        self.log('val_r_peak_loss', loss_r_peak_pos.detach().item(), prog_bar=True, batch_size=self.batch_size)
        self.log('val_rec_r_peak', self.valid_acc_r_peak, prog_bar=True, batch_size=self.batch_size)
        self.log('val_f1_r_peak', self.valid_f1_r_peak, prog_bar=True, batch_size=self.batch_size)

        self.log('val_acc', self.valid_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('val_f1', self.valid_f1, prog_bar=True, batch_size=self.batch_size)

        # specificity
        self.val_spec = self.val_spec.to(preds.device)
        self.val_spec(preds, targets)
        self.log('val_specificity/N', self.val_spec[0], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_spec')
        self.log('val_specificity/S', self.val_spec[1], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_spec')
        self.log('val_specificity/V', self.val_spec[2], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_spec')
        if self.num_classes == 5:
            self.log('val_specificity/F', self.val_spec[3], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_spec')
            self.log('val_specificity/Q', self.val_spec[4], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_spec')

        # sensitivity
        self.val_recall = self.val_recall.to(preds.device)
        self.val_recall(preds, targets)
        self.log('val_sensitivity/N', self.val_recall[0], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_recall')
        self.log('val_sensitivity/S', self.val_recall[1], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_recall')
        self.log('val_sensitivity/V', self.val_recall[2], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_recall')
        if self.num_classes == 5:
            self.log('val_sensitivity/F', self.val_recall[3], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_recall')
            self.log('val_sensitivity/Q', self.val_recall[4], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_recall')

        # ppv
        self.val_precision = self.val_precision.to(preds.device)
        self.val_precision(preds, targets)
        self.log('val_ppv/N', self.val_precision[0], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_precision')
        self.log('val_ppv/S', self.val_precision[1], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_precision')
        self.log('val_ppv/V', self.val_precision[2], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_precision')
        if self.num_classes == 5:
            self.log('val_ppv/F', self.val_precision[3], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_precision')
            self.log('val_ppv/Q', self.val_precision[4], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_precision')

        # auroc
        self.valid_auroc = self.valid_auroc.to(logits.device)
        self.valid_auroc(logits, targets)
        self.log('val_auroc', self.valid_auroc, prog_bar=True, batch_size=self.batch_size)

        return loss_cls + loss_r_peak_pos * self.r_peaks_lambda
            
    def test_step(self, batch, _):
        loss_cls, loss_r_peak_pos, preds, targets, logits, r_peak_pos, r_peaks = self.predict_batch(batch)

        self.test_acc = self.test_acc.to(preds.device)
        self.test_acc(preds, targets)

        self.test_acc_no_avg = self.test_acc_no_avg.to(preds.device)
        self.test_acc_no_avg(preds, targets)

        self.test_acc_r_peak = self.test_acc_r_peak.to(r_peak_pos.device)
        self.test_acc_r_peak(r_peak_pos, r_peaks)

        self.test_f1_r_peak = self.test_f1_r_peak.to(r_peak_pos.device)
        self.test_f1_r_peak(r_peak_pos, r_peaks)

        self.test_f1 = self.test_f1.to(preds.device)
        self.test_f1(preds, targets)

        self.log("test_loss", loss_cls.detach().item(), batch_size=self.batch_size)

        self.log("test_r_peak_loss", loss_r_peak_pos.detach().item(), batch_size=self.batch_size)
        self.log("test_rec_r_peak", self.test_acc_r_peak, batch_size=self.batch_size)
        self.log("test_f1_r_peak", self.test_f1_r_peak, batch_size=self.batch_size)

        self.log("test_acc", self.test_acc, batch_size=self.batch_size)

        # accuracy
        self.log("test_acc/N", self.test_acc_no_avg[0], batch_size=self.batch_size, metric_attribute='test_acc')
        self.log("test_acc/S", self.test_acc_no_avg[1], batch_size=self.batch_size, metric_attribute='test_acc')
        self.log("test_acc/V", self.test_acc_no_avg[2], batch_size=self.batch_size, metric_attribute='test_acc')
        if self.num_classes == 5:
            self.log("test_acc/F", self.test_acc_no_avg[3], batch_size=self.batch_size, metric_attribute='test_acc')
            self.log("test_acc/Q", self.test_acc_no_avg[4], batch_size=self.batch_size, metric_attribute='test_acc')

        # f1
        self.log("test_f1/N", self.test_f1[0], batch_size=self.batch_size, metric_attribute='test_f1')
        self.log("test_f1/S", self.test_f1[1], batch_size=self.batch_size, metric_attribute='test_f1')
        self.log("test_f1/V", self.test_f1[2], batch_size=self.batch_size, metric_attribute='test_f1')
        if self.num_classes == 5:
            self.log("test_f1/F", self.test_f1[3], batch_size=self.batch_size, metric_attribute='test_f1')
            self.log("test_f1/Q", self.test_f1[4], batch_size=self.batch_size, metric_attribute='test_f1')
            self.log("test_f1/mean", (self.test_f1[0] + self.test_f1[1] + self.test_f1[2] + self.test_f1[3] + self.test_f1[4]) / 5, batch_size=self.batch_size, metric_attribute='test_f1')
        else:
            self.log("test_f1/mean", (self.test_f1[0] + self.test_f1[1] + self.test_f1[2]) / 3, batch_size=self.batch_size, metric_attribute='test_f1')

        # specificity
        self.test_spec = self.test_spec.to(preds.device)
        self.test_spec(preds, targets)
        self.log("test_specificity/N", self.test_spec[0], batch_size=self.batch_size, metric_attribute='test_spec')
        self.log("test_specificity/S", self.test_spec[1], batch_size=self.batch_size, metric_attribute='test_spec')
        self.log("test_specificity/V", self.test_spec[2], batch_size=self.batch_size, metric_attribute='test_spec')
        if self.num_classes == 5:
            self.log("test_specificity/F", self.test_spec[3], batch_size=self.batch_size, metric_attribute='test_spec')
            self.log("test_specificity/Q", self.test_spec[4], batch_size=self.batch_size, metric_attribute='test_spec')

        # sensitivity
        self.test_recall = self.test_recall.to(preds.device)
        self.test_recall(preds, targets)
        self.log("test_sensitivity/N", self.test_recall[0], batch_size=self.batch_size, metric_attribute='test_recall')
        self.log("test_sensitivity/S", self.test_recall[1], batch_size=self.batch_size, metric_attribute='test_recall')
        self.log("test_sensitivity/V", self.test_recall[2], batch_size=self.batch_size, metric_attribute='test_recall')
        if self.num_classes == 5:
            self.log("test_sensitivity/F", self.test_recall[3], batch_size=self.batch_size, metric_attribute='test_recall')
            self.log("test_sensitivity/Q", self.test_recall[4], batch_size=self.batch_size, metric_attribute='test_recall')

        # ppv
        self.test_precision = self.test_precision.to(preds.device)
        self.test_precision(preds, targets)
        self.log("test_ppv/N", self.test_precision[0], batch_size=self.batch_size, metric_attribute='test_precision')
        self.log("test_ppv/S", self.test_precision[1], batch_size=self.batch_size, metric_attribute='test_precision')
        self.log("test_ppv/V", self.test_precision[2], batch_size=self.batch_size, metric_attribute='test_precision')
        if self.num_classes == 5:
            self.log("test_ppv/F", self.test_precision[3], batch_size=self.batch_size, metric_attribute='test_precision')
            self.log("test_ppv/Q", self.test_precision[4], batch_size=self.batch_size, metric_attribute='test_precision')

        # auroc  
        self.test_auroc = self.test_auroc.to(logits.device)
        self.test_auroc(logits, targets)
        self.log("test_auroc", self.test_auroc, batch_size=self.batch_size)

        return loss_cls + loss_r_peak_pos * self.r_peaks_lambda
            
    
    def predict_batch(self, batch):
        x = batch["signal"]
        targets = batch['label'].unfold(1, self.model.patch_size, self.model.patch_size).max(dim=-1)[0].long()
        # get one hot encoding

        r_peaks = batch['r_peak'] # [bs, seq_len]
        cls, r_peak_pos = self.model(x)
        r_peak_pos = r_peak_pos.view(r_peak_pos.shape[0], -1)

        # cls is an array with [batch_size, num_patches, num_classes]
        # target is an array with [batch_size, seq_len]

        # need to transform the targets to [batch_size, num_patches] where if all the values are -1, then the value is -1 if not is the only value non -1
        preds = torch.argmax(cls, dim=-1)
        
        loss_cls = nn.functional.cross_entropy(cls.permute(0, 2, 1), targets, weight=self.weights, label_smoothing=self.label_smoothing, ignore_index=-1)
        
        if self.use_focal_loss:
            pt = torch.exp(-loss_cls)
            alpha = 2.
            gamma = .25
            loss_cls = (alpha * (1-pt)**gamma * loss_cls)
        
        r_peaks = r_peaks[:, :r_peak_pos.shape[1]]
        loss_r_peak_pos = nn.functional.binary_cross_entropy_with_logits(r_peak_pos, r_peaks)

        # return the masket target and cls
        mask = targets != -1
        targets = targets[mask]
        cls = cls[mask]
        preds = preds[mask]
        
        return loss_cls, loss_r_peak_pos, preds, targets, cls, r_peak_pos, r_peaks

    def get_params(self):
        return [
            {'params': self.model.training_params(), 'lr': self.lr_head, 'weight_decay': self.wd},
            {'params': self.model.finetuning_params(), 'lr': self.lr_xlstm, 'weight_decay': self.wd}
        ]
        
    def get_lr(self):
        return self.lr_head
    
    def configure_optimizers(self):
        return common.configure_optimizers(self)