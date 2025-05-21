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

class CommonTrainerDownstream(L.LightningModule):
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
        self.num_classes = config.num_classes


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
