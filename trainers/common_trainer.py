from torch import optim, nn
import lightning as pl
import torchmetrics
import numpy as np
import torch
from schedulers import get_cosine_schedule_with_warmup
from utils.loss_utils import focal_loss


class CommonTrainerDownstream(pl.LightningModule):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__()
        self.lr_head = config.lr_head
        self.lr_core = config.lr_core
        self.wd = config.wd
        self.model = model
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.weights = weights
        self.use_scheduler = config.use_scheduler
        self.len_train_dataset = len_train_dataset
        self.num_epochs_warmup = config.num_epochs_warmup
        self.sched_decay_factor = config.sched_decay_factor
        self.epochs = config.epochs
        self.use_focal_loss = config.use_focal_loss
        self.linear_probing = config.linear_probing
        self.num_classes = config.num_classes
        self.patch_size = config.patch_size
        self.layerwise_lr_decay = config.layerwise_lr_decay
        self.task = config.task
        self.use_st_mem = config.use_st_mem
        self.use_ecg_jepa = config.use_ecg_jepa
        self.discriminative_lr_factor = config.discriminative_lr_factor
        
    def get_params(self):
        if self.linear_probing:
            params = [ {'params': self.model.training_params(), 'lr': self.lr_head, 'weight_decay': self.wd, 'name': 'head'} ]
        elif self.layerwise_lr_decay > 0. and self.layerwise_lr_decay < 1.:
            params = [ {'params': self.model.training_params(), 'lr': self.lr_head, 'weight_decay': self.wd, 'name': 'head'} ]   
            params.extend(self.model.get_params_layerwise_decay(self.layerwise_lr_decay, self.lr_core, self.wd))
        elif self.discriminative_lr_factor is not None:
            params = self.model.get_params_layerwise_decay(self.discriminative_lr_factor, self.lr_core, self.wd)
        else:
            params = [
                {'params': self.model.training_params(), 'lr': self.lr_head, 'weight_decay': self.wd},
                {'params': self.model.finetuning_params(), 'lr': self.lr_core, 'weight_decay': self.wd}
            ]
        return params
    
    def get_lr(self):
        return self.lr_head

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)
        
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(params=self.get_params(), lr=self.get_lr(), weight_decay=self.wd)
        elif self.optimizer == 'adamw':
            optimizer = optim.AdamW(params=self.get_params(), lr=self.get_lr(), weight_decay=self.wd)
        elif self.optimizer == 'adafactor':
            optimizer = optim.Adafactor(params=self.get_params(), lr=self.get_lr(), weight_decay=self.wd)
        elif self.optimizer == 'momentum':
            optimizer = optim.SGD(self.get_params(), lr=self.get_lr(), momentum=0.9, weight_decay=self.wd)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.get_params(), lr=self.get_lr(), momentum=0., weight_decay=self.wd)

        if self.use_scheduler: 
            steps_per_epoch = np.ceil(self.len_train_dataset / self.batch_size)
            num_training_steps = steps_per_epoch * self.epochs
            warmup_steps = steps_per_epoch * self.num_epochs_warmup

            sched = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps = warmup_steps, 
                num_training_steps = num_training_steps, 
            )

            scheduler = {
                'scheduler': sched,
                'interval': 'step', # or 'epoch' 
                'frequency': 1,
            }
            return [optimizer], [scheduler]
        else:
            return [optimizer]