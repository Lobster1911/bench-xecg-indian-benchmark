from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torchmetrics.classification.precision_recall
import torchmetrics.classification.specificity
import numpy as np
import torch
from schedulers import get_cosine_schedule_with_warmup
import trainers.common as common
from optimizers.lamb import Lamb

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
        self.patch_size = config.patch_size
        self.layerwise_lr_decay = config.layerwise_lr_decay
        self.task = config.task
        

    def get_params(self):
        if self.linear_probing:
            params = [ {'params': self.model.training_params(), 'lr': self.lr_head, 'weight_decay': self.wd, 'name': 'head'} ]
        elif self.layerwise_lr_decay > 0.:
            params = [ {'params': self.model.training_params(), 'lr': self.lr_head, 'weight_decay': self.wd, 'name': 'head'} ]    
            num_layers = len(self.model.xlstm.model.blocks) + 1

            # Assign learning rates to each transformer layer
            for i, layer in enumerate(self.model.xlstm.model.blocks):
                layer_lr = self.lr_xlstm * (self.layerwise_lr_decay ** (num_layers - i - 1))  # Earlier layers get smaller LR
                layer_params = layer.parameters()
                params.append({"params": layer_params, "lr": layer_lr, "name": f"layer_{i}"})

            layer_lr = self.lr_xlstm * (self.layerwise_lr_decay ** num_layers)
            params.append({"params": self.model.patch_embedding.parameters(), "lr": layer_lr, "name": "patch_embedding"})

            if self.model.xlstm_type =='large':
                params.append({'params': self.model.xlstm.model.out_norm.parameters(), 'lr': self.lr_xlstm, 'weight_decay': self.wd, 'name': 'ln2'})
            else:
                params.append({'params': self.model.xlstm.model.post_blocks_norm.parameters(), 'lr': self.lr_xlstm, 'weight_decay': self.wd, 'name': 'ln2'})

            if self.model.cls_type == 'token':
                params.append({'params': self.model.cls_token.parameters(), 'lr': self.lr_xlstm, 'weight_decay': self.wd, 'name': 'cls'})
            elif self.model.cls_type == 'attn_pool':
                params.append({'params': self.model.attn_pool.parameters(), 'lr': self.lr_xlstm, 'weight_decay': self.wd, 'name': 'cls'})

            if self.mode.num_reg_token > 0:
                params.append({'params': self.reg_token, 'lr': self.lr_xlstm, 'weight_decay': self.wd, 'name': 'reg_tokens'})

        else:
            params = [
                {'params': self.model.training_params(), 'lr': self.lr_head, 'weight_decay': self.wd},
                {'params': self.model.finetuning_params(), 'lr': self.lr_xlstm, 'weight_decay': self.wd}
            ]
        return params
    
    def get_lr(self):
        return self.lr_head
        
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(params=self.get_params(), lr=self.get_lr(), weight_decay=self.wd)
        elif self.optimizer == 'adamw':
            optimizer = optim.AdamW(params=self.get_params(), lr=self.get_lr(), weight_decay=self.wd)
        elif self.optimizer == 'adafactor':
            optimizer = optim.Adafactor(params=self.get_params(), lr=self.get_lr(), weight_decay=self.wd)
        elif self.optimizer == 'lamb':
            optimizer = Lamb(params=self.get_params(), lr=self.get_lr(), weight_decay=self.wd)
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
