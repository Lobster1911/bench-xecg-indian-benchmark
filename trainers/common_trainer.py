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
        self.label_smoothing = config.label_smoothing
        self.epochs = config.epochs
        self.use_focal_loss = config.use_focal_loss
        self.linear_probing = config.linear_probing
        self.num_classes = config.num_classes
        self.patch_size = config.patch_size
        self.layerwise_lr_decay = config.layerwise_lr_decay
        self.task = config.task
        self.use_st_mem = config.use_st_mem
        self.use_ecg_jepa = config.use_ecg_jepa

    def get_layers(self):
        if self.use_ecg_jepa:
            # get all the params
            return self.model.encoder.encoder_blocks.blocks
        elif self.use_st_mem:
            return [self.model.__getattr__(f'block{i}') for i in range(self.model.depth)]
        else:
            return self.model.core.model.blocks
        

    def get_params(self):
        if self.linear_probing:
            params = [ {'params': self.model.training_params(), 'lr': self.lr_head, 'weight_decay': self.wd, 'name': 'head'} ]
        elif self.layerwise_lr_decay > 0.:
            params = [ {'params': self.model.training_params(), 'lr': self.lr_head, 'weight_decay': self.wd, 'name': 'head'} ]   
            layers = self.get_layers()
            num_layers = len(layers) + 1 

            # Assign learning rates to each transformer layer
            for i, layer in enumerate(layers):
                layer_lr = self.lr_xlstm * (self.layerwise_lr_decay ** (num_layers - i - 1))  # Earlier layers get smaller LR
                layer_params = layer.parameters()
                params.append({"params": layer_params, "lr": layer_lr, "name": f"layer_{i}"})

            layer_lr = self.lr_xlstm * (self.layerwise_lr_decay ** num_layers)

            if self.use_ecg_jepa:
                # linear projection, need the smallest layer_lr
                params.append({"params": self.model.encoder.W_P.parameters(), "lr": layer_lr, "name": "W_P"})
                # final layer norm, normal lr
                params.append({"params": self.model.encoder.norm.parameters(), "lr": self.lr_xlstm, "name": "ln"})
            elif self.use_st_mem:
                # embeddings, need the smallest layer_lr
                params.append({'params': self.model.to_patch_embedding.parameters(), 'lr': layer_lr, 'name': 'to_patch_embedding'})
                params.append({'params': self.model.pos_embedding, 'lr': layer_lr, 'name': 'pos_embedding'})
                params.append({'params': self.model.sep_embedding, 'lr': layer_lr, 'name': 'sep_embedding'})
                params.append({'params': self.model.lead_embeddings.parameters(), 'lr': layer_lr, 'name': 'lead_embeddings'})
                params.append({'params': self.model.norm.parameters(), 'lr': self.lr_xlstm, 'name': 'ln'})
            else:
                params.append({"params": self.model.patch_embedding.parameters(), "lr": layer_lr, "name": "patch_embedding"})

                if self.model.encoder_type =='large':
                    params.append({'params': self.model.core.model.out_norm.parameters(), 'lr': self.lr_xlstm, 'weight_decay': self.wd, 'name': 'ln2'})
                else:
                    params.append({'params': self.model.core.model.post_blocks_norm.parameters(), 'lr': self.lr_xlstm, 'weight_decay': self.wd, 'name': 'ln2'})

                if self.model.cls_type == 'token' or self.model.cls_type == 'token_2':
                    params.append({'params': self.model.cls_token, 'lr': self.lr_xlstm, 'weight_decay': self.wd, 'name': 'cls'})
                elif self.model.cls_type == 'attn_pool' or self.model.cls_type == 'lin_attn_pool':
                    params.append({'params': self.model.attn_pool.parameters(), 'lr': self.lr_xlstm, 'weight_decay': self.wd, 'name': 'cls'})

                if self.model.num_reg_tokens > 0:
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
