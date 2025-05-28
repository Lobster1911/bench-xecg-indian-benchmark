import os
from torch import utils
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.classification import xLSTMClassification

import dataset.cpsc2018 as cpsc2018

import dataset.generic_utils as generic_utils
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from trainers.cpsc_2018_trainer import TrainingCPSC_2018
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
import utils.utils as utils
from utils.utils import get_training_class_weights_multilabel
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import transforms
from dataset.generic_utils import get_transforms
import st_mem.encoder as encoder


# os.environ['XLSTM_EXTRA_INCLUDE_PATHS']='/usr/local/include/cuda/:/usr/include/cuda/'

import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_cpsc2018_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)
    
    train_dataset =  cpsc2018.ECGCPSC2018Dataset(config, split='train', global_augmentations=get_transforms(config))
    print(f"Train dataset size: {len(train_dataset)}")
    val_dataset = cpsc2018.ECGCPSC2018Dataset(config, split='val', global_augmentations=get_transforms(config, split='val'))
    print(f"Val dataset size: {len(val_dataset)}")

    if config.use_class_weights:
        if config.num_classes == 9:
            # weights = get_training_class_weights_multilabel(train_dataset, label_key='labels').to('cuda')
            weights = torch.tensor([1.2444, 1.1167, 1.0653, 0.8114, 3.1714, 0.6416, 3.4688, 0.4102, 0.8866]).to('cuda')
            print(f'Class weights: {weights}')
        else:
            weights = None # TODO
    else:
        weights = None

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=cpsc2018.make_collate_fn(config, split='train'))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=cpsc2018.make_collate_fn(config, split='val'))

    test_dataset = cpsc2018.ECGCPSC2018Dataset(config, split='test', global_augmentations=get_transforms(config, split='test'))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=cpsc2018.make_collate_fn(config, split='test'), num_workers=config.num_workers)
    
    if config.use_st_mem:
        base_model = encoder.__dict__['st_mem_vit_base'](seq_len=2250, patch_size=75, num_leads=12, num_classes=config.num_classes, linear_probing=config.linear_probing)
        checkpoint = torch.load('pretrained_models/st_mem_vit_base_encoder.pth', weights_only=False)
        checkpoint_model = checkpoint['model']
        state_dict = base_model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Remove key {k} from pre-trained checkpoint")
                del checkpoint_model[k]
        msg = base_model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    else:
        base_model = xLSTMClassification(config=config, num_classes=config.num_classes, num_channels=len(config.leads))
        if config.checkpoint is not None and config.checkpoint != '':   
            checkpoint = torch.load(config.checkpoint, weights_only=False)
            new_state_dict = {utils.format_keys(k): v for k, v in checkpoint['state_dict'].items()}
            # remove the fc layer
            new_state_dict = {k: v for k, v in new_state_dict.items() if 'fc' not in k}
            message = base_model.load_state_dict(new_state_dict, strict=False) 
            print(message) 
            
    model = TrainingCPSC_2018(model=base_model, config=config, len_train_dataset=len(train_dataset), weights=weights)

    early_stopping = EarlyStopping(monitor='val_auroc', patience=config.patience, mode='max')
    nan_stop = EarlyStopping(monitor='val_loss', check_finite=True, patience=config.epochs, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if wandb:
        checkpoint_callback = ModelCheckpoint(monitor='val_auroc', mode='max')
        prj = f'train-cpsc2018-{config.task}'
        wand_logger = WandbLogger(project=prj, experiment=run, config=config)
        wand_logger.watch(model, log='gradients')
        trainer = L.Trainer(max_epochs=config.epochs, logger=wand_logger, callbacks=[early_stopping, lr_monitor, checkpoint_callback, nan_stop], gradient_clip_val=config.grad_clip, log_every_n_steps=20)
    else:
        trainer = L.Trainer(logger=False, max_epochs=config.epochs, callbacks=[early_stopping, nan_stop], gradient_clip_val=config.grad_clip, log_every_n_steps=20)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)


# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/train_cpsc2018_defaults.yaml')

    train(config, wandb=config.wandb_log)