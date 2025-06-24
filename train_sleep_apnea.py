import os
from torch import utils
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.classification import xLSTMFeatureClassification

import dataset.sleep_apnea as sleep_apnea
import dataset.generic_utils as generic_utils
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from trainers.sleep_apnea_trainer import TrainingSleepApnea
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
import utils.utils as utils
from utils.utils import get_training_class_weights
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import transforms
from dataset.generic_utils import get_transforms

import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_sleep_apnea_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)
    
    train_dataset =  sleep_apnea.ECGSleepApneaDataset(config, split='train', augmentations=get_transforms(config))
    print(f"Train dataset size: {len(train_dataset)}")
    val_dataset = sleep_apnea.ECGSleepApneaDataset(config, split='val', augmentations=get_transforms(config, split='val'))
    print(f"Val dataset size: {len(val_dataset)}")

    if config.use_class_weights:
        weights = get_training_class_weights(train_dataset, label_key='annotation').to('cuda')
        print(f"Using class weights: {weights}")
    else:
        weights = None


    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=sleep_apnea.make_collate_fn(config, split='train'))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False, collate_fn=sleep_apnea.make_collate_fn(config, split='val'))

    test_dataset = sleep_apnea.ECGSleepApneaDataset(config, split='test', augmentations=get_transforms(config, split='test'))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False, collate_fn=sleep_apnea.make_collate_fn(config, split='test'))

    xlstm = xLSTMFeatureClassification(config=config, num_classes=config.num_classes, num_channels=len(config.leads))

    if config.checkpoint is not None and config.checkpoint != '':   
        checkpoint = torch.load(config.checkpoint, weights_only=False)
        new_state_dict = {utils.format_keys(k): v for k, v in checkpoint['state_dict'].items()}
        # remove the fc layer
        new_state_dict = {k: v for k, v in new_state_dict.items() if 'fc' not in k}
        message = xlstm.load_state_dict(new_state_dict, strict=False) 
        print(message) 

    model = TrainingSleepApnea(model=xlstm, config=config, len_train_dataset=len(train_dataset), weights=weights)

    early_stopping = EarlyStopping(monitor='val_f1', patience=config.patience, mode='max')
    nan_stop = EarlyStopping(monitor='val_loss', check_finite=True, patience=config.epochs, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if wandb:
        checkpoint_callback = ModelCheckpoint(monitor='val_f1', mode='max')
        prj = f'train-sleep-apnea'
        wand_logger = WandbLogger(project=prj, experiment=run, config=config)
        wand_logger.watch(model, log='gradients')
        trainer = L.Trainer(max_epochs=config.epochs, logger=wand_logger, callbacks=[early_stopping, lr_monitor, checkpoint_callback, nan_stop], gradient_clip_val=config.grad_clip, log_every_n_steps=1)
    else:
        trainer = L.Trainer(logger=False, max_epochs=config.epochs, callbacks=[early_stopping, nan_stop], gradient_clip_val=config.grad_clip, log_every_n_steps=1)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)


# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/train_sleep_apnea_defaults.yaml')

    train(config, wandb=config.wandb_log)