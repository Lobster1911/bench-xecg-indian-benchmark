import os
from torch import utils
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.classification import xLSTMClassification

import dataset.code_dataset as code
import dataset.ptb_xl as ptbxl
import dataset.generic_utils as generic_utils
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from trainers.regression_trainer import TrainingAge
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
from ecg_jepa.models import load_encoder


# os.environ['XLSTM_EXTRA_INCLUDE_PATHS']='/usr/local/include/cuda/:/usr/include/cuda/'

import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_age_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)
    
    code_dataset = code.ECGCODE15AgeDataset(config, split='train', global_augmentations=get_transforms(config))
    # split the dataset in val and train
    train_dataset, val_dataset = torch.utils.data.random_split(code_dataset, [int(len(code_dataset) * 0.8), len(code_dataset) - int(len(code_dataset) * 0.8)])
    #utils.split_dataset_preserve_labels(code_dataset, split_ratio=0.8, key='age')
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=code.make_collate_fn(config))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=code.make_collate_fn(config))

    test_dataset = ptbxl.ECGPTBXLAgeDataset(config, split='test', global_augmentations=get_transforms(config, split='test'))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=ptbxl.make_collate_fn(config))

    base_model = utils.get_base_model(config)

    log_every_n_steps = max(1, len(train_dataset) // (config.batch_size * 10))
    print(f"Logging every {log_every_n_steps} steps")

    model = TrainingAge(model=base_model, config=config, len_train_dataset=len(train_dataset))

    early_stopping = EarlyStopping(monitor=config.monitor_metric, check_finite=True, patience=config.patience, mode=config.monitor_mode)
    nan_stop = EarlyStopping(monitor='val_loss', check_finite=True, patience=config.epochs, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if wandb:
        checkpoint_callback = ModelCheckpoint(monitor=config.monitor_metric, mode=config.monitor_mode)
        wand_logger = WandbLogger(project='train-age', experiment=run, config=config)
        wand_logger.watch(model, log='gradients')
        trainer = L.Trainer(max_epochs=config.epochs, logger=wand_logger, callbacks=[early_stopping, lr_monitor, checkpoint_callback, nan_stop], gradient_clip_val=config.grad_clip, log_every_n_steps=log_every_n_steps)
    else:
        trainer = L.Trainer(logger=False, max_epochs=config.epochs, callbacks=[early_stopping, nan_stop], gradient_clip_val=config.grad_clip, log_every_n_steps=log_every_n_steps)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)


# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/train_age_config_defaults.yaml')

    train(config, wandb=config.wandb_log)