import os
from torch import utils
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.classification import xLSTMClassification

import dataset.mimic_iv as mimic
import dataset.generic_utils as generic_utils
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

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
from trainers.regression_trainer import RegressionTrainer


# os.environ['XLSTM_EXTRA_INCLUDE_PATHS']='/usr/local/include/cuda/:/usr/include/cuda/'

import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_lvef_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)
    
    train_dataset =  mimic.ECGMIMICDataset(config, split='train', global_augmentations=get_transforms(config), downstream_task='lvef')
    print(f"Train dataset size: {len(train_dataset)}")
    val_dataset = mimic.ECGMIMICDataset(config, split='val', global_augmentations=get_transforms(config, split='val'), downstream_task='lvef')
    print(f"Val dataset size: {len(val_dataset)}")

    if config.use_class_weights:
        weights = get_training_class_weights_multilabel(train_dataset, label_key='class_label').to('cuda')
    else:
        weights = None

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    test_dataset = mimic.ECGMIMICDataset(config, split='test', global_augmentations=get_transforms(config, split='test'), downstream_task='lvef')
    print(f"Test dataset size: {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    base_model = utils.get_base_model(config)

    log_every_n_steps = max(1, len(train_dataset) // (config.batch_size * 10))
    print(f"Logging every {log_every_n_steps} steps")

    model = RegressionTrainer(model=base_model, config=config, len_train_dataset=len(train_dataset), weights=weights, target_key='lvef')

    trainer = utils.get_trainer(config, model, 'train-lvef', wandb=wandb, run=run)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)


# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/train_lvef_config_defaults.yaml')

    train(config, wandb=config.wandb_log)