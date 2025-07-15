import os
from torch import utils
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from models.mit_bih_models import xLSTMClassificationMIT_BIH
import dataset.mit_bih as mit_bih
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from trainers.mit_bih_trainer import TrainingMIT_BIH_R_Peak
import torch
import argparse
import os
from ecg_jepa.models import load_encoder
import st_mem.encoder as encoder
import dataset.intense_exercise as intense_exercise

import utils.utils as utils
from utils.utils import get_training_class_weights
from torch.utils.data import DataLoader
from dataset.generic_utils import get_transforms


import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_high_intensity_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: pl.seed_everything(42)

    train_dataset =  intense_exercise.ECGHighIntensity(config, split='train', global_augmentations=get_transforms(config))
    print(f"Train dataset size: {len(train_dataset)}")
    val_dataset = intense_exercise.ECGHighIntensity(config, split='val', global_augmentations=get_transforms(config, split='val'))
    print(f"Val dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    test_dataset = intense_exercise.ECGHighIntensity(config, split='test', global_augmentations=get_transforms(config, split='test'))
    print(f"Test dataset size: {len(test_dataset)}")

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    base_model = utils.get_base_model(config, feature_classification=True)

    model = TrainingMIT_BIH_R_Peak(model=base_model, config=config, len_train_dataset=len(train_dataset))

    trainer = utils.get_trainer(config, model, "train-exercise-r_peak", wandb=wandb, run=run)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)

# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/train_high_intensity_config_defaults.yaml')

    train(config, wandb=config.wandb_log)