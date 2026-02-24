import os
from torch import utils
import lightning as L
import torch
import numpy as np
import argparse
import os

from dataset.deepbeat import DeepBeatDataset
from trainers.deepbeat_trainer import TrainingDeepBeat

import utils.utils as utils
from torch.utils.data import DataLoader, Subset
from dataset.generic_utils import get_transforms
from torch.utils.data import WeightedRandomSampler
from config import parse_config


import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_ppg_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)
    
    train_dataset =  DeepBeatDataset(config, split='train', global_augmentations=get_transforms(config))
    print(f"Train dataset size: {len(train_dataset)}")
    val_dataset = DeepBeatDataset(config, split='val', global_augmentations=get_transforms(config, split='val'))
    print(f"Val dataset size: {len(val_dataset)}")

    if config.data_pct < 1.0:
        # At every epoch, we will sample a different subset of the data, so we use less data and training is faster.
        N = len(train_dataset)
        num_samples = int(N * config.data_pct)  # 10% of the dataset
        weights = torch.ones(N) / N  # Initialize weights to 1.0
        sampler = WeightedRandomSampler(weights, num_samples, replacement=True)

        # take the 10% of the dataset for validation
        rng = np.random.default_rng(42)
        indices = rng.choice(len(val_dataset), size=int(len(val_dataset) * config.data_pct), replace=False)
        val_subset = Subset(val_dataset, indices)

        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, sampler=sampler)
        val_dataloader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    test_dataset = DeepBeatDataset(config, split='test', global_augmentations=get_transforms(config, split='test'))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    base_model = utils.get_base_model(config)
    
    log_every_n_steps = max(1, len(train_dataset) // (config.batch_size * 10))
    print(f"Logging every {log_every_n_steps} steps")

    model = TrainingDeepBeat(model=base_model, config=config, len_train_dataset=len(train_dataset))

    trainer = utils.get_trainer(config, 'train-deepbeat', wandb=wandb, run=run)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')

# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = parse_config(args.config_file, 'config_defaults/train_deepbeat_defaults.yaml')
    print(f"Using config: {args.config_file}")

    train(config, wandb=config.wandb_log)