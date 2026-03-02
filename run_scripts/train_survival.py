import argparse

from torch import utils
import lightning as L
import torch
from torch.utils.data import DataLoader, ConcatDataset


import bench_xecg.dataset.mimic_iv as mimic
import bench_xecg.dataset.code_dataset as code
import bench_xecg.dataset.heedb as heedb
import bench_xecg.dataset.generic_utils as generic_utils

import bench_xecg.utils.utils as utils
from bench_xecg.dataset.generic_utils import get_transforms
from bench_xecg.trainers.survival_trainer import TrainerSurvival
from bench_xecg.config import parse_config

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_mortality_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)

    datasets = []
    for d in config.training_dataset:
        if d.lower() == 'code15':
            datasets.append(code.ECGCODE15MortalityDataset(config, split='train', global_augmentations=get_transforms(config)))
        if d.lower() == 'mimic':
            datasets.append(mimic.ECGMIMICDataset(config, split='train', global_augmentations=get_transforms(config), downstream_task='survival'))
        if d.lower() == 'heedb':
            datasets.append(heedb.ECGHEEDBMortalityDataset(config, global_augmentations=get_transforms(config)))

    dataset = ConcatDataset(datasets)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=generic_utils.make_collate_fn_task(config, ['death', 'timey']))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=generic_utils.make_collate_fn_task(config, ['death', 'timey']))

    test_dataset = mimic.ECGMIMICDataset(config, split='test', global_augmentations=get_transforms(config, split='test'), downstream_task='survival')
    print(f"Test dataset size: {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=generic_utils.make_collate_fn_task(config, ['death', 'timey']))

    base_model = utils.get_base_model(config)

    model = TrainerSurvival(model=base_model, config=config, len_train_dataset=len(train_dataset))

    trainer = utils.get_trainer(config, f'train-survival', wandb=wandb, run=run)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')


# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = parse_config(args.config_file, 'config_defaults/train_survival_defaults.yaml')

    train(config, wandb=config.wandb_log)