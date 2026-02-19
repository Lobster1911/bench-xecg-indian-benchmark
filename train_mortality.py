from torch import utils
import lightning as L
import torch
import argparse

import dataset.mimic_iv as mimic
import dataset.code_dataset as code
import dataset.heedb as heedb
import dataset.generic_utils as generic_utils

import utils.utils as utils
from torch.utils.data import DataLoader, ConcatDataset
from dataset.generic_utils import get_transforms
from trainers.mortality_trainer import TrainerMortality


# os.environ['XLSTM_EXTRA_INCLUDE_PATHS']='/usr/local/include/cuda/:/usr/include/cuda/'

import argparse
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
            datasets.append(mimic.ECGMIMICDataset(config, split='train', global_augmentations=get_transforms(config), downstream_task='mortality'))
        if d.lower() == 'heedb':
            datasets.append(heedb.ECGHEEDBMortalityDataset(config, global_augmentations=get_transforms(config)))

    dataset = ConcatDataset(datasets)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=generic_utils.make_collate_fn_task(config, ['death', 'timey']))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=generic_utils.make_collate_fn_task(config, ['death', 'timey']))

    test_dataset = mimic.ECGMIMICDataset(config, split='test', global_augmentations=get_transforms(config, split='test'), downstream_task='mortality')
    print(f"Test dataset size: {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=generic_utils.make_collate_fn_task(config, ['death', 'timey']))

    base_model = utils.get_base_model(config)

    model = TrainerMortality(model=base_model, config=config, len_train_dataset=len(train_dataset))

    trainer = utils.get_trainer(config, model, f'train-mortality', wandb=wandb, run=run)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')


# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/train_mortality_defaults.yaml')

    train(config, wandb=config.wandb_log)