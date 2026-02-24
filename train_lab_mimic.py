from torch import utils
import lightning as L

import dataset.mimic_iv as mimic

import torch
import argparse
import utils.utils as utils
from torch.utils.data import DataLoader
from dataset.generic_utils import get_transforms, make_collate_fn_task
from trainers.mimic_lab_trainer import TrainingMIMIC_LAB
from config import parse_config

import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_lab_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)

    # force the number of classes to be 3 times the number of labels (below, inside, and over the normal range)
    config.num_classes = len(config.label_list) * 3
    
    train_dataset =  mimic.ECGMIMICDataset(config, split='train', global_augmentations=get_transforms(config), downstream_task='lab')
    print(f"Train dataset size: {len(train_dataset)}")
    val_dataset = mimic.ECGMIMICDataset(config, split='val', global_augmentations=get_transforms(config, split='val'), downstream_task='lab')
    print(f"Val dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=make_collate_fn_task(config, 'labels'), drop_last=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=make_collate_fn_task(config, 'labels'), pin_memory=True)

    test_dataset = mimic.ECGMIMICDataset(config, split='test', global_augmentations=get_transforms(config, split='test'), downstream_task='lab')
    print(f"Test dataset size: {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=make_collate_fn_task(config, 'labels'), pin_memory=True)

    base_model = utils.get_base_model(config)

    log_every_n_steps = max(1, len(train_dataset) // (config.batch_size * 10))
    print(f"Logging every {log_every_n_steps} steps")

    model = TrainingMIMIC_LAB(model=base_model, config=config, len_train_dataset=len(train_dataset))

    trainer = utils.get_trainer(config, f'train-lab', wandb=wandb, run=run)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')


# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = parse_config(args.config_file, 'config_defaults/train_blood_test_defaults.yaml')

    train(config, wandb=config.wandb_log)