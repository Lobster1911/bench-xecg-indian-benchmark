import argparse

from torch import utils
import lightning as L
import torch
from torch.utils.data import DataLoader

import bench_xecg.dataset.cpsc2018 as cpsc2018
from bench_xecg.trainers.cpsc_2018_trainer import TrainingCPSC_2018
import bench_xecg.utils.utils as utils
from bench_xecg.utils.utils import get_training_class_weights_multilabel
from bench_xecg.dataset.generic_utils import get_transforms
from bench_xecg.config import parse_config

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_cpsc2018_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)
    
    train_dataset =  cpsc2018.ECGCPSC2018Dataset(config, split='train', global_augmentations=get_transforms(config))
    print(f"Train dataset size: {len(train_dataset)}")
    val_dataset = cpsc2018.ECGCPSC2018Dataset(config, split='val', global_augmentations=get_transforms(config, split='val'))
    print(f"Val dataset size: {len(val_dataset)}")

    if config.training_pct < 1.0:
        train_dataset = utils.split_dataset_preserve_labels(train_dataset, split_ratio=config.training_pct, key='labels')

    if config.use_class_weights:
        weights = get_training_class_weights_multilabel(train_dataset, label_key='labels').to('cuda')
        print(f'Class weights: {weights}')
    else:
        weights = None

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=cpsc2018.make_collate_fn(config, split='train'))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=cpsc2018.make_collate_fn(config, split='val'))

    test_dataset = cpsc2018.ECGCPSC2018Dataset(config, split='test', global_augmentations=get_transforms(config, split='test'))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,  num_workers=config.num_workers, collate_fn=cpsc2018.make_collate_fn(config, split='test'))
    
    base_model = utils.get_base_model(config)
            
    model = TrainingCPSC_2018(model=base_model, config=config, len_train_dataset=len(train_dataset), weights=weights)
    
    prj_string = f'train-cpsc2018-{config.task}'
    trainer = utils.get_trainer(config, prj_string, wandb=wandb, run=run)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')


# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = parse_config(args.config_file, 'config_defaults/train_cpsc2018_defaults.yaml')

    train(config, wandb=config.wandb_log)