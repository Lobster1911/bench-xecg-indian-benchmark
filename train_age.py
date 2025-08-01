from torch import utils
import lightning as L


import dataset.code_dataset as code
import dataset.ptb_xl as ptbxl


from trainers.regression_trainer import TrainingAge
import torch
import argparse

import utils.utils as utils
from torch.utils.data import DataLoader
from dataset.generic_utils import get_transforms


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

    trainer = utils.get_trainer(config, model, 'train-age', wandb=wandb, run=run)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')


# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/train_age_config_defaults.yaml')

    train(config, wandb=config.wandb_log)