import lightning as L
import torch
import argparse

from torch.utils.data import DataLoader
from dataset.generic_utils import get_transforms
from torch import utils
from config import parse_config


import dataset.ptb_xl as ptbxl
from trainers.ptb_xl_trainer import TrainingPTB_XL
from utils.utils import get_training_class_weights_multilabel
import utils.utils as utils


import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_ptb_xl_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)
    
    train_dataset =  ptbxl.ECGPTBXLDataset(config, split='train', global_augmentations=get_transforms(config))
    print(f"Train dataset size: {len(train_dataset)}")
    val_dataset = ptbxl.ECGPTBXLDataset(config, split='val', global_augmentations=get_transforms(config, split='val'))
    print(f"Val dataset size: {len(val_dataset)}")

    if config.training_pct < 1.0:
        train_dataset = utils.split_dataset_preserve_labels(train_dataset, split_ratio=config.training_pct)

    if config.use_class_weights and config.num_classes == 5:
        # weights = get_training_class_weights_multilabel(train_dataset, label_key='class_label').to('cuda')
        # hardcode for faster initialization
        weights = torch.tensor([0.8323, 0.4587, 0.7954, 1.6445, 0.8915]).to('cuda')
    else:
        weights = None


    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=ptbxl.make_collate_fn(config, downstream=True, split='train'), pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=ptbxl.make_collate_fn(config, downstream=True, split='val'), pin_memory=True, drop_last=False)

    test_dataset = ptbxl.ECGPTBXLDataset(config, split='test', global_augmentations=get_transforms(config, split='test'))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=ptbxl.make_collate_fn(config, downstream=True, split='test'), num_workers=config.num_workers, pin_memory=True, drop_last=False)

    base_model = utils.get_base_model(config)
    
    log_every_n_steps = max(1, len(train_dataset) // (config.batch_size * 10))
    print(f"Logging every {log_every_n_steps} steps")

    model = TrainingPTB_XL(model=base_model, config=config, len_train_dataset=len(train_dataset), weights=weights)

    prj_str = f'train-ptbxl-{config.classification_task}-{config.task}'
    trainer = utils.get_trainer(config, model, prj_string=prj_str, wandb=wandb, run=run)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')

# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = parse_config(args.config_file, 'config_defaults/train_ptb_xl_defaults.yaml')

    train(config, wandb=config.wandb_log)