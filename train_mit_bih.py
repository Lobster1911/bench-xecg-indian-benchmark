import os
from torch import utils
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.xLSTM import xLSTMClassificationMIT_BIH
import dataset.mit_bih as mit_bih
import dataset.code_15 as code_15
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from trainers.mit_bih_trainer import TrainingMIT_BIH
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
import utils.utils as utils
from utils.utils import get_training_class_weights
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset

os.environ['XLSTM_EXTRA_INCLUDE_PATHS']='/usr/local/include/cuda/:/usr/include/cuda/'

import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_mit_bih_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)

    if config.split_val_by_patient:
        train_dataset =  mit_bih.ECGMITBIHDataset(config, split='train')
        print(f"Train dataset size: {len(train_dataset)}")
        val_dataset = mit_bih.ECGMITBIHDataset(config, split='val')
        print(f"Val dataset size: {len(val_dataset)}")
    else:
        dataset =  mit_bih.ECGMITBIHDataset(config, split='train')
        dataset_len = len(dataset)
        train_len = int(dataset_len * 0.9)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, dataset_len - train_len])
        print(f"Train dataset size: {len(train_dataset)}")

    if config.use_class_weights:
        # weights = get_training_class_weights(train_dataset).to('cuda')
        # real_weights = [2.2248e-01, 1.0810e+01, 2.6938e+00, 2.4588e+01, 1.2755e+03]
        # without the last class = [0.2781, 13.5098,  3.3668, 30.7307]
        if config.num_classes == 5:
            print('Using class weights for 5 classes')
            weights = torch.tensor([0.2781, 13.5098,  3.3668, 30.7307, 0]).to('cuda')
        elif config.num_classes == 3: 
            print('Using class weights for 3 classes')
            weights = torch.tensor([0.367, 17.866, 4.452]).to('cuda')
    else:
        weights = None

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=mit_bih.collate_fn, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=mit_bih.collate_fn, pin_memory=True)

    test_dataset = mit_bih.ECGMITBIHDataset(config, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=mit_bih.collate_fn, num_workers=config.num_workers, pin_memory=True)

    xlstm = xLSTMClassificationMIT_BIH(config=config, num_classes=config.num_classes, num_channels=len(config.leads))

    def format_keys(key):
        if key.startswith('model.'):
            key = key[6:]

        if key.startswith('xlstm.model.blocks'):
            key.replace('xlstm.model.blocks', 'xlstm.blocks')
        
        return key

    if config.checkpoint is not None and config.checkpoint != '':   
        checkpoint = torch.load(config.checkpoint)
        new_state_dict = {format_keys(k): v for k, v in checkpoint['state_dict'].items()}
        # remove the fc layer
        new_state_dict = {k: v for k, v in new_state_dict.items() if 'fc' not in k}
        message = xlstm.load_state_dict(new_state_dict, strict=False) 
        print(message) 

    model = TrainingMIT_BIH(model=xlstm, config=config, len_train_dataset=len(train_dataset), weights=weights)

    early_stopping = EarlyStopping(monitor='val_f1', patience=config.patience, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if wandb:
        checkpoint_callback = ModelCheckpoint(monitor='val_f1', mode='max')
        wand_logger = WandbLogger(project=f"train-mitbih-{config.num_classes}", experiment=run, config=config)
        wand_logger.watch(model, log='gradients')
        trainer = L.Trainer(max_epochs=config.epochs, logger=wand_logger, callbacks=[early_stopping, lr_monitor, checkpoint_callback], gradient_clip_val=config.grad_clip)
    else:
        trainer = L.Trainer(max_epochs=config.epochs, callbacks=[early_stopping, lr_monitor], gradient_clip_val=config.grad_clip)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)

# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/train_mit_bih_config_defaults.yaml')

    train(config, wandb=config.wandb_log)