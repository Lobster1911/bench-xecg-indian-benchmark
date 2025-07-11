import os
from torch import utils
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.mit_bih_models import xLSTMClassificationMIT_BIH
import dataset.mit_bih as mit_bih
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from trainers.mit_bih_trainer import TrainingMIT_BIH
import torch
import argparse
import os
from ecg_jepa.models import load_encoder
import st_mem.encoder as encoder

import utils.utils as utils
from utils.utils import get_training_class_weights
from torch.utils.data import DataLoader
from dataset.generic_utils import get_transforms

# os.environ['XLSTM_EXTRA_INCLUDE_PATHS']='/usr/local/include/cuda/:/usr/include/cuda/'

import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_mit_bih_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)

    if config.split_val_by_patient:
        train_dataset =  mit_bih.ECGMITBIHDataset(config, split='train', augmentations=get_transforms(config))
        print(f"Train dataset size: {len(train_dataset)}")
        val_dataset = mit_bih.ECGMITBIHDataset(config, split='val', augmentations=get_transforms(config, split='val'))
        print(f"Val dataset size: {len(val_dataset)}")
    else:
        dataset =  mit_bih.ECGMITBIHDataset(config, split='train', augmentations=get_transforms(config))
        dataset_len = len(dataset)
        train_len = int(dataset_len * 0.9)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, dataset_len - train_len])
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")


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

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=mit_bih.make_collate_fn(config))
    val_batch_size = 1 if config.split_val_by_patient and not (config.use_transformer) else config.batch_size
    val_num_workers = 0 if config.split_val_by_patient and not (config.use_transformer) else config.num_workers
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=mit_bih.make_collate_fn(config), num_workers=val_num_workers)

    test_dataset = mit_bih.ECGMITBIHDataset(config, split='test', augmentations=get_transforms(config, split='test'))
    test_dataloader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=mit_bih.make_collate_fn(config), num_workers=val_num_workers)

    base_model = utils.get_base_model(config)

    model = TrainingMIT_BIH(model=base_model, config=config, len_train_dataset=len(train_dataset), weights=weights)

    early_stopping = EarlyStopping(monitor=config.monitor_metric, patience=config.patience, mode=config.monitor_mode)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if wandb:
        checkpoint_callback = ModelCheckpoint(monitor=config.monitor_metric, mode=config.monitor_mode)
        wand_logger = WandbLogger(project=f"train-mitbih-{config.num_classes}", experiment=run, config=config)
        wand_logger.watch(model, log='gradients')
        trainer = L.Trainer(max_epochs=config.epochs, logger=wand_logger, callbacks=[early_stopping, lr_monitor, checkpoint_callback], gradient_clip_val=config.grad_clip)
    else:
        trainer = L.Trainer(logger=False, max_epochs=config.epochs, callbacks=[early_stopping], gradient_clip_val=config.grad_clip)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)

# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/train_mit_bih_config_defaults.yaml')

    train(config, wandb=config.wandb_log)