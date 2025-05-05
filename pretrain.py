import os
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.xLSTM import pretrainedxLSTM
import dataset.mit_bih as mit_bih
import dataset.code_15 as code_15
import dataset.mimic_iv as mimic
import dataset.ptb_xl as ptb_xl
import dataset.generic_utils as generic_utils
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from trainers.ssl_pretrainer import PretrainedxLSTMNetwork
import utils.utils as utils
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from dataset.generic_utils import get_transforms
from utils.utils import get_least_used_gpu

# argparse
import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/pretrain_run_config.yaml', help='Path to the config file')

def pretrain(config, run=None, wandb=False):
    max_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", config.num_workers))
    config.num_workers = min(config.num_workers, max_cpus)

    # set deterministic training
    if config.deterministic: L.seed_everything(42)

    datasets_pretrain = []

    for dataset in config.pretrain_datasets:
        if dataset == 'mimic':
            datasets_pretrain.append(mimic.ECGMIMICDataset(config, leads_to_use=config.leads, split='train', augmentations=get_transforms(config)))
        elif dataset == 'code15':
            datasets_pretrain.append(code_15.ECGCODE15Dataset(config, leads_to_use=config.leads, augmentations=get_transforms(config)))
        elif dataset == 'mit':
            datasets_pretrain.append(mit_bih.ECGMITBIHDataset(config, split='train', augmentations=get_transforms(config)))       
        elif dataset == 'ptbxl':
            datasets_pretrain.append(ptb_xl.ECGPTBXLDataset(config, leads_to_use=config.leads, split='train', augmentations=get_transforms(config)))
        else:
            raise ValueError(f"Dataset {dataset} not found")

    val_dataset_1 = mimic.ECGMIMICDataset(config, leads_to_use=config.leads, split='val', augmentations=get_transforms(config))
    val_dataset_2 = ptb_xl.ECGPTBXLDataset(config, leads_to_use=config.leads, split='val', augmentations=get_transforms(config))

    val_dataset = ConcatDataset([val_dataset_1, val_dataset_2])

    train_dataset = ConcatDataset(datasets_pretrain)
    # keep only 10% of the dataset
    if config.debug: train_dataset = Subset(train_dataset, range(0, len(train_dataset) // 100))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=generic_utils.collate_fn)
    len_train_dataset = len(train_dataset)

    # cat the two dataloaders
    # if config.debug: val_dataset = Subset(val_dataset, range(0, len(val_dataset) // 10))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=generic_utils.collate_fn)

    xlstm = pretrainedxLSTM(config=config, num_channels=len(config.leads))
    # xlstm = torch.compile(xlstm)

    if config.checkpoint != None:
        model = PretrainedxLSTMNetwork.load_from_checkpoint(checkpoint_path=config.checkpoint, model=xlstm, len_train_dataset=len_train_dataset, config=config)
    else:
        model = PretrainedxLSTMNetwork(model=xlstm, len_train_dataset=len_train_dataset, config=config)
        
    checkpoint_callback = ModelCheckpoint(monitor='val_nrmse')

    early_stopping = EarlyStopping(monitor='val_nrmse', patience=config.patience)

    if wandb:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        wand_logger = WandbLogger(project="pretrain-xLSTM", experiment=run, config=config)
        wand_logger.watch(model, log='gradients')
        trainer = L.Trainer(
            max_epochs=config.epochs, 
            logger=wand_logger, 
            callbacks=[checkpoint_callback, early_stopping, lr_monitor], 
            gradient_clip_val=config.grad_clip,
            accelerator='gpu',
            devices=[get_least_used_gpu()],
            strategy='auto'
        )
    else:
        trainer = L.Trainer(
            logger=False,
            max_epochs=config.epochs, 
            callbacks=[checkpoint_callback, early_stopping], 
            gradient_clip_val=config.grad_clip,
            accelerator='gpu',
            devices=get_least_used_gpu(),
            strategy='auto'
        )

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    #test_dataset = mit_bih.ECGMITBIHDataset(config, split='test')
    #test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=generic_utils.collate_fn, num_workers=config.num_workers)
    # trainer.test(model=model, dataloaders=test_dataloader)

# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/pretrain_config_defaults.yaml')

    pretrain(config, wandb=config.wandb_log)
