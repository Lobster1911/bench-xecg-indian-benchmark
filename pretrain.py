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



gpu_id = get_least_used_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# torch.multiprocessing.set_sharing_strategy('file_system')

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
            datasets_pretrain.append(mimic.ECGMIMICDataset(
                config, 
                leads_to_use=config.leads, 
                split='train', 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
        elif dataset == 'code15':
            datasets_pretrain.append(code_15.ECGCODE15Dataset(
                config, 
                leads_to_use=config.leads,                
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
        elif dataset == 'ptbxl':
            datasets_pretrain.append(ptb_xl.ECGPTBXLDataset(
                config, 
                leads_to_use=config.leads, 
                split='train', 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
        else:
            raise ValueError(f"Dataset {dataset} not found")

    val_dataset_1 = mimic.ECGMIMICDataset(
        config, 
        leads_to_use=config.leads, 
        split='val', 
        global_augmentations=get_transforms(config, split='train', type='global'), # I want the training augmentation in this case
        local_augmentations=get_transforms(config, split='train', type='local')
    )
    val_dataset_2 = ptb_xl.ECGPTBXLDataset(
        config, 
        leads_to_use=config.leads, 
        split='val', 
        global_augmentations=get_transforms(config, split='train', type='global'), # I want the training augmentation in this case
        local_augmentations=get_transforms(config, split='train', type='local')
    )

    val_dataset = ConcatDataset([val_dataset_1, val_dataset_2])
    train_dataset = ConcatDataset(datasets_pretrain)

    # knn datasets:
    knn_train_dataset = ptb_xl.ECGPTBXLDataset(config, leads_to_use=config.leads, split='train', global_augmentations=None, local_augmentations=None)
    knn_val_dataset = ptb_xl.ECGPTBXLDataset(config, leads_to_use=config.leads, split='val', global_augmentations=None, local_augmentations=None)
    knn_train_dataloader = DataLoader(knn_train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=ptb_xl.make_collate_fn(config.patch_size))
    knn_val_dataloader = DataLoader(knn_val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=ptb_xl.make_collate_fn(config.patch_size))

    
    # keep only 10% of the dataset
    if config.debug: train_dataset = Subset(train_dataset, range(0, len(train_dataset) // 100))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=generic_utils.make_collate_fn(config.patch_size))
    len_train_dataset = len(train_dataset)

    # cat the two dataloaders
    # if config.debug: val_dataset = Subset(val_dataset, range(0, len(val_dataset) // 10))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=generic_utils.make_collate_fn(config.patch_size))

    xlstm = pretrainedxLSTM(config=config, num_channels=len(config.leads))
    # xlstm = torch.compile(xlstm)

    if config.checkpoint != None:
        model = PretrainedxLSTMNetwork.load_from_checkpoint(
            checkpoint_path=config.checkpoint,
            model=xlstm, 
            len_train_dataset=len_train_dataset, 
            config=config, 
            knn_train_dataloader=knn_train_dataloader, 
            knn_val_dataloader=knn_val_dataloader
        )
    else:
        model = PretrainedxLSTMNetwork(
            model=xlstm, 
            len_train_dataset=len_train_dataset,
            config=config, 
            knn_train_dataloader=knn_train_dataloader, 
            knn_val_dataloader=knn_val_dataloader
        )
        
    checkpoint_callback = ModelCheckpoint(monitor=config.monitor_metric)

    early_stopping = EarlyStopping(monitor=config.monitor_metric, patience=config.patience)

    if wandb:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        wand_logger = WandbLogger(project="pretrain-xLSTM", experiment=run, config=config)
        wand_logger.watch(model, log='gradients')
        trainer = L.Trainer(
            max_epochs=config.epochs, 
            logger=wand_logger, 
            callbacks=[checkpoint_callback, early_stopping, lr_monitor], 
            gradient_clip_val=config.grad_clip if not config.use_teacher_student else None,
            accelerator='gpu',
            devices=1,
            strategy='auto'
        )
    else:
        trainer = L.Trainer(
            logger=False,
            max_epochs=config.epochs, 
            callbacks=[checkpoint_callback, early_stopping], 
            gradient_clip_val=config.grad_clip if not config.use_teacher_student else None,
            accelerator='gpu',
            devices=1,
            strategy='auto'
        )

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    #test_dataset = mit_bih.ECGMITBIHDataset(config, split='test')
    #test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=generic_utils.collate_fn, num_workers=config.num_workers)
    #trainer.test(model=model, dataloaders=test_dataloader)

# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/pretrain_config_defaults.yaml')

    pretrain(config, wandb=config.wandb_log)
