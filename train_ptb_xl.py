import os
from torch import utils
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.classification import xLSTMClassification

import dataset.ptb_xl as ptbxl
import dataset.generic_utils as generic_utils
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from trainers.ptb_xl_trainer import TrainingPTB_XL
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
import utils.utils as utils
from utils.utils import get_training_class_weights_multilabel
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import transforms
from dataset.generic_utils import get_transforms
import st_mem.encoder as encoder
from ecg_jepa.models import load_encoder


# os.environ['XLSTM_EXTRA_INCLUDE_PATHS']='/usr/local/include/cuda/:/usr/include/cuda/'

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

    if config.use_class_weights:
        if config.num_classes == 5:
            print('Using class weights for 5 classes')
            # weights = get_training_class_weights_multilabel(train_dataset, label_key='class_label').to('cuda')
            weights = torch.tensor([0.8323, 0.4587, 0.7954, 1.6445, 0.8915]).to('cuda')
        else:
            weights = None # TODO
    else:
        weights = None

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=ptbxl.make_collate_fn(config, downstream=True, split='train'), drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=ptbxl.make_collate_fn(config, downstream=True, split='val'))

    test_dataset = ptbxl.ECGPTBXLDataset(config, split='test', global_augmentations=get_transforms(config, split='test'))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=ptbxl.make_collate_fn(config, downstream=True, split='test'), num_workers=config.num_workers)

    if config.use_st_mem:
        base_model = encoder.__dict__['st_mem_vit_base'](seq_len=2250, patch_size=75, num_leads=12, num_classes=config.num_classes, linear_probing=config.linear_probing, drop_path_rate=config.drop_path_prob)
        checkpoint = torch.load('pretrained_models/st_mem_vit_base_encoder.pth', weights_only=False)
        checkpoint_model = checkpoint['model']
        state_dict = base_model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Remove key {k} from pre-trained checkpoint")
                del checkpoint_model[k]
        msg = base_model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    elif config.use_ecg_jepa:
        ckpt_dir = 'pretrained_models/multiblock_epoch100.pth'
        base_model = load_encoder(ckpt_dir=ckpt_dir, drop_path_rate=config.drop_path_prob, num_classes=config.num_classes) # dim is the dimension of the latent space
    elif config.use_ecg_founder:
        from ecg_founder.finetune_model import ft_12lead_ECGFounder
        path = './checkpoint/12_lead_ECGFounder.pth'
        base_model = ft_12lead_ECGFounder('cuda', path, config.num_classes, linear_prob=config.linear_probing)
    else:
        base_model = xLSTMClassification(config=config, num_classes=config.num_classes, num_channels=len(config.leads))
        if config.checkpoint is not None and config.checkpoint != '':   
            checkpoint = torch.load(config.checkpoint, weights_only=False)
            new_state_dict = {utils.format_keys(k): v for k, v in checkpoint['state_dict'].items()}
            # remove the fc layer
            new_state_dict = {k: v for k, v in new_state_dict.items() if 'fc' not in k}
            message = base_model.load_state_dict(new_state_dict, strict=False) 
            print(message) 

    log_every_n_steps = max(1, len(train_dataset) // (config.batch_size * 10))
    print(f"Logging every {log_every_n_steps} steps")

    model = TrainingPTB_XL(model=base_model, config=config, len_train_dataset=len(train_dataset), weights=weights)

    early_stopping = EarlyStopping(monitor=config.monitor_metric, check_finite=True, patience=config.patience, mode='max')
    nan_stop = EarlyStopping(monitor='val_loss', check_finite=True, patience=config.epochs, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if wandb:
        checkpoint_callback = ModelCheckpoint(monitor=config.monitor_metric, mode='max')
        prj = f'train-ptbxl-{config.classification_taksk}-{config.task}'
        wand_logger = WandbLogger(project=prj, experiment=run, config=config)
        wand_logger.watch(model, log='gradients')
        trainer = L.Trainer(max_epochs=config.epochs, logger=wand_logger, callbacks=[early_stopping, lr_monitor, checkpoint_callback, nan_stop], gradient_clip_val=config.grad_clip, log_every_n_steps=log_every_n_steps)
    else:
        trainer = L.Trainer(logger=False, max_epochs=config.epochs, callbacks=[early_stopping, nan_stop], gradient_clip_val=config.grad_clip, log_every_n_steps=log_every_n_steps)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)


# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/train_ptb_xl_config_defaults.yaml')

    train(config, wandb=config.wandb_log)