from torch import utils
import lightning as pl
import torch
import argparse
from torch.utils.data import DataLoader

import utils.utils as utils
from dataset.generic_utils import get_transforms
import dataset.mit_bih as mit_bih
from trainers.mit_bih_trainer import TrainingMIT_BIH
from trainers.r_peaks_trainer import TrainingRPeak


import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_mit_bih_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: pl.seed_everything(42)

    # recurrent models can handle longer sequences, so we use full signal for evaluation and testing
    config.is_recurrent = not (config.use_ecg_jepa or config.use_st_mem or config.use_ecg_founder or config.encoder_type == 'transformer')


    dataset_class = mit_bih.ECGMITBIHDatasetSingleHB if config.use_ecg_founder and not config.r_peaks_detection else mit_bih.ECGMITBIHDataset
    print(f"Using dataset class: {dataset_class.__name__}")

    if config.split_val_by_patient:
        # splits the validation set by patient
        train_dataset =  dataset_class(config, split='train', augmentations=get_transforms(config))
        print(f"Train dataset size (split by patient): {len(train_dataset)}")
        val_dataset = dataset_class(config, split='val', augmentations=get_transforms(config, split='val'))
        print(f"Val dataset size (split by patient): {len(val_dataset)}")
    else:
        # split the validation set from the training set, here patients are mixed
        dataset = dataset_class(config, split='train', augmentations=get_transforms(config))
        dataset_len = len(dataset)
        train_len = int(dataset_len * 0.9)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, dataset_len - train_len])
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

    if config.use_class_weights:
        if config.r_peaks_detection:
            print('Using class weights for r-peaks detection')
            weights = torch.tensor([1/config.patch_size, (config.patch_size-1)/config.patch_size]).to('cuda')
        if config.num_classes == 5:
            print('Using class weights for 5 classes')
            weights = torch.tensor([0.2781, 13.5098,  3.3668, 30.7307, 0]).to('cuda')
        elif config.num_classes == 3: 
            print('Using class weights for 3 classes')
            weights = torch.tensor([0.367, 17.866, 4.452]).to('cuda')
    else:
        weights = None

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=mit_bih.make_collate_fn(config))
    val_batch_size = 1 if config.is_recurrent and not config.r_peaks_detection else config.batch_size
    val_num_workers = 0 if config.is_recurrent and not config.r_peaks_detection else config.num_workers
    print("Using val_batch_size:", val_batch_size, "and val_num_workers:", val_num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=mit_bih.make_collate_fn(config), num_workers=val_num_workers)

    test_dataset = dataset_class(config, split='test', augmentations=get_transforms(config, split='test'))
    test_dataloader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=mit_bih.make_collate_fn(config), num_workers=val_num_workers)

    base_model = utils.get_base_model(config, feature_classification=True)

    if config.r_peaks_detection:
       model = TrainingRPeak(model=base_model, config=config, len_train_dataset=len(train_dataset), weights=weights)
    else:
        model = TrainingMIT_BIH(model=base_model, config=config, len_train_dataset=len(train_dataset), weights=weights)

    prj_string = f"train-mitbih-{config.num_classes}" if not config.r_peaks_detection else f"train-mitbih-r_peaks"
    trainer = utils.get_trainer(config, model, prj_string, wandb=wandb, run=run)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')

# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/train_mit_bih_config_defaults.yaml')

    train(config, wandb=config.wandb_log)