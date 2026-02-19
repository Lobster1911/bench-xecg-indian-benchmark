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
from utils.utils import get_training_class_weights


import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_mit_bih_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: pl.seed_everything(42)
    
    # fix max_len signal
    config.max_length_signal = config.win_len * 2 + config.context_len * 2
    
    dataset_class = mit_bih.ECGMITBIHDatasetSingleHB if config.single_hb and not config.r_peaks_detection else mit_bih.ECGMITBIHDataset
    print(f"Using dataset class: {dataset_class.__name__}")

    if config.split_val_by_patient:
        # splits the validation set by patient
        train_dataset = dataset_class(config, split='train', augmentations=get_transforms(config))
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
            # when the patch size is too small we need to set a weight mimicing 1 hb per second
            param = config.sampling_freq if config.patch_size < 5 else config.patch_size
            weights = torch.tensor([1/param, (param-1)/param]).to('cuda')
            print(f'Using class weights for r-peaks detection: {weights}')
        elif config.num_classes == 5:
            if config.win_len == 1600:
                weights = torch.tensor([2.2425e-01, 8.3814e+00, 2.7265e+00, 1.8476e+01, 2.4445e+03]).to('cuda')
            elif config.win_len == 500:
                weights = torch.tensor([2.2468e-01, 7.8135e+00, 2.7218e+00, 1.8698e+01, 2.5163e+03]).to('cuda')
            else:
                weights = get_training_class_weights(train_dataset, label_key='label', do_not_consider_classes=[-1]).to('cuda')
            print('Using class weights for 5 classes:', weights)
            
            # weights = torch.tensor([0.2781, 13.5098,  3.3668, 30.7307, 0]).to('cuda')
        elif config.num_classes == 3: 
            print('Using class weights for 3 classes')
            weights = torch.tensor([0.367, 17.866, 4.452]).to('cuda')
    else:
        weights = None

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=mit_bih.make_collate_fn(config))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=mit_bih.make_collate_fn(config), num_workers=config.num_workers)

    test_dataset = dataset_class(config, split='test', augmentations=get_transforms(config, split='test'))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=mit_bih.make_collate_fn(config), num_workers=config.num_workers)
    print(f"Test dataset size: {len(test_dataset)}")

    if config.predict_no_hb:
        config.num_classes += 1
        
    base_model = utils.get_base_model(config, feature_classification=(not config.single_hb))

    if config.r_peaks_detection:
       model = TrainingRPeak(model=base_model, config=config, len_train_dataset=len(train_dataset), weights=weights)
    else:
        model = TrainingMIT_BIH(model=base_model, config=config, len_train_dataset=len(train_dataset), weights=weights)

    prj_string = f"train-mitbih-{config.num_classes}" if not config.r_peaks_detection else f"train-mitbih-r_peaks"
    trainer = utils.get_trainer(config, model, prj_string, wandb=wandb, run=run)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/train_mit_bih_defaults.yaml')

    train(config, wandb=config.wandb_log)