import lightning as L
import torch
import argparse
import utils.utils as utils
from torch.utils.data import DataLoader

from utils.utils import get_training_class_weights
from dataset.generic_utils import get_transforms
from trainers.sleep_apnea_trainer import TrainingSleepApnea

import dataset.sleep_apnea as sleep_apnea



import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_sleep_apnea_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)
    
    train_dataset =  sleep_apnea.ECGSleepApneaDataset(config, split='train', augmentations=get_transforms(config))
    print(f"Train dataset size: {len(train_dataset)}")
    val_dataset = sleep_apnea.ECGSleepApneaDataset(config, split='val', augmentations=get_transforms(config, split='val'))
    print(f"Val dataset size: {len(val_dataset)}")

    if config.use_class_weights:
        weights = get_training_class_weights(train_dataset, label_key='annotation').to('cuda')
        print(f"Using class weights: {weights}")
    else:
        weights = None


    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=sleep_apnea.make_collate_fn(config, split='train'), drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False, collate_fn=sleep_apnea.make_collate_fn(config, split='val'))

    test_dataset = sleep_apnea.ECGSleepApneaDataset(config, split='test', augmentations=get_transforms(config, split='test'))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False, collate_fn=sleep_apnea.make_collate_fn(config, split='test'))

    base_model = utils.get_base_model(config, feature_classification=config.is_recurrent)
            
    model = TrainingSleepApnea(model=base_model, config=config, len_train_dataset=len(train_dataset), weights=weights)

    trainer = utils.get_trainer(config, model, 'train-sleep-apnea', wandb=wandb, run=run)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')


# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = utils.parse_config(args.config_file, 'config_defaults/train_sleep_apnea_defaults.yaml')

    train(config, wandb=config.wandb_log)