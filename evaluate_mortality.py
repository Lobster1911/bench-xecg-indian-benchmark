from numpy import indices
import torch
import argparse
import os
import wandb
from torch.utils.data import DataLoader

import dataset.music as music
import utils.utils as utils
from dataset.generic_utils import get_transforms, make_collate_fn_task
from trainers.mortality_trainer import TrainerMortality


parser = argparse.ArgumentParser(description='Train a model')
# add argument checkpoints array
parser.add_argument('--run_ids', type=str, help='List of run IDs to evaluate')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
parser.add_argument('--max_length_signal', type=int, default=3600, help='Max length of signal in seconds')
parser.add_argument('--data_folder_music', type=str, default='/media/Volume/data/MUSIC', help='Path to MUSIC data folder')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')

def evaluate(config, run_id):

    test_dataset = music.MUSICDataset(config, global_augmentations=get_transforms(config, split='test'))
    # consider only a 1%
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=config.num_workers, 
        collate_fn=music.make_collate_fn(config), 
        pin_memory=True
    )

    # set deterministic training
    run = wandb.init(
        project="train-mortality",
        id=run_id,
        resume="must"   # or "must" if you want to force resume
    )

    base_model = utils.get_base_model(config)
    model = TrainerMortality(model=base_model, config=config, len_train_dataset=0, evaluate_music=True)
    trainer = utils.get_trainer(config, model, f'train-mortality', wandb=wandb, run=run)
    
    try: 
        trainer.test(model=model, dataloaders=test_dataloader)
    except Exception as e:
        print(f"Error occurred while testing: {e}")

    # stop wandb run
    run.finish()
    wandb.finish()

    # remove testdataset and dataloader
    del test_dataset
    del test_dataloader
    del trainer
    del model
    del base_model
    torch.cuda.empty_cache()
    
# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = parser.parse_args()

    args.run_ids = args.run_ids.split(',')
    print(f"Evaluating run IDs: {args.run_ids}")

    for run_id in args.run_ids:
        # find the file config in the wandb folder i.e. wandb/run-20250821_143635-hq0hw9ne/files/config.yaml via regex, i know only the last part of the path
        # get list of files in the wandb folder
        wandb_folder = 'wandb'
        files = os.listdir(wandb_folder)
        # get the folder that ends with the run_id
        run_folder = [f for f in files if f.endswith(run_id)]
        if len(run_folder) == 0:
            print(f"Run ID {run_id} not found in wandb folder")
            continue

        run_folder = run_folder[0]
        # print(f"Found run folder: {run_folder}")
        run_path = os.path.join(wandb_folder, run_folder, 'files')
        # find the config file in the run folder
        config_file = [f for f in os.listdir(run_path) if f.endswith('.yaml') or f.endswith('.yml')]
        if len(config_file) == 0:
            print(f"No config file found in run folder {run_folder}")
            continue

        config_file = os.path.join(run_path, config_file[0])
        # print(f"Found config file: {config_file}")

        config = utils.parse_config(config_file, 'config_defaults/train_mortality_defaults.yaml')


        # find the checkpoint file in the run folder
        dir = os.path.join('train-mortality', run_id, 'checkpoints')
        # get the file ending with .ckpt
        checkpoint_files = [f for f in os.listdir(dir) if f.endswith('.ckpt')]
        if len(checkpoint_files) == 0:
            print(f"No checkpoint file found in run folder {run_folder}")
            continue

        config.checkpoint = os.path.join(dir, checkpoint_files[0])
        config.max_length_signal = args.max_length_signal * config.sampling_freq  # convert to samples
        config.batch_size = args.batch_size
        config.num_workers = args.num_workers
        config.labels_file_music = args.data_folder_music + '/subject-info.csv'
        config.data_folder_music = args.data_folder_music

        evaluate(config, run_id)
