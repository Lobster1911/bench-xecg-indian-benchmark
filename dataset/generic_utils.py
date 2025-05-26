import numpy as np
import os 
from tqdm import tqdm
import torch
from joblib import Parallel, delayed
from torchvision import transforms
from augmentations import *
import dataset.mit_bih as mit_bih
import dataset.code as code
import dataset.mimic_iv as mimic
import dataset.ptb_xl as ptb_xl
import dataset.chapman as chapman
import dataset.incart as incart
from torch.utils.data import Subset, ConcatDataset

def load_datasets(config):

    datasets_pretrain = []
    val_datasets = []

    for dataset in config.pretrain_datasets:
        if dataset == 'mimic':
            datasets_pretrain.append(mimic.ECGMIMICDataset(
                config, 
                split='train', 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
            val_datasets.append(mimic.ECGMIMICDataset(
                config, 
                split='val', 
                global_augmentations=get_transforms(config, split='train', type='global'), # I want the training augmentation in this case
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
        elif dataset == 'incart':
            # only training because small sample size
            incart_dataset = incart.ECGIncartDataset(
                config, 
                split='train', 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            )
            datasets_pretrain.append(incart_dataset)
            
        elif dataset == 'code15':
            code15 = code.ECGCODE15Dataset(
                config, 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            )
            # split the dataset into train and val
            train_size = int(0.9 * len(code15))
            train_code15, val_code15 = Subset(code15, range(0, train_size)), Subset(code15, range(train_size, len(code15)))
            datasets_pretrain.append(train_code15)
            val_datasets.append(val_code15)
        elif dataset == 'code':
            code = code.ECGCODEDataset(
                config, 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            )
            # split the dataset into train and val
            train_size = int(0.9 * len(code))
            train_code, val_code = Subset(code, range(0, train_size)), Subset(code15, range(train_size, len(code15)))
            datasets_pretrain.append(train_code)
            val_datasets.append(val_code)
        elif dataset == 'ptbxl':
            datasets_pretrain.append(ptb_xl.ECGPTBXLDataset(
                config, 
                split='train', 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
            val_datasets.append(ptb_xl.ECGPTBXLDataset(
                config, 
                split='val', 
                global_augmentations=get_transforms(config, split='train', type='global'), # I want the training augmentation in this case
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
        elif dataset == 'chapman':
            chapman_dataset = chapman.ECGChapmanDataset(
                config, 
                global_augmentations=get_transforms(config, split='train', type='global'),
                local_augmentations=get_transforms(config, split='train', type='local')
            )
            # split the dataset into train and val
            train_size = int(0.9 * len(chapman_dataset))
            train_chapman, val_chapman = Subset(chapman_dataset, range(0, train_size)), Subset(chapman_dataset, range(train_size, len(chapman_dataset)))
            datasets_pretrain.append(train_chapman)
            val_datasets.append(val_chapman)

        else:
            raise ValueError(f"Dataset {dataset} not found")

    val_dataset = ConcatDataset(val_datasets)
    train_dataset = ConcatDataset(datasets_pretrain)

    return train_dataset, val_dataset


def get_transforms(config, split='train', type=None):
    """
    """
    t = transforms.Compose([])
    if config.normalize:
        t.transforms.append(Normalize())
    
    if split != 'train': return t

    if config.random_crop < 1. and config.random_crop > 0.:
        t.transforms.append(RandomCrop(config.global_random_crop if type == 'global' else  config.local_random_crop if type == 'local' else config.random_crop))

    if config.shift_baseline_wander_in_sample:
        t.transforms.append(RandomShiftBaselineWander(config.sampling_freq, 0.5))

    if config.random_drop_leads > 0.:
        t.transforms.append(RandomDropLeads(config.random_drop_leads))

    if config.random_surrogate_prob > 0.:
        t.transforms.append(FTSurrogate(0.05, prob=config.random_surrogate_prob))
    if config.random_jitter_prob > 0.:  
        t.transforms.append(Jitter(sigma=0.1, prob=config.random_jitter_prob))
    if config.random_resample:
        t.transforms.append(RandomResample(config.sampling_freq, 0.03))
    
    if config.random_change_amplitude > 0.:
        t.transforms.append(RandomChangeAmplitude(amplitude_range=0.2, prob=config.random_change_amplitude))
    return t


def get_max_n_jobs(default=-1):
    n_jobs = int(os.getenv("SLURM_CPUS_PER_TASK", default))
    return n_jobs

    
def find_records(folder, header_extension='.dat'):
    def process_file(root, file):
        extension = os.path.splitext(file)[1]
        if extension == header_extension:
            record = os.path.relpath(os.path.join(root, file), folder)[:-len(header_extension)]
            return record
        return None

    records = set()

    print(f'Finding records in {folder}...')
    results = Parallel(n_jobs=get_max_n_jobs())(delayed(process_file)(root, file) for root, _, files in os.walk(folder) for file in files)
    records.update(filter(None, results))
    records = sorted(records)
    return records



def make_collate_fn(config):

    if config.shuffle_baseline_wander_in_batch:
        baseline_shuffler = RandomSwitchtBaselineWanderBatched(config.sampling_freq, 0.5)
    
    def collate_fn(batch):
        # Pad and clean global signals
        result = {
            'global_signals': pad_multi_view_batch(batch, 'global_signals', config.patch_size),
        }

        if config.shuffle_baseline_wander_in_batch:
            # Apply baseline shuffling to the global signals
            result['global_signals'] = [baseline_shuffler(signal) for signal in result['global_signals']]

        # Optional: handle local signals if present
        if 'local_signals' in batch[0] and batch[0]['local_signals'] is not None:
            result['local_signals'] = pad_multi_view_batch(batch, 'local_signals', config.patch_size)

            if config.shuffle_baseline_wander_in_batch:
                # Apply baseline shuffling to the local signals
                result['local_signals'] = [baseline_shuffler(signal) for signal in result['local_signals']]

        return result

    return collate_fn

def pad(x, patch_size):
    if x.dim() == 2:
        x = x.unsqueeze(-1)
        
    length = x.shape[1]
    excess = length % patch_size
    if excess != 0:
        x = x[:, :-excess, :]
    return x

def pad_multi_view_batch(batch, key, patch_size):
    signals = [sample[key] for sample in batch]
    signals = list(map(list, zip(*signals)))
    signals = [
        pad(torch.nn.utils.rnn.pad_sequence(g_signal, batch_first=True), patch_size)
        for g_signal in signals
    ]
    return signals