import numpy as np
import os 
from tqdm import tqdm
import torch
from joblib import Parallel, delayed
from torchvision import transforms
from augmentations import RandomDropLeads, FTSurrogate, Jitter, RandomResample, Normalize, RandomCrop


def get_transforms(config, split='train', type=None):
    """
    """
    t = transforms.Compose([])
    if config.normalize:
        t.transforms.append(Normalize(mean=config.mean, std=config.std))
    if config.random_crop < 1. and split == 'train':
        if type == 'global':
            t.transforms.append(RandomCrop(config.global_random_crop))
        elif type == 'local':
            t.transforms.append(RandomCrop(config.local_random_crop))
        else:
            t.transforms.append(RandomCrop(config.random_crop))
    if config.random_drop_leads > 0. and split == 'train':
        t.transforms.append(RandomDropLeads(config.random_drop_leads))
    if config.random_surrogate_prob > 0. and split == 'train':
        t.transforms.append(FTSurrogate(0.05, prob=config.random_surrogate_prob))
    if config.random_jitter_prob > 0. and split == 'train':  
        t.transforms.append(Jitter(sigma=0.1, prob=config.random_jitter_prob))
    if config.random_resample and split == 'train':
        t.transforms.append(RandomResample(360, max_freq_delta=10))
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



def make_collate_fn(patch_size):
    def collate_fn(batch):
        # Pad and clean global signals
        result = {
            'global_signals': pad_multi_view_batch(batch, 'global_signals', patch_size),
        }

        # Optional: handle local signals if present
        if 'local_signals' in batch[0] and batch[0]['local_signals'] is not None:
            result['local_signals'] = pad_multi_view_batch(batch, 'local_signals', patch_size)

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