import numpy as np
import os 
from tqdm import tqdm
import torch
from joblib import Parallel, delayed
from torchvision import transforms
from augmentations import *


def get_transforms(config, split='train', type=None):
    """
    """
    t = transforms.Compose([])
    if config.normalize:
        t.transforms.append(Normalize())

    if config.standardize:
        t.transforms.append(Standardize())

    if config.low_pass_filter:
        t.transforms.append(LowpassFilter(config.sampling_freq, config.low_pass_filter))
    
    if config.high_pass_filter:
        t.transforms.append(HighpassFilter(config.sampling_freq, config.high_pass_filter))
    
    if split != 'train': 
        t.transforms.append(CropFixedLen(config.max_length_signal))
        return t

    if config.random_crop < 1. and config.random_crop > 0.:
        t.transforms.append(RandomCrop(
            config.global_random_crop if type == 'global' else  config.local_random_crop if type == 'local' else config.random_crop,
            max_length=config.max_length_signal
        ))
    else:
        t.transforms.append(CropFixedLen(config.max_length_signal))

    if config.shift_baseline_wander_in_sample:
        t.transforms.append(RandomShiftBaselineWander(config.sampling_freq, 0.5))

    if config.random_drop_leads > 0.:
        t.transforms.append(RandomDropLeads(config.random_drop_leads, keep_lead_II = config.keep_lead_II))

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


def make_collate_fn_task(config, key_label='age'):
    def collate_fn(batch):
        if 'signal' in batch[0]:
            # If 'signal' is present, use it
            signals = [item['signal'] for item in batch]
            signals = pad(torch.nn.utils.rnn.pad_sequence(signals, batch_first=True), patch_size=config.patch_size)
        else:
            signals = [item['global_signals'][0] for item in batch]
            signals = pad(torch.nn.utils.rnn.pad_sequence(signals, batch_first=True), patch_size=config.patch_size)

        return {
            'signals': signals,
            key_label: torch.stack([item[key_label] for item in batch])
        }

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