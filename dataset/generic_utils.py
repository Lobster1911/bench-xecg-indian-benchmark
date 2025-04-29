import numpy as np
import os 
from tqdm import tqdm
import torch
from joblib import Parallel, delayed
from torchvision import transforms
from augmentations import RandomDropLeads, FTSurrogate, Jitter, RandomResample, Normalize, RandomCrop


def get_transforms(config):
    t = transforms.Compose([])
    if config.normalize:
        t.transforms.append(Normalize(mean=config.mean, std=config.std))
    if config.random_crop < 1.:
        t.transforms.append(RandomCrop(config.random_crop))
    if config.random_drop_leads > 0.:
        t.transforms.append(RandomDropLeads(config.random_drop_leads))
    if config.random_surrogate_prob > 0.:
        t.transforms.append(FTSurrogate(0.05, prob=config.random_surrogate_prob))
    if config.random_jitter_prob > 0.:  
        t.transforms.append(Jitter(sigma=0.1, prob=config.random_jitter_prob))
    if config.random_resample > 0.:
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


def collate_fn(batch):
    signals = [item['signal'] for item in batch]
    # pad the signals to the same length
    signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)

    tortn = {
        'signals': signals,
    }

    if 'signal_2' in batch[0].keys() is not None:    
        signals_2 = [item['signal_2'] for item in batch]
        # pad the signals to the same length
        signals_2 = torch.nn.utils.rnn.pad_sequence(signals_2, batch_first=True)
        tortn['signals_2'] = signals_2
        
    return tortn