import torch
import os
import pandas as pd
import wfdb
import neurokit2 as nk
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import neurokit2 as nk
from dataset.generic_utils import RandomSwitchtBaselineWanderBatched
from dataset.pretraining_dataset import PretrainDataset
from typing_extensions import override

class ECGIncartDataset(PretrainDataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        """
        Args:
            config: configuration object
            split: 'train', 'val'or 'test'
        """
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        
        self.data_folder = config.data_folder_incart
        self.win_len = 10

        self.load_records(split)

    def load_records(self, subset):
        with open(os.path.join(self.data_folder, 'RECORDS'), 'r') as f:
            self.records = [line.strip()[:3] for line in f.readlines()]
            print(f'INCART: loaded {len(self.records)} records for {subset}')

    @override
    def __getitem__(self, idx):
        patient = self.records[idx]

        signal, info = wfdb.rdsamp(os.path.join(self.data_folder, str(patient)))
        # get a random 10s window

        sample_fs = info['fs']
        random_start = np.random.randint(0, len(signal) - self.win_len * sample_fs)
        random_end = random_start + self.win_len * sample_fs
        
        signal = signal[random_start:random_end]

        self.map_leads_and_clean(signal, info)
        signal = self.resample_if_needed(signal, info)

        if self.global_augmentations is not None:
            global_signals = [ self.global_augmentations(signal) for _ in range(self.n_global_view)]
        else:
            global_signals = signal
        
        if self.local_augmentations is not None and self.n_local_view > 0:
            local_signals = [ self.local_augmentations(signal) for _ in range(self.n_local_view)]
        else:
            local_signals = None

        return  {
            'global_signals': global_signals,
            'local_signals': local_signals,
        }