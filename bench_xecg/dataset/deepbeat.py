
from pathlib import Path

import torch
import yaml
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from tqdm import tqdm
from dataset.pretraining_dataset import PretrainDataset

class DeepBeatDataset(PretrainDataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = Path(config.data_folder_deepbeat)
        self.signals = None
        self.info_dict = {"fs": 32, "sig_name": ["II"]}

        if split == 'train':
            path = self.data_folder / 'train.npz'
        elif split == 'val':
            path = self.data_folder / 'validate.npz'
        elif split == 'test':
            path = self.data_folder / 'test.npz'
        else:
            raise ValueError(f'Unknown split {split}')

        data = np.load(path, allow_pickle=True)
        self.signals = data['signal']
        print(f"Loaded {len(self.signals)} signals from {path} with shape {self.signals.shape}")

        self.qa_label = data['qa_label']
        self.rhythm = data['rhythm']
        self.parameters = data['parameters']

        # get index of nan signals
        nan_indices = np.where(np.isnan(self.signals).any(axis=1))[0]

        if len(nan_indices) > 0:
            print(f"Found {len(nan_indices)} signals with NaN values, removing them")
            self.signals = np.delete(self.signals, nan_indices, axis=0)
            self.rhythm = np.delete(self.rhythm, nan_indices, axis=0)
            self.qa_label = np.delete(self.qa_label, nan_indices, axis=0)
            self.parameters = np.delete(self.parameters, nan_indices, axis=0)
            
        self.rhythm_label = torch.from_numpy(self.rhythm).float()

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]

        signal = self.resample_if_needed(signal, self.info_dict)
        signal = self.map_leads_and_clean(signal, self.info_dict)

        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).float()

        if self.global_augmentations is not None:
            signal = self.global_augmentations(signal)

        labels = self.rhythm_label[idx]

        out = {
            'signal': signal.float(),
            'label': labels,
        }
        return out