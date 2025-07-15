
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
        self.discard_bad_samples = config.discard_bad_samples
        self.discard_augmented_samples = config.discard_augmented_samples

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

        if self.discard_bad_samples:
            # keep only records where qa_label is [1, 0, 0]
            valid_indices = np.where((self.qa_label[:, 0] == 1) & (self.qa_label[:, 1] == 0) & (self.qa_label[:, 2] == 0))[0]
            if len(valid_indices) > 0:
                print(f"Found {len(valid_indices)} good quality signals, keeping them")
                self.signals = self.signals[valid_indices]
                self.qa_label = self.qa_label[valid_indices]
                self.rhythm = self.rhythm[valid_indices]
                self.parameters = self.parameters[valid_indices]
            else:
                print("No valid signals found, using all signals")

        if self.discard_augmented_samples:
            last_datetime = None
            last_patient = None
            to_keep = []

            print(f"Number of samples before filtering: {len(self.parameters)}")

            for i, p in tqdm(enumerate(self.parameters)):
                if last_datetime is None or last_patient is None:
                    last_datetime = p[0]
                    last_patient = p[2]
                    to_keep.append(i)
                else:
                    # difference between timestamp is less than 25 seconds
                    if isinstance(p[0], str):
                        p[0] = datetime.strptime(p[0].strip(), '%d-%b-%Y %H:%M:%S')
                    if p[2] == last_patient and (p[0] - last_datetime) <= timedelta(seconds=24):
                        # keep the first one
                        continue
                    else:
                        last_datetime = p[0]
                        last_patient = p[2]
                        to_keep.append(i)

            print(f"Number of samples after filtering: {len(to_keep)}")
            self.signals = self.signals[to_keep]
            self.qa_label = self.qa_label[to_keep]
            self.rhythm = self.rhythm[to_keep]
            self.parameters = self.parameters[to_keep]
            
        self.rhythm_label = torch.from_numpy(self.rhythm).float()

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]

        signal = self.resample_if_needed(signal, self.info_dict)
        signal = self.map_leads_and_clean(signal, self.info_dict)

        if self.global_augmentations is not None:
            signal = self.global_augmentations(signal)

        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).float()

        labels = self.rhythm_label[idx]

        out = {
            'signal': signal.float(),
            'label': labels,
        }
        return out