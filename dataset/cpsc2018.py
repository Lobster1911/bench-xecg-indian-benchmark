from torch.utils.data import Dataset, random_split
import torch
import numpy as np
import wfdb
import os
import pandas as pd
from dataset.pretraining_dataset import PretrainDataset
from dataset.generic_utils import pad, RandomSwitchtBaselineWanderBatched


class ECGCPSC2018Dataset(PretrainDataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        """
        Args:
            records (list): List of records of ECG traces
        """
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = config.data_folder_cpsc2018
        self.labels_file = config.labels_file_cpsc2018

        self.tab_data = pd.read_csv(self.labels_file)
        self.load_records()
        self.load_labels()  

    def load_records(self):
        self.records = self.tab_data['file_name'].tolist()
        if self.split == 'train':
            self.records = [record for record in self.records if 'g7' not in record and 'g6' not in record]
        elif self.split == 'val':
            self.records = [record for record in self.records if 'g6' in record]
        elif self.split == 'test':
            self.records = [record for record in self.records if 'g7' in record]

        print(f'sample path CODE15: {self.records[0]}')
        print(f'loaded {len(self.records)} records')

    def load_labels(self):
        labels = self.tab_data['diagnosis_code'].unique()
        splitted_labels = []
        for label in labels:
            splitted_labels.extend(label.split(','))

        self.labels = list(set(splitted_labels))

        print(f'loaded {len(self.labels)} labels: {self.labels}')

    
    def __getitem__(self, idx):
        signal = super().__getitem__(idx)
        info = wfdb.rdheader(os.path.join(self.data_folder, self.records[idx]))

        label_str = extract_diagnosis_code(info).split(',')
        labels = [1 if label in label_str else 0 for label in self.labels]
        
        return {
            'signal': signal['global_signals'][0],
            'labels': torch.tensor(labels, dtype=torch.float32)
        }


def extract_diagnosis_code(record):
    for comment in record.comments:
        if comment.startswith('Dx:'):
            return comment.split(': ')[1]
    return None


def make_collate_fn(config, split='train'):

    if config.shuffle_baseline_wander_in_batch:
        baseline_shuffler = RandomSwitchtBaselineWanderBatched(config.sampling_freq, 0.5)

    def collate_fn(batch):
        signals = [item['signal'] for item in batch]
        class_labels = [item['labels'] for item in batch]

        # pad the signals to the same length
        if split == 'train' and config.shuffle_baseline_wander_in_batch:
            signals = baseline_shuffler(pad(torch.nn.utils.rnn.pad_sequence(signals, batch_first=True), patch_size=config.patch_size))
        else:
            signals = pad(torch.nn.utils.rnn.pad_sequence(signals, batch_first=True), patch_size=config.patch_size)
            
        return {
            'signals': signals,
            'class_labels': torch.stack(class_labels),
        }
    return collate_fn