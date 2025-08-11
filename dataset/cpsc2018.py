from torch.utils.data import Dataset, random_split
import torch
import numpy as np
import wfdb
import os
import pandas as pd
from dataset.pretraining_dataset import PretrainDataset
from dataset.generic_utils import pad, RandomSwitchtBaselineWanderBatched
from tqdm import tqdm


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
        self.load_labels(config.task)  

    def load_records(self):
        self.records = self.tab_data['file_name'].tolist()
        if self.split == 'train':
            self.records = [record for record in self.records if 'g7' not in record and 'g6' not in record]
        elif self.split == 'val':
            self.records = [record for record in self.records if 'g6' in record]
        elif self.split == 'test':
            self.records = [record for record in self.records if 'g7' in record]
        elif self.split == 'all':
            pass
        
        self.tab_data = self.tab_data[self.tab_data['file_name'].isin(self.records)]
        print('CPSC2018:', self.tab_data.head())


    def load_labels(self, task='multilabel'):
        labels = self.tab_data['diagnosis_code'].unique()
        splitted_labels = []
        for label in labels:
            splitted_labels.extend(label.split(','))

        self.labels_unique = list(set(splitted_labels))
        labels = []

        new_records = []

        if task == 'multiclass':
            print(f'loaded {len(self.records)} records before removing multi label labels')

        for record in tqdm(self.records, desc='Processing labels'):
            info = wfdb.rdheader(os.path.join(self.data_folder, record))
            label_str = extract_diagnosis_code(info).split(',')
            num_labels = sum([1 if label in label_str else 0 for label in self.labels_unique])
            if task=='multiclass' and num_labels == 1:
                new_records.append(record)
                labels.append(label_str)
            elif task == 'multilabel':
                new_records.append(record)
                labels.append(label_str)

        # print number of sample for each label
        for label in self.labels_unique:
            count = sum([1 for label_str in labels if label in label_str])
            print(f'Label {label} has {count} samples')

        self.records = new_records
        print(f'loaded {len(self.records)} records')
    
    def __getitem__(self, idx):
        signal = super().__getitem__(idx)
        info = wfdb.rdheader(os.path.join(self.data_folder, self.records[idx]))

        label_str = extract_diagnosis_code(info).split(',')
        labels = [1 if label in label_str else 0 for label in self.labels_unique]

        signal = signal['global_signals'][0]  # Assuming global_signals is a list of signals
        # if signal.shape[0] > self.sampling_freq * 10:
        #   signal = signal[:self.sampling_freq * 10]

        return {
            'signal': signal,
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


class ECGCPSC2018AgeDataset(ECGCPSC2018Dataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        """
        Args:
            records (list): List of records of ECG traces
        """
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.filter_valid_records()

    def filter_valid_records(self):
        # filter records that have age in the comments
        valid_records = []
        for record in self.records:
            info = wfdb.rdheader(os.path.join(self.data_folder, record))
            age = None
            for comment in info.comments:
                if comment.startswith('Age:'):
                    age = comment.split(': ')[1]
                    try:
                        int_age = int(age)
                        if int_age < 0 or int_age > 120:
                            print(f'Invalid age {age} in record {record}')
                        else:
                            valid_records.append(record)
                    except:
                        print(f'Invalid age {age} in record {record}')
                    break
        self.records = valid_records
        print(f'Filtered to {len(self.records)} records with age information')

    def __getitem__(self, idx):
        obj = super().__getitem__(idx)
        info = wfdb.rdheader(os.path.join(self.data_folder, self.records[idx]))
        info.comments
        # regex 'Age: (\d+)' to extract age from comments
        age = None
        for comment in info.comments:
            if comment.startswith('Age:'):
                age = int(comment.split(': ')[1])
                break
        if age is None:
            raise ValueError(f'Age not found in comments for record {self.records[idx]}')

        obj['age'] = torch.tensor(age, dtype=torch.float32)
        return obj

