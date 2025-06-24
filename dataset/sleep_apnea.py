
import torch
import os
import pandas as pd
import wfdb
import neurokit2 as nk
import numpy as np
from tqdm import tqdm
from dataset.generic_utils import pad, RandomSwitchtBaselineWanderBatched


class ECGSleepApneaDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train', augmentations=None):
        self.patch_size = config.patch_size
        self.sampling_freq = config.sampling_freq
        self.data_folder = config.data_folder_sleep_apnea
        self.split = split
        self.patch_size = config.patch_size
        if self.split == 'train':
            self.window_size = config.window_size_train
        else:
            self.window_size = config.window_size_val
        self.augmentations = augmentations
        self.segment_size = 6000

        if self.sampling_freq % self.patch_size != 0:
            raise ValueError(f"Patch size {self.patch_size} must be divisible by sampling frequency {self.sampling_freq}")
        if self.window_size % self.segment_size != 0:
            raise ValueError(f"Window size {self.window_size} must be divisible by sampling frequency {self.sampling_freq}")

        self.load_records()
        self.load_samples()

    def load_records(self):
        # list os files from the RECORDS file
        with open(os.path.join(self.data_folder, 'RECORDS'), 'r') as f:
            records = f.readlines()

        # keep only the first 4 characters of each line
        records = [record.strip()[:3] for record in records]
        records = list(set(records))  # remove duplicates

        test_idxs = [i for i, record in enumerate(records) if record.startswith('x')]

        if self.split == 'train':
            records = [record for i, record in enumerate(records) if i not in test_idxs]
            # keep the 90% of the records
            records = records[:int(len(records) * 0.9)]
        elif self.split == 'val':
            records = [record for i, record in enumerate(records) if i in test_idxs]
            # keep the 10% of the records
            records = records[int(len(records) * 0.9):]
        elif self.split == 'test':
            records = [record for i, record in enumerate(records) if i in test_idxs]

        self.records = [os.path.join(self.data_folder, record) for record in records]

    def load_samples(self):
        # for each record, divide it into 5 min segments
        # and save the segments in a list
        self.samples = []
        self.annotations = []

        for record in tqdm(self.records, desc='Loading samples'):
            
            # read the record
            signal, info = wfdb.rdsamp(record)
            ann = wfdb.rdann(record, 'apn')

            # get the length of the signal
            length = len(signal)
            # divide the signal into 5 min segments
            count = 0
            for i in range(0, length, self.window_size):
                if i + self.window_size < length:
                    self.samples.append(signal[i:i + self.window_size])
                else:
                    self.samples.append(signal[i:length])

                annotations = []
                # get the annotations for the segment
                while count < len(ann.sample) and ann.sample[count] < i + self.window_size:
                    annotations.append((ann.sample[count] - i, ann.symbol[count]))
                    count += 1

                if len(annotations) == 0:
                    # do not add the segment
                    self.samples.pop()
                    continue

                self.annotations.append(annotations)
                # print(f"segment {i} - {i + self.window_size} with {len(annotations)} annotations")
        

        patches_in_segment = self.segment_size // self.patch_size

        annotations_tmp = []
        # covnert the annotations to torch tensors
        for i, ann_list in enumerate(self.annotations):
            ann = []
            for pos, label in ann_list:
                if label == 'A':
                    ann.append(torch.ones(patches_in_segment, dtype=torch.float32))
                elif label == 'N':
                    ann.append(torch.zeros(patches_in_segment, dtype=torch.float32))
                else:   
                    raise ValueError(f"Unknown label {label}")
            ann = torch.cat(ann, dim=0)

            if len(ann) > len(self.samples[i]) // self.patch_size:
                ann = ann[:len(self.samples[i]) // self.patch_size]
                # print(f'ann shape: {ann.shape} should match {len(self.samples[i]) / self.patch_size}')
            elif len(ann) < len(self.samples[i]) // self.patch_size:
                self.samples[i] = self.samples[i][:len(ann) * self.patch_size]
                print(f'ann shape: {ann.shape} should match {len(self.samples[i]) / self.patch_size}, should never see this')
                
            annotations_tmp.append(ann)

        self.annotations = annotations_tmp
            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ann = self.annotations[idx]

        # map to the correct lead
        tensor = torch.zeros(sample.shape[0], 12)
        tensor[:, 1] = torch.tensor(sample[:, 0], dtype=torch.float32)  # ECG


        if self.augmentations is not None:
            sample = self.augmentations(sample)

        return {
            'signal': tensor,
            'annotation': ann,
        }
    

def make_collate_fn(config, split='train'):

    if config.shuffle_baseline_wander_in_batch:
        baseline_shuffler = RandomSwitchtBaselineWanderBatched(config.sampling_freq, 0.5)
    
    def collate_fn(batch):
        signals = [item['signal'] for item in batch]
        labels = [item['annotation'] for item in batch]

        # pad to same length and pad to match the patch size module
        if config.shuffle_baseline_wander_in_batch and split == 'train':
            signals = baseline_shuffler(torch.nn.utils.rnn.pad_sequence(signals, batch_first=True))
        else:
            signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
            
        return {
            'signals': signals,
            'labels': labels,
        }

    return collate_fn