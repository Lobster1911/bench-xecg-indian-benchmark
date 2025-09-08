
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
        self.leads = config.leads
        self.patch_size = config.patch_size
        self.window_size = config.window_size * 100

        self.augmentations = augmentations
        self.segment_size = 6000  # 60 seconds in samples

        if self.sampling_freq % self.patch_size != 0:
            print(f"Warning: Sampling freq {self.sampling_freq} should be divisible by patch size {self.patch_size}")
        if self.window_size % self.segment_size != 0 and config.is_recurrent:
            raise ValueError(f"Window size {self.window_size} must be divisible by segment_size {self.segment_size}")

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
            records = records[:int(len(records) * 0.8)]
            print(f"Training records: {records}")
        elif self.split == 'val':
            records = [record for i, record in enumerate(records) if not i in test_idxs]
            # keep the 10% of the records
            records = records[int(len(records) * 0.8):]
            print(f"Validation records: {records}")

        elif self.split == 'test':
            records = [record for i, record in enumerate(records) if i in test_idxs]
            print(f"Test records: {records}")

        self.records = [os.path.join(self.data_folder, record) for record in records]

    def load_samples(self):
        # for each record, divide it into 5 min segments
        # and save the segments in a list
        self.samples = []
        self.annotations = []
        self.segment_id = []

        # for each ecg record, read the signal and annotations
        for record in tqdm(self.records, desc='Loading samples'):
            # read the record
            signal, _ = wfdb.rdsamp(record)
            ann = wfdb.rdann(record, 'apn')

            # get the length of the signal (considering it only as a multiple of segment_size)
            length = len(signal) - (len(signal) % self.window_size)

            # divide the signal into window_size segments
            count = 0
            if self.window_size < self.segment_size: count_2 = 0

            for i in range(0, length, self.window_size):
                # append the segment  of window_size to the samples list
                sample = signal[i:min(length, i + self.window_size)]

                # get the annotations for the segment
                annotations = []
                while count < len(ann.sample) and ann.sample[count] < i + self.window_size:
                    annotations.append(1. if ann.symbol[count] == 'A' else 0.)
                    self.segment_id.append((record, count))
                    if self.window_size < self.segment_size: 
                        count_2 += 1
                        # case of transformer where segment size is smaller than window size (10 seconds)
                        if count_2  % (self.segment_size // self.window_size) == 0:
                            count += 1
                        break
                    else:
                        count += 1

                if len(annotations) > 0:
                    num_minutes = sample.shape[0] // self.segment_size
                    if len(annotations) < num_minutes:
                        # print(f"Warning: Less annotations ({len(annotations)}) than expected ({num_minutes}) for record {record}, segment {i}, cropping")
                        sample = sample[:len(annotations) * self.segment_size]

                    self.samples.append(sample)
                    self.annotations.append(torch.tensor(annotations))
                # else:
                    # print(f"Warning: No annotations found for segment {i} in record {record}, skipping... (This should not happen)")
            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ann = self.annotations[idx]

        sample = self.resample_if_needed(sample, {'fs': 100})  # original sampling freq is 100 Hz

        # map to the correct lead
        tensor = torch.zeros(sample.shape[0], len(self.leads), dtype=torch.float32)
        if len(self.leads) == 1:
            tensor[:, 0] = sample[:, 0]  # ECG
        else:
            tensor[:, 1] = sample[:, 0]  # ECG


        if self.augmentations is not None:
            tensor = self.augmentations(tensor)

        return {
            'signal': tensor,
            'annotation': ann,
            'segment_id': self.segment_id[idx],
        }
    
    def resample_if_needed(self, signal, info):
        if self.sampling_freq != info['fs']:
            signal = nk.signal_resample(signal, sampling_rate=info['fs'], desired_sampling_rate=self.sampling_freq, method='FFT')   
        signal = torch.tensor(signal, dtype=torch.float32)
        return signal
    

def make_collate_fn(config, split='train'):

    if config.shuffle_baseline_wander_in_batch:
        baseline_shuffler = RandomSwitchtBaselineWanderBatched(config.sampling_freq, 0.5)
    
    def collate_fn(batch):
        signals = [item['signal'] for item in batch]
        labels = [item['annotation'] for item in batch]
        segment_ids = [item['segment_id'] for item in batch]

        # pad to same length and pad to match the patch size module
        if config.shuffle_baseline_wander_in_batch and split == 'train':
            signals = baseline_shuffler(torch.nn.utils.rnn.pad_sequence(signals, batch_first=True))
        else:
            signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
            
        return {
            'signals': signals,
            'labels': labels,
            'segment_ids': segment_ids,
        }

    return collate_fn