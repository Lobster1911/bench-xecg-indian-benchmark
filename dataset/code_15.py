from torch.utils.data import Dataset, random_split
import torch
import numpy as np
import wfdb
import os
import pandas as pd
from dataset.generic_utils import random_shift

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class ECGCODE15Dataset(Dataset):
    def __init__(self, config, leads_to_use=leads):
        """
        Args:
            records (list): List of records of ECG traces
        """
        self.data_folder = config.data_folder_code15
        self.labels_file = config.labels_file_code15
        self.normalize = config.normalize 
        self.leads = leads if leads_to_use == ['*'] else leads_to_use
        self.random_shift = config.random_shift
        self.patch_size = config.patch_size
        self.load_tabular_data()
        self.load_records()

    def load_records(self):
        self.records = self.tab_data.index.tolist()
        print(f'loaded {len(self.records)} records')

    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file)
        # set exam_id as index
        self.tab_data.set_index('exam_id', inplace=True)
        # remove trace_file, patient_id and nn_predicted_age
        self.tab_data.drop(columns=['trace_file', 'patient_id', 'nn_predicted_age', 'death', 'timey'], inplace=True)
        print("tabular data fields for CODE 15: ", self.tab_data.head())
        # 1dAVB = 144.0
        # RBBB = 145.1
        # LBBB = 144.7
        # SB (sinus bradycardya) R00.1
        # ST (sinus tachycardya) I147.1, R00.0
        # AF (atrial fibrillation) I48

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = str(self.records[idx])

        signal, _ = wfdb.rdsamp(os.path.join(self.data_folder, record, record))

        # remove a random number of datapoints from the signal from 0 to patch size 
        if self.random_shift: signal = random_shift(signal, self.patch_size)

        signal = torch.tensor(signal, dtype=torch.float32)

        if self.leads != leads:
            # keep the selected leads
            signal = signal[:, [leads.index(lead) for lead in self.leads]].squeeze()
            
        # normalize the signal by subtracting the mean and dividing by the standard deviation
        if self.normalize:
            std = signal.std(axis=(0, -1))
            std[std == 0] = 1 # avoid division by zero, samples with std = 0 are all zero
            signal = (signal - signal.mean(axis=(0, -1))) / std

        return {
            'signal':signal,
        }
