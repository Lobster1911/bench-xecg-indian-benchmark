import torch
import os
import pandas as pd
import wfdb
import neurokit2 as nk
import numpy as np
from dataset.generic_utils import random_shift
import json

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class ECGMIMICDataset(torch.utils.data.Dataset):

    def __init__(self, config, leads_to_use=leads, split='train'):
        self.data_folder = config.data_folder_mimic
        self.random_shift = config.random_shift and split == 'train'
        self.leads = leads if leads_to_use == ['*'] else leads_to_use
        self.patch_size = config.patch_size
        self.normalize = config.normalize
        self.labels_file = config.labels_file_mimic
        self.split = split
        self.load_tabular_data()
        self.load_records(split)

    def load_records(self, split):
        # fold 19 is for testing, while fold 18 is for validation
        if split == 'train':
            # get all the tab data index where the fold is not 18 or 19
            self.records = self.tab_data[self.tab_data['fold'] != 18][self.tab_data['fold'] != 19].index.tolist()
        elif split == 'val':
            # get all the tab data index where the fold is 18
            self.records = self.tab_data[self.tab_data['fold'] == 18].index.tolist()
        elif split == 'test':
            # get all the tab data index where the fold is 19
            self.records = self.tab_data[self.tab_data['fold'] == 19].index.tolist()

    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file)
        # set exam_id as index
        self.tab_data.set_index('study_id', inplace=True)
        # change type of age columns from float to int
        self.tab_data['age'] = self.tab_data['age'].fillna(0)
        self.tab_data['age'] = self.tab_data['age'].astype(int)
        # remove some unised columns
        self.tab_data.drop(columns=['file_name', 'subject_id', 'hosp_diag_hosp', 'ecg_taken_in_ed', 'gender'], inplace=True)
        print("tabular data fields for  MIMIC-IV: ", self.tab_data.head())

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = str(self.records[idx])

        signal, _ = wfdb.rdsamp(os.path.join(self.data_folder, record, record))
        
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

        if '/' in str(record): record = record.split('/')[0]

        return {
            'signal':signal,
        }
         