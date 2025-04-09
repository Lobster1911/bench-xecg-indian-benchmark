import torch
import os
import pandas as pd
import wfdb
import neurokit2 as nk
import numpy as np
from dataset.generic_utils import random_shift
from torch.utils.data import random_split
import json
import ast

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class ECGPTBXLDataset(torch.utils.data.Dataset):

    def __init__(self, config, leads_to_use=leads, split='train'):
        self.data_folder = config.data_folder_ptbxl
        self.random_shift = config.random_shift if split == 'train' else False
        self.nkclean = config.nk_clean
        self.leads = leads if leads_to_use == ['*'] else leads_to_use
        self.patch_size = config.patch_size
        self.normalize = config.normalize
        self.labels_file = config.labels_file_ptbxl
        self.split = split
        self.load_tabular_data()
        self.load_records(split)

    def load_records(self, split):
        # fold 19 is for testing, while fold 18 is for validation
        if split == 'train':
            # get all the tab data index where the fold is not 18 or 19
            self.records = self.tab_data[self.tab_data['strat_fold'] != 9][self.tab_data['strat_fold'] != 10]['filename_hr'].values.tolist()
        elif split == 'val':
            # get all the tab data index where the fold is 18
            self.records = self.tab_data[self.tab_data['strat_fold'] == 9]['filename_hr'].values.tolist()
        elif split == 'test':
            # get all the tab data index where the fold is 19
            self.records = self.tab_data[self.tab_data['strat_fold'] == 10]['filename_hr'].values.tolist()


    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file)
        self.tab_data.scp_codes = self.tab_data.scp_codes.apply(lambda x: ast.literal_eval(x))


        statements = pd.read_csv(os.path.join(self.data_folder, 'scp_statements.csv'), index_col=0)
        statements = statements[statements.diagnostic == 1]

        def aggregate_diagnostic(y_dic, statements=statements, column='diagnostic_class'):
            """
            Aggregate the diagnostic classes into superclasses and subclasses
            """
            tmp = []
            for key in y_dic.keys():
                if key in statements.index:
                    tmp.append(statements.loc[key].diagnostic_class)
            return list(set(tmp))
        
        self.tab_data['diagnostic_superclass'] = self.tab_data.scp_codes.apply(lambda x: aggregate_diagnostic(x, statements=statements, column='diagnostic_superclass'))
        self.tab_data['diagnostic_subclass'] = self.tab_data.scp_codes.apply(lambda x: aggregate_diagnostic(x, statements=statements, column='diagnostic_subclass'))

        self.classes = self.tab_data['diagnostic_superclass'].unique()
        print("Classes for PTB-XL: ", self.classes)
        self.subclasses = self.tab_data['diagnostic_subclass'].unique()
        print("Subclasses for PTB-XL: ", self.subclasses)
        
        # change type of age columns from float to int
        self.tab_data['age'] = self.tab_data['age'].fillna(0)
        self.tab_data['age'] = self.tab_data['age'].astype(int)
        # remove some unised columns
        print("Tabular data fields for  PTB-XL: ", self.tab_data.head())

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        signal, _ = wfdb.rdsamp(os.path.join(self.data_folder, self.records[idx].split('/')[1], self.records[idx].split('/')[2]))

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

        superclass_label = self.tab_data.iloc[idx]['diagnostic_superclass']
        # convert the superclass label to a one-hot encoding
        superclass_label = [1 if label in superclass_label else 0 for label in self.classes]

        subclass_label = self.tab_data.iloc[idx]['diagnostic_subclass']
        # convert the subclass label to a one-hot encoding
        subclass_label = [1 if label in subclass_label else 0 for label in self.subclasses]
        
        return {
            'signal':signal,
            'class_label': torch.tensor(superclass_label, dtype=torch.float32),
            'subclass_label': torch.tensor(subclass_label, dtype=torch.float32),
        }
    

    def collate_fn(self, batch):
        signals = [item['signal'] for item in batch]
        superclass_labels = [item['class_label'] for item in batch]
        subclass_labels = [item['subclass_label'] for item in batch]

        # pad the signals to the same length
        signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)

        return {
            'signals': signals,
            'class_labels': torch.stack(superclass_labels),
            'subclass_labels': torch.stack(subclass_labels),
        }