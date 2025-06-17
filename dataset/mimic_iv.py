import torch
import os
import pandas as pd
import wfdb
from dataset.pretraining_dataset import PretrainDataset
import numpy as np

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class ECGMIMICDataset(PretrainDataset):

    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = config.data_folder_mimic
        self.labels_file = config.labels_file_mimic
        self.load_tabular_data()
        self.load_records(split)
        
    def load_records(self, split):
        # fold 19 is for testing, while fold 18 is for validation
        if split == 'train':
            # get all the tab data index where the fold is not 18 or 19
            self.records = self.tab_data[self.tab_data['fold'] != 18][self.tab_data['fold'] != 19]['file_name'].tolist()
        elif split == 'val':
            # get all the tab data index where the fold is 18
            self.records = self.tab_data[self.tab_data['fold'] == 18]['file_name'].tolist()
        elif split == 'test':
            # get all the tab data index where the fold is 19
            self.records = self.tab_data[self.tab_data['fold'] == 19]['file_name'].tolist()

        self.unique_patients = list(self.tab_data['subject_id'].unique())
        self.patient_to_records = self.tab_data.groupby("subject_id")["file_name"].apply(list).to_dict()

        print(f'MIMIC-IV: sample path: {self.records[0]}')
        print(f'MIMIC-IV: loaded {len(self.records)} records')
        print(f'MIMIC-IV: number of unique patients {len(self.unique_patients)}')

    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file)
        # set exam_id as index
        # self.tab_data.set_index('study_id', inplace=True)
        # change type of age columns from float to int
        # self.tab_data['age'] = self.tab_data['age'].fillna(0)
        # self.tab_data['age'] = self.tab_data['age'].astype(int)
        # remove some unised columns
        print("MIMIC-IV: tabular data fields", self.tab_data.head())
        print(f'MIMIC-IV: colums {self.tab_data.columns}')

    def __len__(self):
        return len(self.unique_patients)

    def __getitem__(self, idx):
        patient = str(self.unique_patients[idx])
        records = self.patient_to_records[int(patient)]

        # records = self.tab_data[self.tab_data['subject_id'] == patient]['study_id'].tolist()
        num_views = self.n_global_view + self.n_local_view
        if len(records) > num_views:
            records = np.random.choice(records, num_views)

        signals = [ wfdb.rdsamp(os.path.join(self.data_folder, record)) for record in records ]

        # mapping leads in the correct position
        new_signals = []
        for signal in signals:
            s, info = signal
            s = self.map_leads_and_clean(s, info)
            s = self.resample_if_needed(s, info)
            new_signals.append(s)
        signals = new_signals
    
        if self.global_augmentations is not None:
            global_signals = [ self.global_augmentations(signals[i % len(signals)]) for i in range(self.n_global_view)]
        else:
            global_signals = s
        
        if self.local_augmentations is not None and self.n_local_view > 0:
            local_signals = [ self.local_augmentations(signals[(i + self.n_global_view)% len(signals)]) for i in range(self.n_local_view)]
        else:
            local_signals = []

        return  {
            'global_signals': global_signals,
            'local_signals': local_signals,
        }