import torch
import os
import pandas as pd
import wfdb
from dataset.pretraining_dataset import PretrainDataset
import numpy as np

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class ECGMIMICDataset(PretrainDataset):

    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None, downstream_task=None):
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = config.data_folder_mimic
        self.labels_file = config.labels_file_mimic
        self.downstream_task = downstream_task
        self.load_tabular_data()
        self.load_records(split)
        
    def load_records(self, split):
        # fold 19 is for testing, while fold 18 is for validation
        if split == 'train':
            # get all the tab data index where the fold is not 18 or 19
            self.tab_data = self.tab_data[self.tab_data['fold'] != 18][self.tab_data['fold'] != 19]
        elif split == 'val':
            # get all the tab data index where the fold is 18
            self.tab_data = self.tab_data[self.tab_data['fold'] == 18]
        elif split == 'test':
            # get all the tab data index where the fold is 19
            self.tab_data = self.tab_data[self.tab_data['fold'] == 19]
        elif split == 'all':
            # keep all the tab data index
            self.tab_data = self.tab_data

        self.unique_patients = list(self.tab_data['subject_id'].unique())
        self.patient_to_records = self.tab_data.groupby("subject_id")["file_name"].apply(list).to_dict()

        # print(f'MIMIC-IV: sample path: {self.records[0]}')
        # print(f'MIMIC-IV: loaded {len(self.records)} records')
        print(f'MIMIC-IV: number of unique patients {len(self.unique_patients)}')

    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file)

        if self.downstream_task == 'lvef':
            # load levf labels
            self.lvef = pd.read_csv(os.path.join(self.data_folder, 'lvef.csv'))
            # rename the columns waveform_path to file_name match the tabular data
            self.lvef.rename(columns={'waveform_path': 'file_name'}, inplace=True)
            # filter the tabular data to keep only the patients that have lvef
            self.tab_data = self.tab_data[self.tab_data['file_name'].isin(self.lvef['file_name'])]
            # add lvef labels to the tabular data
            self.tab_data = self.tab_data.merge(self.lvef[['file_name', 'LVEF']], on='file_name', how='left')

        elif self.downstream_task == 'age':
            # remove nan, negative and unrealistic ages
            initial_count = self.tab_data.shape[0]
            self.tab_data = self.tab_data[self.tab_data['age'].notna()]
            self.tab_data = self.tab_data[self.tab_data['age'] >= 0]
            self.tab_data = self.tab_data[self.tab_data['age'] <= 120]
            print(f'MIMIC-IV: filtered age records from {initial_count} to {self.tab_data.shape[0]}')

        # print the unique number of stratified folds
        print(f'MIMIC-IV: number of unique folds {self.tab_data["fold"].nunique()}')

        print("MIMIC-IV: tabular data fields", self.tab_data.head())
        print(f'MIMIC-IV: colums {self.tab_data.columns}')

    def __len__(self):
        if not self.downstream_task:
            # for downstream task, we return the number of unique patients
            return len(self.unique_patients)
        else:
            return len(self.tab_data)

    def __getitem__(self, idx):
        if not self.downstream_task:
            # for downstream task, we return the number of unique patients
            return self.get_item_pretraining(idx)
        elif self.downstream_task == 'lvef':
            # for pretraining, we return the number of records
            return self.get_item_lvef(idx)
        elif self.downstream_task == 'age':
            # for age prediction, we return the number of records
            return self.get_item_age(idx)
        
    def get_signal(self, idx):
        record = self.tab_data.iloc[idx]['file_name']
        signal, info = wfdb.rdsamp(os.path.join(self.data_folder, record))
        signal = self.map_leads_and_clean(signal, info)
        signal = self.resample_if_needed(signal, info)

        if self.global_augmentations is not None:
            signal = self.global_augmentations(signal)
        return signal

    def get_item_lvef(self, idx):
        signal = self.get_signal(idx)
        lvef = self.tab_data.iloc[idx]['LVEF']

        return {
            'signal': signal,
            'lvef': torch.tensor(lvef, dtype=torch.float32),
        }
    
    def get_item_age(self, idx):
        signal = self.get_signal(idx)
        age = self.tab_data.iloc[idx]['age']

        return {
            'signal': signal,
            'age': torch.tensor(age, dtype=torch.float32),
        }
    
    def get_item_pretraining(self, idx):
        patient = str(self.unique_patients[idx])
        records = self.patient_to_records[int(patient)]

        # records = self.tab_data[self.tab_data['subject_id'] == patient]['study_id'].tolist()
        num_views = self.n_global_view + self.n_local_view
        if len(records) > num_views:
            records = np.random.choice(records, num_views)

        unique_records = set([record for record in records])
        unique_signals = { record: wfdb.rdsamp(os.path.join(self.data_folder, record)) for record in unique_records }
        signals = [ unique_signals[record] for record in records ]

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
    
