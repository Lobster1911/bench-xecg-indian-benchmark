import torch
import os
import pandas as pd
import wfdb
from dataset.pretraining_dataset import PretrainDataset


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

    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file)
        # set exam_id as index
        self.tab_data.set_index('study_id', inplace=True)
        # change type of age columns from float to int
        self.tab_data['age'] = self.tab_data['age'].fillna(0)
        self.tab_data['age'] = self.tab_data['age'].astype(int)
        # remove some unised columns
        self.tab_data.drop(columns=['subject_id', 'hosp_diag_hosp', 'ecg_taken_in_ed', 'gender'], inplace=True)
        print("tabular data fields for  MIMIC-IV: ", self.tab_data.head())
        print(f'mimic colums {self.tab_data.columns}')

    def __len__(self):
        return len(self.records)
         