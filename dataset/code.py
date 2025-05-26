from torch.utils.data import Dataset, random_split
import torch
import numpy as np
import wfdb
import os
import pandas as pd
from dataset.pretraining_dataset import PretrainDataset

class ECGCODE15Dataset(PretrainDataset):
    def __init__(self, config, global_augmentations=None, local_augmentations=None):
        """
        Args:
            records (list): List of records of ECG traces
        """
        super().__init__(config, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = config.data_folder_code15
        self.labels_file = config.labels_file_code15
        self.load_tabular_data()
        self.load_records()

    def load_records(self):
        self.records = self.tab_data.index.tolist()
        print(f'sample path CODE15: {self.records[0]}')
        print(f'loaded {len(self.records)} records')

    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file)
        # set exam_id as index
        self.tab_data.set_index('exam_id', inplace=True)

        print("tabular data fields for CODE 15: ", self.tab_data.head())


class ECGCODEDataset(PretrainDataset):
    def __init__(self, config, global_augmentations=None, local_augmentations=None):
        """
        Args:
            records (list): List of records of ECG traces
        """
        super().__init__(config, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = config.data_folder_code
        self.labels_file = config.labels_file_code
        self.load_tabular_data()
        self.load_records()

    def load_records(self):
        self.records = self.tab_data.index.tolist()
        print(f'sample path CODE: {self.records[0]}')
        print(f'loaded {len(self.records)} records')

    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file)
        # set exam_id as index
        self.tab_data.set_index('file_name', inplace=True)
        # remove trace_file, patient_id and nn_predicted_age
        print("tabular data fields for CODE: ", self.tab_data.head())
