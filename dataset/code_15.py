from torch.utils.data import Dataset, random_split
import torch
import numpy as np
import wfdb
import os
import pandas as pd
from dataset.pretraining_dataset import PretrainDataset

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class ECGCODE15Dataset(PretrainDataset):
    def __init__(self, config, leads_to_use=leads, augmentations=None):
        """
        Args:
            records (list): List of records of ECG traces
        """
        super().__init__(config, leads_to_use=leads_to_use, augmentations=augmentations)
        self.data_folder = config.data_folder_code15
        self.labels_file = config.labels_file_code15
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

