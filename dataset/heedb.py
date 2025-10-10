
import torch
import numpy as np
import wfdb
import os
import pandas as pd
from dataset.pretraining_dataset import PretrainDataset

class ECGHEEDBDataset(PretrainDataset):
    def __init__(self, config, global_augmentations=None, local_augmentations=None):
        """
        Args:
            records (list): List of records of ECG traces
        """
        super().__init__(config, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = config.data_folder_heedb
        self.load_tabular_data()
        self.load_records()

    def load_records(self):
        self.records = self.tab_data.index.tolist()
        print(f'HEEDB: sample path: {self.records[0]}')
        print(f'HEEDB: loaded {len(self.records)} records')
        print(f'HEEDB: number of unique patients {len(self.unique_patients)}')

    def __len__(self):
        return len(self.unique_patients)


    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(os.path.join(self.data_folder, 'exams_filtered.csv'))
        # set exam_id as index
        # remove trace_file, patient_id and nn_predicted_age

        self.tab_data['patient_id'] = self.tab_data.parallel_apply(lambda row: row['file_name'].split('/')[-1].split('_')[1], axis=1)
        self.unique_patients = self.tab_data['patient_id'].unique()
        self.patient_to_records = self.tab_data.groupby("patient_id")["file_name"].apply(list).to_dict()

        print("Tabular data fields for HEEDB: ", self.tab_data.head())

    
    def __getitem__(self, idx):
        patient = str(self.unique_patients[idx])
        records = self.patient_to_records[patient]

        # records = self.tab_data[self.tab_data['patient_id'] == int(patient)]['file_name'].tolist()
        num_views = self.n_global_view # + self.n_local_view
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
    

