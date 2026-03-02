import os

import torch
import numpy as np
import wfdb
import pandas as pd
from pandarallel import pandarallel

from .pretraining_dataset import PretrainDataset

pandarallel.initialize(progress_bar=False, verbose=0)

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
            'records': records,
        }
    

class ECGHEEDBMortalityDataset(ECGHEEDBDataset):
    def __init__(self, config, global_augmentations=None, local_augmentations=None):
        """
        Args:
            records (list): List of records of ECG traces
        """
        super().__init__(config, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        # load metadata
        self.load_metadata()

    def load_metadata(self):
        i0001_metadata_file = os.path.join(self.data_folder, 'I0001', 'metadata', 'metadata.csv')
        i0006_metadata_file = os.path.join(self.data_folder, 'I0006', 'metadata', 'metadata.csv')

        metadata_df = pd.concat([
            self.load_metadata_df(i0001_metadata_file),
            self.load_metadata_df(i0006_metadata_file)
        ], ignore_index=True)

        metadata_df = metadata_df[['BDSPPatientID', 'FileID', 'timey', 'death']]


        # keep in tab data only the records with metadata
        self.tab_data = self.tab_data[self.tab_data['patient_id'].isin(metadata_df['BDSPPatientID'].astype(int).astype(str).values)]

        # merge metadata with tab_data on file name
        self.tab_data['file_id'] = self.tab_data.parallel_apply(lambda row: str(row['file_name'].split('/')[-1]), axis=1)
        self.tab_data = pd.merge(self.tab_data, metadata_df, left_on='file_id', right_on='FileID')

        self.unique_patients = self.tab_data['patient_id'].unique()
        self.patient_to_records = self.tab_data.groupby("patient_id")["file_name"].apply(list).to_dict()

        self.tab_data = self.tab_data.set_index('file_name')

        print('HEEDB: unique patients:', len(self.unique_patients))

    def load_metadata_df(self, meta_path):
        # Define optimal dtypes for memory efficiency
        optimal_dtypes = {
            'BDSPPatientID': np.float64,
            'FileName': 'object',
            'FileID': 'object',

            # Categorical/Low-Cardinality Strings (saves massive memory)
            'PatientRace': 'category',
            'EthnicGroupDSC': 'category',
            'MaritalStatusDSC': 'category',
            'ReligionDSC': 'category',
            'LanguageDSC': 'category',
            'VeteranStatusDSC': 'category',
            'SexDSC': 'category',
            'Sex': 'category',
            'EducationLevelDSC': 'category',
            'GenderIdentityDSC': 'category',
            'SexAssignedAtBirthDSC': 'category',

            # String Columns (likely high cardinality)
            'PrimaryCauseOfDeathDSC': 'object',
            'PrimaryCauseOfDeathUNOS': 'object',
            'FirstContributoryCauseOfDeathDSC': 'object',
            'FirstContributoryCauseOfDeathUNOS': 'object',
            'SecondContributoryCauseOfDeathDSC': 'object',
            'SecondContributoryCauseOfDeathUNOS': 'object',

            # Numeric Columns (Downcast from float64 to float32)
            'AgeAtAcquisition': np.float32,
            'AgeAtDeath': np.float32,
            'AgeAtDeathMA': np.float32,
            'AgeAtLastVisit': np.float32,

            # Date/Time Columns (Load as string, convert after loading for speed)
            'DateOfDeath': 'object',
            'DateOfDeathMARegistryData': 'object',
            'LastKnownVisitDate': 'object',
            'ECGAcquisitionTime': 'object',
            'DateOfBirth': 'object'
        }
        df = pd.read_csv(
            meta_path,
            dtype=optimal_dtypes,
            engine='c',
        )

        date_columns = [
            'DateOfDeath', 'DateOfDeathMARegistryData', 'LastKnownVisitDate',
            'ECGAcquisitionTime', 'DateOfBirth'
        ]

        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')  # 'coerce' converts invalid dates to NaT

        if "Sex" not in df.columns:
            df["Sex"] = df["SexDSC"]

        df["death"] = ~df.DateOfDeath.isnull()

        df["censor_date"] = df["LastKnownVisitDate"]
        df.loc[df.death, "censor_date"] = df.loc[df.death, "DateOfDeath"]
        df["time_to_event"] = df["censor_date"] - df["ECGAcquisitionTime"]
        df["time_to_event_days"] = df["time_to_event"].dt.days
        df["time_to_event_years"] = df["time_to_event_days"] / 365
        df['timey'] = df['time_to_event_years']

        df = df.dropna(subset=['BDSPPatientID', 'timey', 'death'])

        return df
    
    def __getitem__(self, idx):
        obj = super().__getitem__(idx)
        record = obj['records'][0]

        # KeyError: (1007376, 'I0001/WFDB/S0003/2009/08/de_120872262_20070704215403_18990731000000')
        row = self.tab_data.loc[record]

        return {
            'signal': obj['global_signals'][0],
            'death': torch.tensor(row['death'], dtype=torch.float32),
            'timey': torch.tensor(row['timey'], dtype=torch.float32)
        }