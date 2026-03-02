import torch
import os
import pandas as pd
import wfdb
from dataset.pretraining_dataset import PretrainDataset
import numpy as np
from datetime import timedelta
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False, verbose=0)

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class ECGMIMICDataset(PretrainDataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None, downstream_task=None):
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = config.data_folder_mimic
        self.labels_file = os.path.join(config.data_folder_mimic, 'records_w_diag_icd10.csv')
        self.downstream_task = downstream_task
        self.label_list = config.label_list

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
        self.tab_data['file_name'] = self.tab_data.parallel_apply(lambda row:  str(row['file_name']).replace('mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/', ''), axis=1)

        if self.downstream_task == 'age':
            self.load_age_labels()
        elif self.downstream_task == 'lab':
            self.load_lab_labels(split=self.split)
        elif self.downstream_task == 'survival':
            self.load_survival_labels()

        # print the unique number of stratified folds
        # print(f'MIMIC-IV: number of unique folds {self.tab_data["fold"].nunique()}')

        print("MIMIC-IV: tabular data fields", self.tab_data.head())
        print(f'MIMIC-IV: colums {self.tab_data.columns}')

    def load_age_labels(self):
        # remove nan, negative and unrealistic ages
        initial_count = self.tab_data.shape[0]
        self.tab_data = self.tab_data[self.tab_data['age'].notna()]
        self.tab_data = self.tab_data[self.tab_data['age'] >= 0]
        self.tab_data = self.tab_data[self.tab_data['age'] <= 120]
        print(f'MIMIC-IV: filtered age records from {initial_count} to {self.tab_data.shape[0]}')

    def load_survival_labels(self):
        machine_measurement_path = os.path.join(self.data_folder, 'machine_measurements.csv')
        if not os.path.exists(machine_measurement_path):
            raise FileNotFoundError(f"MIMIC-IV: machine measurements file not found at {machine_measurement_path}")
        dat_ecg = pd.read_csv(machine_measurement_path)

        record_list_path = os.path.join(self.data_folder, 'record_list.csv')
        if not os.path.exists(record_list_path):
            raise FileNotFoundError(f"MIMIC-IV: record list file not found at {record_list_path}")
        dat_record = pd.read_csv(record_list_path)

        patients_path = os.path.join(self.data_folder, 'patients.csv.gz')
        if not os.path.exists(patients_path):
            raise FileNotFoundError(f"MIMIC-IV: patients file not found at {patients_path}")
        dat_patients = pd.read_csv(patients_path)
        dat_patients['dod'] = pd.to_datetime(dat_patients['dod'], errors='coerce')

        admission_path = os.path.join(self.data_folder, 'admissions.csv.gz')
        if not os.path.exists(admission_path):
            raise FileNotFoundError(f"MIMIC-IV: admissions file not found at {admission_path}")
        dat_admissions = pd.read_csv(admission_path)

        # the last discharge date
        max_disch_time = dat_admissions.sort_values(["subject_id",'dischtime']).groupby("subject_id").last().reset_index()
        max_disch_time = max_disch_time[['subject_id', 'dischtime']]
        max_disch_time.rename(columns={'dischtime': 'max_disch_time'}, inplace=True)
        max_disch_time = max_disch_time.reset_index()

        dat_ecg['ecg_time'] = pd.to_datetime(dat_ecg['ecg_time'], errors='coerce')  # Convert to datetime, handle errors
        dat_ecg['ecg_date'] = dat_ecg['ecg_time'].dt.date

        # the last ecg time
        max_ecg_time = dat_record.sort_values(by=['subject_id', 'ecg_time']).groupby('subject_id').last().reset_index()
        max_ecg_time.rename(columns={'ecg_time': 'max_ecg_time'}, inplace=True)
        max_ecg_time = max_ecg_time[['subject_id', 'max_ecg_time']]
        max_ecg_time = max_ecg_time.reset_index()

        dat = pd.merge(dat_ecg, dat_record, on=["subject_id", "study_id"])
        dat = dat.rename(columns={"ecg_time_y":"ecg_time", "ecg_date_y":"ecg_date"})
        dat = pd.merge(dat, dat_patients, how="inner", on="subject_id")
        dat = pd.merge(dat, max_ecg_time, how="left", on="subject_id")
        dat['ecg_time'] = pd.to_datetime(dat['ecg_time'], errors='coerce')  # Convert to datetime, handle errors
        dat = pd.merge(dat, max_disch_time, how="left", on="subject_id")
        dat['max_disch_time'] = pd.to_datetime(dat['max_disch_time'], errors='coerce')  # Convert to datetime, handle errors
        dat['max_ecg_time'] = pd.to_datetime(dat['max_ecg_time'], errors='coerce')  # Convert to datetime, handle errors
        dat["death"] = ~pd.isna(dat["dod"])

        dat['dod'] = pd.to_datetime(dat['dod'], errors='coerce')  # Convert to datetime, handle errors
        dat["timey"] = dat['dod'].dt.date - dat["ecg_time"].dt.date
        dat['timey'] = dat['timey'].fillna(dat["max_disch_time"].dt.date - dat["ecg_time"].dt.date + timedelta(days=365))
        dat['timey'] = dat['timey'].fillna(dat["max_ecg_time"].dt.date - dat["ecg_time"].dt.date)
        dat['timey'] = dat['timey'].dt.days / 365

        # fix file name to align with previous implementation
        dat['file_name'] = dat['path']

        # preserve fold column from the original tab_data
        dat = dat.merge(self.tab_data[['study_id', 'fold']], on='study_id', how='left')

        # get only ecgs with timey > 0
        self.tab_data = dat[dat["timey"].notna() & (dat["timey"] > 0)]
        print(self.tab_data.head())

    def load_lab_labels(self, split='train'):
        """
        Load the tabular data from a cached file.
        Args:
            path (str): Path to the cached file.
        """
        path_cache = os.path.join(self.data_folder, f'labevents_processed_{split}.csv')
        if not os.path.exists(path_cache):
            path_labevents = os.path.join(self.data_folder, 'labevents.csv.gz')
            if not os.path.exists(path_labevents):
                raise FileNotFoundError(f"Lab events file not found at {path_labevents}")
            labevent = pd.read_csv(path_labevents)

            path_lab_item = os.path.join(self.data_folder, 'd_labitems.csv.gz')
            if not os.path.exists(path_lab_item):
                raise FileNotFoundError(f"Lab items file not found at {path_lab_item}")
            labitem = pd.read_csv(path_lab_item)
            # keep only fluid == blood
            labitem = labitem[labitem['fluid'].str.lower() == 'blood']

            label_fds = []
            for label in self.label_list.keys():
                label_fds.append(labitem[labitem['label'] == label])

            item_ids_to_keep = pd.concat(label_fds).drop_duplicates()
            filtered_labevent = labevent[labevent['itemid'].isin(item_ids_to_keep['itemid'])]
            filtered_labevent = filtered_labevent.dropna(subset=['valuenum', 'charttime'])
            ref_ranges = filtered_labevent.groupby('itemid').agg({
                'ref_range_lower': 'median',
                'ref_range_upper': 'median',
                'valueuom': 'first',
                'valuenum': 'count'
            }).reset_index()
            # merge with event names
            ref_ranges = ref_ranges.merge(labitem[['itemid', 'label']], on='itemid', how='left')
            # order by label
            ref_ranges = ref_ranges.sort_values(by='label').reset_index(drop=True)
            ref_ranges.rename(columns={'label': 'item_label', 'valuenum': 'item_count'}, inplace=True)
            ref_ranges = ref_ranges.sort_values(by='item_count', ascending=False).drop_duplicates(subset=['item_label'])
            ref_ranges = ref_ranges.sort_values(by='item_label').reset_index(drop=True)
            # merge with filtered labevent
            filtered_labevent = filtered_labevent.merge(ref_ranges[['itemid', 'item_label']], on='itemid', how='left')
            filtered_labevent.dropna(subset=['valuenum', 'charttime'], inplace=True) 
            filtered_labevent = filtered_labevent[filtered_labevent['item_label'].notnull()]
            subject_set = set(filtered_labevent['subject_id']) & set(self.tab_data['subject_id'])
            
            lab_events = filtered_labevent[filtered_labevent['subject_id'].isin(subject_set)].copy()
            ecgs = self.tab_data[self.tab_data['subject_id'].isin(subject_set)].copy()

            lab_events['charttime'] = pd.to_datetime(lab_events['charttime'])
            ecgs['ecg_time'] = pd.to_datetime(ecgs['ecg_time'])

            lab_events.dropna(subset=['charttime'], inplace=True)
            ecgs.dropna(subset=['ecg_time'], inplace=True)

            #sort
            lab_events.sort_values(by='charttime', inplace=True)
            ecgs.sort_values(by='ecg_time', inplace=True)

            # Now merge_asof
            merged = pd.merge_asof(
                lab_events,  # LEFT: what you want to preserve (all lab events)
                ecgs,        # RIGHT: what you want to match to (ECGs)
                by='subject_id',
                left_on='charttime',   # Note: swapped these too
                right_on='ecg_time',   # Note: swapped these too
                direction='nearest',
                suffixes=('_lab', '_ecg')  # Explicitly control the suffixes
            )

            merged['time_diff'] = (merged['ecg_time'] - merged['charttime'])

            # there are multiple lab events of the same type for the same study_id, i want to keep only the closer in time_diff
            merged['time_diff_hrs'] = merged['time_diff'].dt.total_seconds() / 3600  # Convert to hours
            merged = merged.sort_values(by=['study_id', 'item_label', 'time_diff_hrs'])
            merged = merged.drop_duplicates(subset=['study_id', 'item_label'], keep='first')
            merged_1hr = merged[merged['time_diff_hrs'].abs() <= 1].copy()

            def encode_value(row):
                value = row['valuenum']
                lower = row['ref_range_lower']
                upper = row['ref_range_upper']
                
                if pd.isna(value) or pd.isna(lower) or pd.isna(upper):
                    return np.nan
                elif value < lower:
                    return -1
                elif value > upper:
                    return 1
                else:
                    return 0
                
            merged_1hr['encoded_value'] = merged_1hr.apply(encode_value, axis=1)
            study_info = merged_1hr.groupby('study_id').first()[['subject_id', 'ecg_time', 'charttime', 'file_name', 'fold']].reset_index()


            # Create pivot table with encoded values
            pivot_df = merged_1hr.pivot_table(
                index='study_id',
                columns='item_label',
                values='encoded_value',
                aggfunc='first'  # In case of duplicates, take first
            ).reset_index()

            result_df = study_info.merge(pivot_df, on='study_id', how='left')
            # save the result to a CSV file
            result_df.to_csv(path_cache, index=False)
            print(f'MIMIC-IV: saved merged tabular data to {path_cache}')
            self.tab_data = result_df
        else:
            print(f'MIMIC-IV: loading cached tabular data from {path_cache}')
            self.tab_data = pd.read_csv(path_cache)
            print(f'MIMIC-IV: loaded {len(self.tab_data)} records from cached file')
            print(f'MIMIC-IV: columns {self.tab_data.columns}')

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
        elif self.downstream_task == 'age':
            # for age prediction, we return the number of records
            return self.get_item_age(idx)
        elif self.downstream_task == 'lab':
            # for lab events prediction, we return the number of records
            return self.get_item_lab(idx)
        elif self.downstream_task == 'survival':
            # for survival prediction, we return the number of records
            return self.get_item_survival(idx)
        else:
            raise ValueError(f"Unknown downstream task: {self.downstream_task}")
        
    def get_signal(self, idx):
        record = self.tab_data.iloc[idx]['file_name']
        signal, info = wfdb.rdsamp(os.path.join(self.data_folder, record))

        # remove nans
        signal = np.nan_to_num(signal)

        signal = self.map_leads_and_clean(signal, info)
        signal = self.resample_if_needed(signal, info)

        if self.global_augmentations is not None:
            signal = self.global_augmentations(signal)
        return signal
    
    def get_item_age(self, idx):
        signal = self.get_signal(idx)
        age = self.tab_data.iloc[idx]['age']

        return {
            'signal': signal,
            'age': torch.tensor(age, dtype=torch.float32),
        }

    
    def get_item_lab(self, idx):
        signal = self.get_signal(idx)
        # extract the one hot encoding of the signal
        row = self.tab_data.iloc[idx]
        # print(f"Row for lab item {idx}: {row}")
        one_hot_values = one_hot_values = row[self.label_list.keys()].astype(float).fillna(-2)
        one_hot_values = one_hot_values + 1

        encoding = {
            -1.0: [-1, -1, -1],
            0.0: [ 1,  0,  0],
            1.0: [ 0,  1,  0],
            2.0: [ 0,  0,  1]
        }

        # Map and expand into a DataFrame
        encoded_df = pd.DataFrame(one_hot_values.map(encoding).to_list(),
                                index=one_hot_values.index,
                                columns=[i for i in range(3)])
        
        flattened_values = encoded_df.values.tolist()

        return {
            'signal': signal,
            'labels': torch.tensor(flattened_values, dtype=torch.float32),
        }
    
    def get_item_survival(self, idx):
        signal = self.get_signal(idx)

        timey = self.tab_data.iloc[idx]['timey']
        death = self.tab_data.iloc[idx]['death']

        return {
            'signal': signal,
            'timey': torch.tensor(timey, dtype=torch.float32),
            'death': torch.tensor(death, dtype=torch.bool),
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
    
