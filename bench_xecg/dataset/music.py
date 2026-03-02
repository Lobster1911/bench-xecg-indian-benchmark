import os

import torch
import numpy as np
import wfdb
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from .pretraining_dataset import PretrainDataset
from .generic_utils import pad


class MUSICDataset(PretrainDataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        """
        Args:
            records (list): List of records of ECG traces
        """
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = config.data_folder_music
        self.labels_file = config.labels_file_music

        music = pd.read_csv(self.labels_file, sep=';')
        music = music[[
            'Patient ID', 
            'Follow-up period from enrollment (days)', 
            'Cause of death', 'Exit of the study',
            'Age', 
            'Gender (male=1)', 
            'Weight (kg)', 
            'Height (cm)',  
            'Body Mass Index (Kg/m2)'
        ]]
        self.tab_data = music.rename(columns={
            'Follow-up period from enrollment (days)': 'timey', 
            'Cause of death': 'death', 
            'Exit of the study': 'exit',
            'Patient ID': 'subject_id',
            'Age': 'age',
            'Gender (male=1)': 'gender',
            'Weight (kg)': 'weight',
            'Height (cm)': 'height',
            'Body Mass Index (Kg/m2)': 'bmi'
        })

        self.load_records()

        # using cache to speed up and avoid continuous long loading times
        self._cached_read = lru_cache(maxsize=config.num_workers + 1)(self._read_signal)

    def _read_signal(self, subj):
        path = os.path.join(self.data_folder, 'Holter_ECG', subj)
        signal, info = wfdb.rdsamp(path)
        return signal, info

    def load_records(self):
        cached_path = os.path.join(self.data_folder, f'music_records_{self.max_length_signal}.csv')
        if os.path.exists(cached_path):
            print(f'Loading cached records from {cached_path}')
            records_df = pd.read_csv(cached_path)
            self.records = [(row['start'], row['end'], row['subject_id']) for _, row in records_df.iterrows()]
            return

        records_file = os.path.join(self.data_folder, 'RECORDS')
        # read lines of the file
        with open(records_file, 'r') as f:
            record_names = f.read().splitlines()
        
        # keep only the records starting with Holter_ECG
        record_names = [r for r in record_names if r.startswith('Holter_ECG/')]
        print('MUSIC:', self.tab_data.head())
        initial_pad = 10000
        print(f'Max signal lenghth in seconds: {self.max_length_signal / self.sampling_freq}')

        def process_subject(subj):
            # try to load the csv first

            record_path = os.path.join(self.data_folder, f"{subj}.dat")
            if not os.path.exists(record_path):
                raise FileNotFoundError(f"Record file {subj}.dat not found")

            _, info = wfdb.rdsamp(os.path.join(self.data_folder, subj))  # check if the record can be read
            sig_len_on_model = ((info['sig_len'] - initial_pad) / info['fs']) * self.sampling_freq
            subj = subj.split('/')[1]

            records = []
            if self.max_length_signal < sig_len_on_model:
                segment_len = int((self.max_length_signal / self.sampling_freq) * info['fs'])  # in samples
                records.extend((i, i + segment_len, subj) for i in range(initial_pad, info['sig_len'], segment_len))
                # drop the last, thus it will not have a 10 second duration
                records = records[:-1]
            else:
                records.append((0, info['sig_len'], subj))
            return records

        self.records = []
        with ThreadPoolExecutor() as executor:
            for recs in tqdm(executor.map(process_subject, record_names), total=len(record_names)):
                self.records.extend(recs)

        # save records as csv
        records_df = pd.DataFrame(self.records, columns=['start', 'end', 'subject_id'])
        records_df.to_csv(cached_path, index=False)
        print(f'Saved cached records to {cached_path}')

    def __getitem__(self, idx):
        start, end, subj = self.records[idx]
        signal, info = self._cached_read(subj)
        # signal, info = wfdb.rdsamp(os.path.join(self.data_folder, 'Holter_ECG', subj))  # check if the record can be read

        # print(f'loaded {subj} with shape {signal.shape} from {start} to {end}')

        signal = torch.tensor(signal[start:end, :], dtype=torch.float32)  # crop the signal
        signal = self.resample_if_needed(signal, info)

        # print(f'resampled {subj} with shape {signal.shape}')

        signal, sig_names = xyz_to_12lead(signal, info['sig_name'])
        # print(f'converted {subj} to 12 leads with shape {signal.shape}')

        signal = self.map_leads_and_clean(signal, {'sig_name': sig_names, 'fs': info['fs']})
        # print(f'mapped {subj} to standard leads with shape {signal.shape}')

        # get timey and death
        tab_row = self.tab_data[self.tab_data['subject_id'] == subj]
        timey = torch.tensor(tab_row['timey'].values[0] / 365.5, dtype=torch.float32)
        death = torch.tensor(tab_row['death'].values[0] != 0, dtype=torch.float32)
        cardiac_death = torch.tensor(tab_row['death'].values[0] in [3, 6, 7], dtype=torch.float32)

        # print(f'got labels for {subj}: timey={timey}, death={death}, cardiac_death={cardiac_death}')

        return {
            'signal': signal,
            'subj': subj,
            'timey': timey,
            'death': death,
            'cardiac_death': cardiac_death
        }
    
    def map_xyz_to_leads(self, signal):
        # map the 3 leads to I, II, III
        lead_I = signal[:, 0]
        lead_II = signal[:, 1]
        lead_III = lead_II - lead_I

        return torch.stack([lead_I, lead_II, lead_III], dim=1)
    
def make_collate_fn(config):
    def collate_fn(batch):
        signals = [item['signal'] for item in batch]
        subjs = [item['subj'] for item in batch]
        timeys = torch.tensor([item['timey'] for item in batch], dtype=torch.float32)
        deaths = torch.tensor([item['death'] for item in batch], dtype=torch.float32)
        cardiac_deaths = torch.tensor([item['cardiac_death'] for item in batch], dtype=torch.float32)

        signals = pad(torch.nn.utils.rnn.pad_sequence([torch.from_numpy(sig.copy()) for sig in signals], batch_first=True).float(), patch_size=config.patch_size)
            
        return {
            'signals': signals,
            'subj': subjs,
            'timey': timeys,
            'death': deaths,
            'cardiac_death': cardiac_deaths
        }
    return collate_fn
   

def xyz_to_12lead(data, sig_name, method="kors"):
    """
    Convert orthogonal XYZ signals into 12-lead ECG using Kors or inverse Dower.

    Parameters
    ----------
    data : ndarray
        Array of shape [seq_len, channels] containing the signals.
    sig_name : list
        List of channel names, must include 'X', 'Y', 'Z'.
    method : str
        'kors' or 'dower'.

    Returns
    -------
    ecg : ndarray
        Array of shape [seq_len, 12] with reconstructed ECG leads.
    lead_names : list
        Names of the 12 standard leads.
    """

    # find indices for X, Y, Z
    ix = sig_name.index("X")
    X = data[:, ix]
    iy = sig_name.index("Y")
    Y = data[:, iy]

    if "Z" not in sig_name:
        Z = np.zeros_like(X)
    else:
        iz = sig_name.index("Z")
        Z = data[:, iz]

    if method.lower() == "kors":
        M = np.array([
            [-0.130,  0.060, -0.430],  # V1
            [ 0.050, -0.020, -0.060],  # V2
            [-0.010, -0.050, -0.140],  # V3
            [ 0.140,  0.060, -0.200],  # V4
            [ 0.060, -0.170, -0.110],  # V5
            [ 0.540,  0.130,  0.310],  # V6
            [ 0.380, -0.070,  0.110],  # I
            [-0.070,  0.930, -0.230],  # II
        ])
        M = np.linalg.pinv(M @ M.T) @ M  # convert to regression form

    elif method.lower() == "dower":
        M = np.array([
            [-0.172,  0.057, -0.229],  # V1
            [-0.074, -0.019, -0.310],  # V2
            [ 0.122, -0.106, -0.246],  # V3
            [ 0.231, -0.022, -0.063],  # V4
            [ 0.239,  0.041,  0.055],  # V5
            [ 0.194,  0.048,  0.108],  # V6
            [ 0.156, -0.227,  0.022],  # I
            [-0.010,  0.887,  0.102],  # II
        ])
        M = np.linalg.pinv(M @ M.T) @ M  # convert to regression form
    else:
        raise ValueError("method must be 'kors' or 'dower'")

    XYZ = np.vstack([X, Y, Z])  # shape (3, seq_len)
    V1, V2, V3, V4, V5, V6, I, II = M @ XYZ

    III = II - I
    aVR = -(I + II) / 2.0
    aVL = I - II / 2.0
    aVF = II - I / 2.0

    ecg = np.vstack([I, II, III, aVR, aVL, aVF,
                     V1, V2, V3, V4, V5, V6]).T

    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]

    return ecg, lead_names
