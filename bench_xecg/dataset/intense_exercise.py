from pathlib import Path
import re

import numpy as np
import torch
import pandas as pd

from .pretraining_dataset import PretrainDataset

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class ECGHighIntensity(PretrainDataset):

    def __init__(self, config, split='train', global_augmentations=None):
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=None)
        self.data_folder = Path(config.data_folder_high_intensity) # /media/Volume/data/ECG_high_intensity_exercise
        self.info_dict = {"fs": 250, "sig_name": ["II"]}

        self.load_records(split)
        self.load_tabular_data()
        self.reformat_data()

    def load_records(self, split):
        segments = self.data_folder / 'ecg_segments'
        # list all the files in the directory
        files = list(segments.glob('*.csv'))

        # open each file
        list_of_dataframes = []
        for file in files:
            if file.is_file():
                if '_raw_' in file.name:
                    continue
                df = pd.read_csv(file, header=None)
                # rename the column 0 to the file name without extension
                df.columns = [file.stem, '']
                list_of_dataframes.append(df)

        # concatenate all dataframes into one where each column is a row instead
        df = pd.concat(list_of_dataframes, axis=1)
        df = df / 1000
        # remove columns with all NaN values

        # order columns by name
        df = df.dropna(axis=1, how='all').T

        pattern = re.compile(r'(?<=sub)\d+')

        # filter for training set
        if split == 'train':
            # filter for subject IDs that are greater than 13
            subject_ids = [f for f in df.index if int(pattern.search(f).group()) <= 9]
        elif split == 'val':
            # filter for subject IDs that are greater than 9 and less than or equal to 13
            subject_ids = [f for f in df.index if 9 < int(pattern.search(f).group()) <= 13]
        elif split == 'test':
            # filter for subject IDs that are greater than 13
            subject_ids = [f for f in df.index if int(pattern.search(f).group()) > 13]

        # keep only these subject IDs
        df = df.loc[subject_ids]
        # convert the dataframe to a numpy array
        self.samples = df


    def load_tabular_data(self):
        manual_annotations = self.data_folder / 'manual_annotations'

        columns = self.samples.index.tolist()

        r_peaks = []
        # for each column get the annotations of r-peaks on manual annotation
        for column in columns:
            # get the file name
            file_name = column + '_labels.csv'
            # read the manual annotations
            manual_annotation_file = manual_annotations / file_name
            if manual_annotation_file.is_file():
                manual_df = pd.read_csv(manual_annotation_file, header=None)
                # set the column name
                manual_df.columns = [file_name, 1, 2]
                # remove the second column
                # get the r-peaks
                r_peaks.append(manual_df[file_name])
            else:
                print(f'File: {file_name} not found in manual annotations')

        # create a new dataframe with the r-peaks
        self.r_peaks = pd.DataFrame(r_peaks)

    def reformat_data(self):
        self.samples = self.samples.to_numpy() # [patients, len]
        self.r_peaks = self.r_peaks.to_numpy() # [patients, len]

        # resample all the signals to the model sampling frequency
        if self.info_dict['fs'] != self.sampling_freq:
            self.samples = np.array([self.resample_if_needed(signal, self.info_dict) for signal in self.samples])


        if self.max_length_signal < self.samples.shape[1]:
            new_samples = []
            new_r_peaks = []

            freq_factor = self.sampling_freq / self.info_dict['fs']
            print(f"Resampling factor: {freq_factor}")
            for sample, r_peaks in zip(self.samples, self.r_peaks):
                # add a new sample for every max_length_signal
                for i in range(0, len(sample), self.max_length_signal):
                    new_samples.append(sample[i:i + self.max_length_signal])
                    i_orig = i / freq_factor
                    new_r_peaks.append([r - i_orig for r in r_peaks if i_orig <= r <= (i_orig + self.max_length_signal / freq_factor)])
                    # print the shapes of the new samples and r_peaks    list_r_peaks = torch.tensor([int(r) / self.orig_freq for r in r_peaks_orig[i] if not torch.isnan(r)]).to(preds.device)

                    # print(new_samples[-1].shape, len(new_r_peaks[-1]))

            # pad with zeros if the last sample is shorter than max_length_signal
            self.samples = np.array([
                np.pad(arr, (0, self.max_length_signal - len(arr)), constant_values=0)
                for arr in new_samples
            ])

            max_len = max(len(arr) for arr in new_r_peaks)
            self.r_peaks  = np.array([
                np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
                for arr in new_r_peaks
            ])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal = self.samples[idx]
        r_peaks = self.r_peaks[idx]
        r_peaks_orig_tensor = torch.tensor(r_peaks, dtype=torch.float32)

        # signal = self.resample_if_needed(signal, self.info_dict)
        signal = self.map_leads_and_clean(np.expand_dims(signal, axis=-1), self.info_dict)

        # get a tensor of the same shape of the signal where the r_peaks indexes are set to 1
        r_peaks = r_peaks * self.sampling_freq / self.info_dict['fs']
        r_peaks = [int(r) for r in r_peaks if not pd.isna(r)]
        r_peaks_tensor = torch.zeros(len(signal), dtype=torch.float32)

        # if one r-peak is at len(signal) do -1
        if r_peaks and r_peaks[-1] == len(signal):
            r_peaks[-1] -= 1
 
        r_peaks_tensor[r_peaks] = 1.0


        if self.global_augmentations is not None:
            signal = self.global_augmentations(signal)

        # print(f"Signal shape: {signal.shape}, R-peaks shape: {r_peaks_tensor.shape}")

        obj = {
            'signals': torch.tensor(signal.copy()).float(),
            'r_peak': r_peaks_tensor,
            'r_peak_orig': r_peaks_orig_tensor,
        }

        return obj