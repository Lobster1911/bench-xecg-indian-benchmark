import torch
import os
import pandas as pd
import wfdb
from dataset.pretraining_dataset import PretrainDataset
import numpy as np
from pathlib import Path
import re

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class ECGHighIntensity(PretrainDataset):

    def __init__(self, config, split='train', global_augmentations=None):
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=None)
        self.data_folder = Path(config.data_folder_high_intensity) # /media/Volume/data/ECG_high_intensity_exercise
        self.info_dict = {"fs": 250, "sig_name": ["II"]}
        self.max_length_signal = config.max_length_signal

        self.load_records(split)
        self.load_tabular_data()

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

        if self.max_length_signal < 20:
            new_samples = []
            new_r_peaks = []
            for sample, r_peaks in zip(self.samples, self.r_peaks):
                # add a new sample for every max_length_signal
                for i in range(0, len(sample), self.max_length_signal * self.info_dict['fs']):
                    new_samples.append(sample[i:i + self.max_length_signal * self.info_dict['fs']])
                    new_r_peaks.append([r for r in r_peaks if i <= r < i + self.max_length_signal * self.info_dict['fs']])

            self.samples = np.array(new_samples)
            self.r_peaks = np.array(new_r_peaks)
                    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal = self.samples[idx]
        r_peaks = self.r_peaks[idx]

        signal = self.resample_if_needed(signal, self.info_dict)
        signal = self.map_leads_and_clean(signal, self.info_dict)

        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).float()

        if self.global_augmentations is not None:
            signal = self.global_augmentations(signal)

        obj = {
            'signal': signal,
            'r_peaks': r_peaks,
        }
        return obj


# if main
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config_file', type=str, default='configs/train_high_intensity_run_config.yaml', help='Path to the config file')
    args = parser.parse_args()

    from utils import parse_config
    config = parse_config(args.config_file, 'config_defaults/train_high_intensity_config_defaults.yaml')

    dataset = ECGHighIntensity(config, split='train')
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample: {dataset[0]}")