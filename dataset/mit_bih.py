import torch
import os
import pandas as pd
import wfdb
import neurokit2 as nk
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import neurokit2 as nk
from dataset.generic_utils import RandomSwitchtBaselineWanderBatched
from typing_extensions import override


leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
conversion = {
    'MLII' : 'II',
}

train = [101, 106, 108, 109, 112, 115, 116, 118, 119, 122, 124, 201, 205, 207, 208, 209, 215, 220, 223, 230]
test = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
val = [203, 114]

def convert_label(symbol):
    """
    Convert the symbols to the main class label
    """
    if symbol in ['N', 'L', 'R', 'e', 'j']:
        return 'N'
    elif symbol in ['A', 'a', 'J', 'S']:
        return 'S'  # Supraventricular ectopic
    elif symbol in ['V', 'E']:
        return 'V'  # Ventricular ectopic
    elif symbol in ['F']:
        return 'F'  # Fusion
    elif symbol in ['/', 'f', 'Q']:
        return 'Q'  # Unknown
    else:
        print(symbol)
        raise (f'Unknown symbol {symbol}')
    
valid_annotations = set(['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q'])

class ECGMITBIHDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train', augmentations=None):
        """
        Args:
            config: configuration object
            split: 'train', 'val'or 'test'
        """
        
        self.data_folder = config.data_folder_mit
        self.split = split
        self.samples = []
        self.nkclean = config.nk_clean
        self.patch_size = config.patch_size
        self.normalize = config.normalize
        self.num_classes = config.num_classes 
        self.win_len = config.win_len
        self.skip_majority_class_samples = config.skip_majority_class_samples
        self.bidirectional = config.bidirectional
        self.split_val_by_patient = config.split_val_by_patient
        self.augmentations = augmentations
        self.sampling_freq = config.sampling_freq
        self.leads_to_use = config.leads
        self.is_recurrent = config.is_recurrent 
        self.original_freq = 360
        self.freq_factor = self.sampling_freq / self.original_freq
        self.r_peaks_detection = config.r_peaks_detection
        print(f"freq_factor: {self.freq_factor}, sampling_freq: {self.sampling_freq}, original_freq: {self.original_freq}")

        self.load_patient_data(split)
        self.load_samples(split)


    def load_patient_data(self, subset):
        if self.split_val_by_patient:
            self.patients = train if subset == 'train' else val if subset == 'val' else test
        else:   
            self.patients = train + val if subset == 'train' else test if subset == 'test' else val

        self.headers = {}
        self.annotations = {}
        self.signals = {}
        self.r_peaks = {}
        self.labels = {}

        def process_patient(patient):
            signal, _ = wfdb.rdsamp(os.path.join(self.data_folder, 'raw', f'{patient}'))

            if self.nkclean:
                for i in range(signal.shape[1]):
                    signal[:, i] = nk.ecg_clean(signal[:, i], sampling_rate=360, method='neurokit')

            header = wfdb.rdheader(os.path.join(self.data_folder, 'raw', f'{patient}'))
            annotations = wfdb.rdann(os.path.join(self.data_folder + 'raw', f'{patient}'), 'atr')

            r_peaks = [(r_peak, convert_label(annotations.symbol[i])) for i, r_peak in enumerate(annotations.sample) if annotations.symbol[i] in valid_annotations]
            labels_orig = [label for label in annotations.symbol if label in valid_annotations]
            labels = [convert_label(l) for l in labels_orig]

            # filter classes
            if self.num_classes == 3:
                r_peaks = [(r_peak, label) for r_peak, label in r_peaks if label in ['N', 'S', 'V']]
                labels = [l for l in labels if l in ['N', 'S', 'V']]

            return patient, signal, header, annotations, r_peaks, labels

        results = Parallel(n_jobs=-1)(delayed(process_patient)(patient) for patient in self.patients)
        # results = [process_patient(patient) for patient in self.patients]

        for patient, signal, header, annotations, r_peaks, labels in results:
            if self.sampling_freq != header.fs:
                signal = nk.signal_resample(signal, sampling_rate=header.fs, desired_sampling_rate=self.sampling_freq, method='FFT')

            self.signals[patient] = signal
            self.headers[patient] = header
            self.annotations[patient] = annotations
            self.r_peaks[patient] = r_peaks
            self.labels[patient] = labels
            # map labels with r_peaks in a tuple

    def load_samples(self, subset):
        def process_sample(patient, r_peaks):
            samples = []
            last_class = None
            skipped = 0
            win_orig = self.win_len / self.freq_factor
            len_signal = len(self.signals[patient])
            # print((f"win_orig: {win_orig}, freq_factor: {freq_factor}, sampling_freq: {self.sampling_freq}, original_freq: {self.original_freq}"))

            if self.r_peaks_detection:
                # len signal and win_len are in the same frequency domain, r_peaks are not
                for i in range(0, len_signal, self.win_len * 2):
                    samples.append({
                        'start': i,
                        'end': min(i + self.win_len * 2, len_signal),
                        'patient': patient,
                        'r_peak': -1,
                        'around_r_peaks': [r for r, _ in r_peaks if i // self.freq_factor <= r < (i + self.win_len) // self.freq_factor],
                    })
                    # print(samples[-1]['around_r_peaks'])
                    # print around r_peaks
            elif subset == 'train' or not self.is_recurrent:
                for i, r_peak in enumerate(r_peaks):
                    sample_class = r_peaks[i][1]

                    if (sample_class != last_class or skipped > 10 or sample_class != 'N') or not self.skip_majority_class_samples:
                        samples.append({
                            'patient': patient,
                            'r_peak': r_peak[0],
                            'around_r_peaks': [(r, l) for r, l in r_peaks if r_peak[0] - win_orig < r <= r_peak[0] + win_orig],
                        })
                        skipped = 0
                    else:
                        skipped += 1

                    last_class = sample_class
            else:
                samples.append({
                    'patient': patient,
                    'r_peak': -1,
                    'around_r_peaks': r_peaks,
                })

            return samples

        results = Parallel(n_jobs=-1)(
            delayed(process_sample)(patient, self.r_peaks[patient]) for patient in self.patients
        )

        # Flatten results and reindex with unique keys
        self.samples = {i: sample for i, sample in enumerate(sum(results, []))}
        
    def __len__(self):
        return len(self.samples)
    
    def get_label_int(self, label):
        if label == 'N': return 0
        if label == 'S': return 1
        if label == 'V': return 2
        if label == 'F': return 3
        if label == 'Q': return 4
        else: raise ValueError(f'Unknown label {label}')

    def __getitem__(self, idx):
        if self.r_peaks_detection:
            return self.get_item_r_peaks(idx)
        else:
            return self.get_item_classification(idx)
        

    def get_item_r_peaks(self, idx):
        sample = self.samples[idx]
        patient = sample['patient']
        header = self.headers[patient]
        start = sample['start']
        end = sample['end']
        signal = self.signals[patient][start:end]
        signal = torch.tensor(signal, dtype=torch.float32)

        r_peaks_mask = torch.zeros(signal.shape[0], dtype=torch.float32)
        indexes = np.round(np.array(sample['around_r_peaks']) * self.freq_factor).astype(int) - start
        # print(f"indexes: {indexes}, start: {start}, freq_factor: {self.freq_factor}")
        r_peaks_mask[indexes] = 1
        signal = self.filter_leads(signal, header.__dict__['sig_name'])
        if self.augmentations is not None:
            signal = self.augmentations(signal)

        orig_start = np.round(start / self.freq_factor)
        around_r_peaks = torch.tensor([r - orig_start for r in sample['around_r_peaks']])

        return {
            'signal': signal,
            'patient_id': patient,
            'r_peak': r_peaks_mask,
            'r_peak_orig': around_r_peaks
        }

    def get_item_classification(self, idx):
        sample = self.samples[idx]
        patient = sample['patient']
        signal = torch.tensor(self.signals[patient], dtype=torch.float32)
        header = self.headers[patient]
        r_peak = int(sample['r_peak'] * self.sampling_freq / header.fs)
        # print("around_r_peaks", sample['around_r_peaks'])
        around_r_peaks = [(int(r * (self.sampling_freq / header.fs)), l) for r, l in sample['around_r_peaks']]
        # print(f"r_peak: {r_peak}, around_r_peaks: {around_r_peaks}")
        # print("around_r_peaks after resampling", around_r_peaks)
        len_signal = signal.shape[0]

        if self.split == 'train' or not self.is_recurrent:
            window_start = max(0, r_peak - self.win_len)
            window_end = min(r_peak + self.win_len, len_signal)
        else:
            window_start = 0
            window_end = len_signal

        # print(f"Window start: {window_start}, Window end: {window_end}, Signal length: {len_signal}")
        window_signal = signal[window_start:window_end]
        window_signal = self.filter_leads(window_signal, header.__dict__['sig_name'])

        if self.augmentations is not None:
            signal = self.augmentations(signal)
        
        r_peaks_mask = torch.zeros(window_signal.shape[0], dtype=torch.float32)

        # Use a list comprehension to filter and set the mask
        valid_r_peaks = [r - window_start for r, l in around_r_peaks if window_start <= r < window_end]
        r_peaks_mask[valid_r_peaks] = 1

        labels_mask = torch.zeros(window_signal.shape[0], dtype=torch.float32) - 1

        valid_labels = around_r_peaks

        for r, l in valid_labels:
            # print(r, l)
            if window_start <= r < window_end:
                labels_mask[r - window_start] = self.get_label_int(l)

        # start_original = window_start / self.freq_factor
        # original_r_peaks = torch.tensor([r - start_original for r, _ in sample['around_r_peaks']], dtype=torch.float32)
        # print(f"Original r_peaks: {original_r_peaks}")
        # print(f"Window signal shape: {window_signal.shape}, R peaks mask shape: {r_peaks_mask.shape}, Labels mask shape: {labels_mask.shape}")

        return {
            'signal': window_signal,
            'patient_id': patient,
            'r_peak': r_peaks_mask,
            'label': labels_mask,
            # 'r_peak_orig': original_r_peaks,
        }

    def filter_leads(self, signal, leads):
        # convert leads if needed 
        for i, lead in enumerate(leads):
            if lead in conversion.keys():
                leads[i] = conversion[lead]

        # print('leads', leads)

        signal_to_return = torch.zeros(signal.shape[0], len(self.leads_to_use), dtype=torch.float32)
        # if the leads to use are not present in the signal set them to zero
        # leads present in the signal that are not in the leads to use are removed
        # the rest is kept unchanged
        for i, lead in enumerate(self.leads_to_use):
            if lead not in leads:
                signal_to_return[:, i] = torch.zeros(signal.shape[0])
            else:
                signal_to_return[:, i] = signal[:, leads.index(lead)]
        return signal_to_return

class ECGMITBIHDatasetSingleHB(ECGMITBIHDataset):
    def __init__(self, config, split='train', augmentations=None):
        """
        Args:
            config: configuration object
            split: 'train', 'val'or 'test'
        """
        super().__init__(config, split, augmentations)

    @override
    def load_samples(self, subset):
        self.samples = []
        for patient in tqdm(self.patients, desc="Processing patients"):
            for i, r_peak in enumerate(self.r_peaks[patient]):
                signal = self.signals[patient]
                self.samples.append({
                    'patient': patient,
                    'r_peak': r_peak,
                    'signal': signal[max(0, r_peak[0] - 200): min(len(signal), r_peak[0] + 200)],
                })

    def __getitem__(self, idx):
        sample = self.samples[idx]
        patient = sample['patient']
        signal = torch.tensor(sample['signal'], dtype=torch.float32)
        r_peak = sample['r_peak'][0]
        header = self.headers[patient]

        signal = self.filter_leads(signal, header.__dict__['sig_name'])

        if self.augmentations is not None:
            signal = self.augmentations(signal)


        return {
            'signal': signal,
            'patient_id': patient,
            # 'r_peaks': torch.tensor([1.0] if r_peak >= 0 else [0.0], dtype=torch.float32),
            'label': torch.tensor([self.get_label_int(sample['r_peak'][1])], dtype=torch.float32) if r_peak >= 0 else torch.tensor([-1.0], dtype=torch.float32),
        }


def make_collate_fn(config, split='train'):

    if config.shuffle_baseline_wander_in_batch:
        baseline_shuffler = RandomSwitchtBaselineWanderBatched(config.sampling_freq, 0.5)
    
    def collate_fn(batch):
        signals = [item['signal'] for item in batch]
        patients = [item['patient_id'] for item in batch]
        
        if 'r_peak' not in batch[0].keys(): 
            r_peaks = None
        else:
            r_peaks = [item['r_peak'] for item in batch]
            r_peaks = torch.nn.utils.rnn.pad_sequence(r_peaks, batch_first=True)

        if 'label' not in batch[0].keys():
            labels = None
        else:
            labels = [item['label'] for item in batch]
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)


        if 'r_peak_orig' not in batch[0].keys():
            r_peaks_orig = None
        else:
            r_peaks_orig = [item['r_peak_orig'] for item in batch]
            r_peaks_orig = torch.nn.utils.rnn.pad_sequence(r_peaks_orig, batch_first=True, padding_value=torch.nan)

        # pad to same length and pad to match the patch size module
        if config.shuffle_baseline_wander_in_batch and split == 'train':
            signals = baseline_shuffler(torch.nn.utils.rnn.pad_sequence(signals, batch_first=True))
        else:
            signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)
            

        return {
            'signal': signals,
            'label': labels,
            'patient_ids': torch.tensor(patients),
            'r_peak': r_peaks,
            'labels': labels,
            'r_peak_orig': r_peaks_orig,
        }

    return collate_fn