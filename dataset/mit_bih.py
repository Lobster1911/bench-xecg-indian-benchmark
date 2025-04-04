import torch
import os
import pandas as pd
import wfdb
import neurokit2 as nk
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

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
    def __init__(self, config, subset='train', random_shift=False):
        """
        Args:
            config: configuration object
            subset: 'train' or 'test'
            name: name of the labels folder, default is 't_wave_split' so here the in the labels csv file the heartbeats will be divided by the t wave
        """
        
        self.data_folder = config.data_folder_mit
        self.subset = subset
        self.samples = []
        self.random_shift = random_shift
        self.nkclean = config.nk_clean
        self.patch_size = config.patch_size
        self.normalize = config.normalize
        self.name = config.name
        self.num_classes = config.num_classes 
        self.win_len = config.win_len
        self.skip_majority_class_samples = config.skip_majority_class_samples
        self.bidirectional = config.bidirectional

        self.leads_to_use = leads if config.leads == ['*'] else config.leads

        # self.samples = pd.read_csv(os.path.join(self.data_folder, self.name, f'labels_{subset}.csv'))
        # ensure no Nan values
        # self.samples['extra_annotations'] = self.samples['extra_annotations'].fillna('')

        # if config.num_classes == 3:
            # keep only the classes N, S and V
        #    self.samples = self.samples[self.samples['label'].isin(['N', 'S', 'V'])]
        
        # print(self.samples.head())  
        # get all the different values for column patient

        self.load_patient_data(subset)
        self.load_samples(subset)


    def load_patient_data(self, subset):
        self.patients = train if subset == 'train' else val if subset == 'val' else test
        self.headers = {}
        self.annotations = {}
        self.signals = {}
        self.r_peaks = {}
        self.labels = {}

        def process_patient(patient):
            if self.nkclean:
                signal, _ = wfdb.rdsamp(os.path.join(self.data_folder, self.name, f'{patient}'))
            else:
                signal, _ = wfdb.rdsamp(os.path.join(self.data_folder, 'raw', f'{patient}'))

            header = wfdb.rdheader(os.path.join(self.data_folder, 'raw', f'{patient}'))
            annotations = wfdb.rdann(os.path.join(self.data_folder + 'raw', f'{patient}'), 'atr')

            r_peaks = [(r_peak, convert_label(annotations.symbol[i])) for i, r_peak in enumerate(annotations.sample) if annotations.symbol[i] in valid_annotations]
            labels_orig = [label for label in annotations.symbol if label in valid_annotations]
            labels = [convert_label(l) for l in labels_orig]

            # filter classes
            if self.num_classes == 3:
                r_peaks = [(r_peak, label) for r_peak, label in r_peaks if label in ['N', 'S', 'V']]
                labels = [l for l in labels if l in ['N', 'S', 'V']]

            # (f'Patient {patient} has {len(r_peaks)} r peaks')

            return patient, signal, header, annotations, r_peaks, labels_orig, labels

        results = Parallel(n_jobs=-1)(delayed(process_patient)(patient) for patient in self.patients)

        for patient, signal, header, annotations, r_peaks, labels_orig, labels in results:
            self.signals[patient] = signal
            self.headers[patient] = header
            self.annotations[patient] = annotations
            self.r_peaks[patient] = r_peaks
            self.labels[patient] = labels
            # map labels with r_peaks in a tuple

    def load_samples(self, subset):
        def process_sample(patient, r_peaks, signal=None):
            samples = []
            last_class = None
            skipped = 0
            if subset == 'train':
                for i, r_peak in enumerate(r_peaks):
                    sample_class = r_peaks[i][1]
                    if (sample_class != last_class or skipped > 10 or sample_class != 'N') or not self.skip_majority_class_samples:
                        samples.append({
                            'patient': patient,
                            'r_peak': r_peak[0],
                            'around_r_peaks': [(r, l) for r, l in r_peaks if r_peak[0] - self.win_len < r <= r_peak[0] + self.win_len],
                        })
                        skipped = 0
                    else:
                        skipped += 1

                    last_class = sample_class
            else:
                for n in range(0, len(signal), self.win_len * 2):
                    around_r_peaks = [(r, l) for r, l in r_peaks if n - self.win_len < r <= n + self.win_len]
                    samples.append({
                        'patient': patient,
                        'r_peak': n,
                        'around_r_peaks': around_r_peaks,
                    })
            return samples

        results = Parallel(n_jobs=-1)(
            delayed(process_sample)(
                patient, self.r_peaks[patient], self.signals[patient] if subset != 'train' else None
            ) for patient in tqdm(self.patients, desc="Processing patients")
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
        sample = self.samples[idx]
        patient = sample['patient']
        signal = torch.tensor(self.signals[patient], dtype=torch.float32)
        header = self.headers[patient]
        r_peak = sample['r_peak']
        around_r_peaks = sample['around_r_peaks']

        if self.random_shift and self.subset == 'train':
            shift = torch.randint(- self.patch_size // 3, self.patch_size // 3, (1,)).item() # shift between 0 and patch_size // 3
            window_start = max(0, r_peak - self.win_len + shift)
            window_end = min(r_peak + self.win_len + shift, len(signal))
        else:
            window_start = max(0, r_peak - self.win_len)
            window_end = min(r_peak + self.win_len, len(signal))

        window_signal = signal[window_start:window_end]
        window_signal = self.filter_leads(window_signal, header.__dict__['sig_name'])

        if self.normalize:
            std = window_signal.std(axis=(0, -1))
            std[std == 0] = 1 # avoid division by zero, samples with std = 0 are all zero
            window_signal = (window_signal - window_signal.mean(axis=(0, -1))) / std
        
        r_peaks_mask = torch.zeros(window_signal.shape[0], dtype=torch.float32)

        # Use a list comprehension to filter and set the mask
        valid_r_peaks = [r - window_start for r, l in around_r_peaks if window_start <= r < window_end]
        r_peaks_mask[valid_r_peaks] = 1

        labels_mask = torch.zeros(window_signal.shape[0], dtype=torch.float32) -1
        # Use a list comprehension to filter and set the mask
        if self.bidirectional:
            valid_labels = around_r_peaks
        else:
            valid_labels = [(around_r_peaks[i + 1][0] - self.patch_size, around_r_peaks[i][1]) for i in range(len(around_r_peaks) -1)]
        for r, l in valid_labels:
            # print(r, l)
            if window_start <= r < window_end:
                labels_mask[r - window_start] = self.get_label_int(l)

        return {
            'signal': window_signal,
            'patient_id': patient,
            'r_peaks': r_peaks_mask,
            'label': labels_mask,
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


def collate_fn(batch):
    signals = [item['signal'] for item in batch]
    patients = [item['patient_id'] for item in batch]
    r_peaks = [item['r_peaks'] for item in batch]
    labels = [item['label'] for item in batch]

    # pad to same length and pad to match the patch size module

    signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)
    r_peaks = torch.nn.utils.rnn.pad_sequence(r_peaks, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)

    return {
        'signal': signals,
        'label': labels,
        'patient_ids': torch.tensor(patients),
        'r_peak': r_peaks,
        'labels': labels,
    }